# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
from einops import rearrange
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2

class BlockGPUManager:
    def __init__(self, device="cuda",):
        self.device = device
        self.managed_modules = []
        self.embedder_modules = []
        self.output_modules = []  
        # 跟踪哪些blocks当前在GPU上
        self.block_types = {} 
        self.blocks_on_gpu = set()
    
    def get_gpu_memory_usage(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
            free = total - allocated
            
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'total_mb': total,
                'free_mb': free
            }
        return None
    
    def setup_for_inference(self, transformer_model,):
        self._collect_managed_modules(transformer_model)
        self._initialize_embedder_modules()
        return self


    def _collect_managed_modules(self, transformer_model):
        self.managed_modules = []
        self.managed_motion_modules = []
        self.managed_motion_proj_modules=[]
        self.embedder_modules = []

        for i, block in enumerate(transformer_model.blocks):
            self.managed_modules.append(block)

        for i, block in enumerate(transformer_model.motion_blocks):
            self.managed_motion_modules.append(block) 
        
        for i , block in enumerate(transformer_model.zero_motion_proj_blocks):
            self.managed_motion_proj_modules.append(block) 


        if hasattr(transformer_model, 'time_embedding'):#
            self.embedder_modules.append(transformer_model.time_embedding)
        
        if hasattr(transformer_model, 'time_projection'):#
            self.embedder_modules.append(transformer_model.time_projection)
        
        if hasattr(transformer_model, 'text_embedding'):#
            self.embedder_modules.append(transformer_model.text_embedding)
        
        if hasattr(transformer_model, 'motion_time_embedding'):
            self.embedder_modules.append(transformer_model.motion_time_embedding)
        
        if hasattr(transformer_model, 'patch_embedding'): #
            self.embedder_modules.append(transformer_model.patch_embedding)
        
        if hasattr(transformer_model, 'head'):#
            self.embedder_modules.append(transformer_model.head)

        if hasattr(transformer_model, 'motion_patch_embedding'):
            self.embedder_modules.append(transformer_model.motion_patch_embedding)

        if hasattr(transformer_model, 'motion_text_embedding'):
            self.embedder_modules.append(transformer_model.motion_text_embedding)

        if hasattr(transformer_model, 'motion_time_projection'):
            self.embedder_modules.append(transformer_model.motion_time_projection)

        if hasattr(transformer_model, 'motion_head'):
            self.embedder_modules.append(transformer_model.motion_head)

        if hasattr(transformer_model, 'audio_proj'):
            self.embedder_modules.append(transformer_model.audio_proj)

  
    def _initialize_embedder_modules(self):
        for module in self.embedder_modules:
            if hasattr(module, 'to'):
                module.to(self.device, non_blocking=True)
        return self


    def unload_all_blocks_to_cpu(self):
        for module in self.managed_modules:
            if hasattr(module, 'to'):
                module.to('cpu')
        
        for module in self.managed_motion_modules:
            if hasattr(module, 'to'):
                module.to('cpu')

        for module in self.managed_motion_proj_modules:
            if hasattr(module, 'to'):
                module.to('cpu')

        for module in self.embedder_modules:
            if hasattr(module, 'to'):
                module.to('cpu')
        
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

def rope_params_cond(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    steps = torch.arange(0, max_seq_len-1, dtype=torch.float64)
    steps = torch.cat([steps, torch.tensor([-1.0], dtype=torch.float64)])
    freqs = torch.outer(
        steps,
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs
@torch.amp.autocast('cuda', enabled=False)
def rope_apply_1d(x, freqs):
    """
    x: [B, S, H, C]
    freqs: [S, C//2]  # complex
    """
    B, S, H, C = x.shape
    freqs=freqs[:S,:]
    assert C % 2 == 0
    x_ = x.float().reshape(B, S, H, C//2, 2)
    x_complex = torch.view_as_complex(x_)  # [B, S, H, C//2]
    # freqs: [S, C//2]，需要广播到 [B, S, H, C//2]
    #print(x_complex.shape,freqs.shape,freqs[None, :, None, :].shape)
    x_out = x_complex * freqs[None, :, None, :]
    x_out = torch.view_as_real(x_out).reshape(B, S, H, C)
    return x_out.type_as(x)


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    对视频特征x应用3D旋转位置编码 (RoPE)，并为前置的条件帧提供特殊处理。

    Args:
        x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, n, c*2)。
                          序列维度(seq_len)应由一个尺寸为 (1*h*w) 的条件帧
                          和尺寸为 (f*h*w) 的视频帧组成。
        grid_sizes (list or tuple): 包含视频部分网格尺寸 [f, h, w] 的列表。
        freqs (torch.Tensor): 预先计算好的旋转频率。

    Returns:
        torch.Tensor: 应用了RoPE的张量，数据类型与输入x相同。
    """
    n, c = x.size(2), x.size(3) // 2

    # 假设批处理中所有样本的网格大小都相同，与原始代码行为保持一致。
    t, h, w = grid_sizes[0]
    f = t - 1
    
    # 将频率张量按 f, h, w 三个维度进行切分
    freqs_f, freqs_h, freqs_w = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # 检查频率张量的长度是否足以支持偏移编码
    if freqs_h.size(0) < 2 * h or freqs_w.size(0) < 2 * w:
        raise ValueError(
            f"用于h和w的freqs张量长度不足以进行偏移编码。"
            f"高度至少需要 {2*h}，宽度至少需要 {2*w}，"
            f"但实际得到 {freqs_h.size(0)} 和 {freqs_w.size(0)}。"
        )
    # 至少需要位置0和1的频率才能通过共轭计算出-1的位置编码
    if freqs_f.size(0) < 2:
        raise ValueError("用于f的freqs张量尺寸必须至少为2。")

    # 对批处理中的每个样本进行处理
    output = []
    for i in range(x.size(0)):
        # 定义序列中不同部分的长度
        cond_len = h * w
        video_len = f * h * w
        
        # 分离输入张量的不同部分
        x_sample = x[i]
        x_cond = x_sample[-cond_len:]
        x_video = x_sample[:video_len]
        x_rest = x_sample[video_len + cond_len:]

        # --- 1. 处理条件帧 ---
        # 转换成复数形式
        x_cond_complex = torch.view_as_complex(
            x_cond.to(torch.float64).reshape(cond_len, n, -1, 2)
        )

        # 为条件帧创建位置编码 (f=-1, h=[h,2h), w=[w,2w))
        # 对于 f=-1: 位置-1的RoPE是位置1的RoPE的复共轭
        freq_f_cond = freqs_f[-1].view(1, 1, 1, -1).expand(1, h, w, -1)
        # freq_h_cond = freqs_h[h : 2 * h].view(1, h, 1, -1).expand(1, h, w, -1)
        # freq_w_cond = freqs_w[w : 2 * w].view(1, 1, w, -1).expand(1, h, w, -1)
        freq_h_cond = freqs_h[:h].view(1, h, 1, -1).expand(1, h, w, -1)
        freq_w_cond = freqs_w[:w].view(1, 1, w, -1).expand(1, h, w, -1)

        # 组合三个维度的频率并调整形状
        freqs_cond = torch.cat([freq_f_cond, freq_h_cond, freq_w_cond], dim=-1).reshape(
            cond_len, 1, -1
        )
        
        # 应用旋转位置编码
        x_cond_rotated = torch.view_as_real(x_cond_complex * freqs_cond).flatten(2)

        # --- 2. 处理视频帧 (原始逻辑) ---
        # 转换成复数形式
        x_video_complex = torch.view_as_complex(
            x_video.to(torch.float64).reshape(video_len, n, -1, 2)
        )
        
        # 为视频帧创建位置编码 (f=[0,f), h=[0,h), w=[0,w))
        freq_f_video = freqs_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1)
        freq_h_video = freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1)
        freq_w_video = freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1)

        # 组合三个维度的频率并调整形状
        freqs_video = torch.cat([freq_f_video, freq_h_video, freq_w_video], dim=-1).reshape(
            video_len, 1, -1
        )

        # 应用旋转位置编码
        x_video_rotated = torch.view_as_real(x_video_complex * freqs_video).flatten(2)

        # --- 3. 组合所有部分 ---
        x_i_processed = torch.cat([x_video_rotated, x_cond_rotated, x_rest], dim=0)
        output.append(x_i_processed)

    return torch.stack(output).to(x.dtype)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # print(x.dtype)
        return super().forward(x).type_as(x)


class WanMMAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 single_block=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.single_block=single_block
        if not single_block:
            self.q2 = nn.Linear(dim, dim)
            self.k2 = nn.Linear(dim, dim)
            self.v2 = nn.Linear(dim, dim)
            self.o2 = nn.Linear(dim, dim)
            self.norm_q2 = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
            self.norm_k2 = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()





    def forward(self, x,x_cond, seq_lens, grid_sizes, freqs,freqs_cond):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        if self.single_block:
            q2_f = self.q
            k2_f = self.k
            v2_f = self.v
            o2 = self.o
            norm_q2 = self.norm_q
            norm_k2 = self.norm_k
        else:
            q2_f = self.q2
            k2_f = self.k2
            v2_f = self.v2
            o2 = self.o2
            norm_q2 = self.norm_q2
            norm_k2 = self.norm_k2
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        s_cond=x_cond.shape[1]
        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, -1, n, d)
            k = self.norm_k(self.k(x)).view(b, -1, n, d)
            v = self.v(x).view(b, -1, n, d)
            return q, k, v

        def qkv_fn2(x_cond):
            q = norm_q2(q2_f(x_cond)).view(b, -1, n, d)
            k = norm_k2(k2_f(x_cond)).view(b, -1, n, d)
            v = v2_f(x_cond).view(b, -1, n, d)
            return q, k, v


        q, k, v = qkv_fn(x)
        q = rope_apply_1d(q, freqs)
        k = rope_apply_1d(k, freqs)

        # if self.single_block:
        #     q2, k2, v2 = qkv_fn(x_cond)
        # else:
        q2, k2, v2 = qkv_fn2(x_cond)

        q2 = rope_apply_1d(q2, freqs_cond)
        k2 = rope_apply_1d(k2, freqs_cond)
        q = torch.cat([q2, q], dim=1)
        k = torch.cat([k2, k], dim=1)
        v = torch.cat([v2, v], dim=1)

        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)

        x_cond,x=x[:,:s_cond],x[:,s_cond:]
        x = self.o(x)

        x_cond = o2(x_cond)
        # x_cond=self.o(x_cond) if self.single_block else self.o2(x_cond)
        return x,x_cond


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs) if grid_sizes is not None else rope_apply_1d(q,freqs),
            k=rope_apply(k, grid_sizes, freqs) if grid_sizes is not None else rope_apply_1d(k,freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)


        # output
        x = x.flatten(2)
        x = self.o(x)
        return x



class WanAttentionBlock_Audio(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 encoder_hidden_states_dim=768,
                 audio_block=False,
                 single_block=False,
                 ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.single_block = single_block
        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.norm2 = WanLayerNorm(dim, eps)

        self.mm_attn = WanMMAttention(dim, num_heads, window_size, qk_norm,
                                          eps,single_block)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

        if not self.single_block:
            self.ffn_cond = nn.Sequential(
                nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
                nn.Linear(ffn_dim, dim))
            self.norm_cond1 = WanLayerNorm(dim, eps)
            self.norm_cond2 = WanLayerNorm(dim, eps)
            self.modulation_cond = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
        # init audio module
        #self.norm_x = WanLayerNorm(dim, eps, elementwise_affine=True)

    def forward(
            self,
            x,
            x_cond,
            e,
            e_cond,
            seq_lens,
            grid_sizes,
            freqs,
            freqs_cond,
            context_lens,
            ref_target_masks=None,
            audio_branch=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        if self.single_block:
            norm_cond1 = self.norm1
            norm_cond2 = self.norm2
            ffn_cond = self.ffn
            modulation_cond = self.modulation
        else:
            norm_cond1 = self.norm_cond1
            norm_cond2 = self.norm_cond2
            ffn_cond = self.ffn_cond
            modulation_cond = self.modulation_cond

        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        assert e_cond.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e_cond = (modulation_cond.unsqueeze(1) + e_cond).chunk(6, dim=2)
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        #     e_cond = (modulation_cond + e_cond).chunk(6, dim=1)
        assert e_cond[0].dtype == torch.float32

        # self-attention
        e_2 = [e[i].squeeze(2).to(dtype=x.dtype, device=x.device) for i in range(6)]
        e_cond2 = [e_cond[i].squeeze(2).to(dtype=x.dtype, device=x.device) for i in range(6)]
        x_cond2 = norm_cond1(x_cond)
        x_cond2 = x_cond2 * (1 + e_cond2[1]) + e_cond2[0]

        xx = self.norm1(x)
        xx = xx * (1 + e_2[1]) + e_2[0]

        yy,y_cond = self.mm_attn(
            xx, x_cond2,seq_lens, grid_sizes,
            freqs,freqs_cond)

        x = x + yy * e_2[2]

        x_cond=x_cond+y_cond*e_cond2[2]

        y = self.ffn(self.norm2(x) * (1 + e_2[4]) + e_2[3])

        y_cond = ffn_cond(norm_cond2(x_cond) * (1 + e_cond2[4]) + e_cond2[3])

        with torch.amp.autocast('cuda', dtype=e_2[0].dtype):
            x = x + y * e_2[5]
            x_cond = x_cond + y_cond * e_cond2[5]
        return x,x_cond


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, freqs_q=None, frecs_k=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        if freqs_q is not None:
            q = rope_apply_1d(q, freqs_q)
        if frecs_k is not None:
            k = rope_apply_1d(k, frecs_k)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class SingleStreamAttention(nn.Module):
    """
    Audio-video cross-attention module for talking avatar
    Extracted from model_audio.py and adapted for Wan2.2 architecture
    FIXED: Proper frame-by-frame audio-video alignment
    """
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
        norm_layer: nn.Module = WanRMSNorm,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qk_norm = qk_norm

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)

        self.add_q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.add_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor, shape=None) -> torch.Tensor:
        """
        FIXED: Proper frame-by-frame audio-video cross-attention
        
        Args:
            x: Video features [B, N_t*S, C]  
            encoder_hidden_states: Audio features [1, N_t, context_tokens, audio_dim]
            shape: (N_t, N_h, N_w) video spatial-temporal shape
        """
        encoder_hidden_states = encoder_hidden_states.squeeze(0)
        N_t, N_h, N_w = shape
        
        # Reshape video features: [B, N_t*S, C] -> [B*N_t, S, C]
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)
        B, N, C = x.shape
        
        # # Compute query for video features
        q = self.q_linear(x)
        q_shape = (B, N, self.num_heads, self.head_dim)
        q = q.view(q_shape)

        if self.qk_norm:
            q = self.q_norm(q)
        
        # Compute key-value for audio features  
        _, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 1, 3, 4)) 
        encoder_k, encoder_v = encoder_kv.unbind(0)

        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)

        # Frame-aligned cross-attention
        x = flash_attention(q, encoder_k, encoder_v, k_lens=None)
        
        # Linear transform
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape) 
        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)

        return x



# WAN_CROSSATTENTION_CLASSES = {
#     't2v_cross_attn': WanT2VCrossAttention,
#     'i2v_cross_attn': WanI2VCrossAttention,
# }


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 encoder_hidden_states_dim=768,
                 # audio_block=False
                 ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        # self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
        #                                                               num_heads,
        #                                                               (-1, -1),
        #                                                               qk_norm,
        #                                                               eps,
        #                                                               audio_block)
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)


    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        grid_sizes_cond,
        freqs,
        context,
        context_lens,
        ref_target_masks=None,
        audio_branch=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32
        # self-attention
        e_2 = [e[i].squeeze(2).to(dtype=x.dtype, device=x.device) for i in range(6)]
        xx = self.norm1(x)
        xx = xx * (1 + e_2[1]) + e_2[0]
        y = self.self_attn(
            xx, seq_lens, grid_sizes,
            freqs)
        x = x + y * e_2[2]
        x_norm = self.norm3(x)

        x = x + self.cross_attn(x_norm, context, context_lens)


        y = self.ffn(self.norm2(x) * (1 + e_2[4]) + e_2[3])
        with torch.amp.autocast('cuda',dtype=e_2[0].dtype):
            x = x + y * e_2[5]
        return x

class WanAttentionBlockHybrid(nn.Module):
    """
    Hybrid attention block combining Wan2.2 architecture with audio conditioning
    Based on Wan2.2's WanAttentionBlock with SingleStreamAttention integration
    """

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers (from Wan2.2)
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation (from Wan2.2)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # Audio cross-attention (from model_audio.py, adapted)
        self.audio_cross_attn = SingleStreamAttention(
            dim=dim,
            encoder_hidden_states_dim=768,  # AudioProjModel output: 32 tokens, each 768 dims
            num_heads=num_heads,
            qk_norm=False,
            qkv_bias=True,
            eps=eps,
            norm_layer=WanRMSNorm
        )
        self.norm_audio = WanLayerNorm(dim, eps, elementwise_affine=True)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        audio_embedding=None,
        ref_target_masks=None,
    ):
        r"""
        Forward pass with optional audio conditioning
        
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            context(Tensor): Text context
            context_lens(Tensor): Text context lengths
            audio_embedding(Tensor, optional): Audio embeddings for talking avatar
            ref_target_masks(Tensor, optional): Face masks for audio conditioning
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention (from Wan2.2)
        norm_x = self.norm1(x)
        if x.dtype == torch.bfloat16:
            # For mixed precision training, keep BFloat16
            # Convert modulation tensors to BFloat16 to match input dtype
            attn_input = norm_x * (1 + e[1].squeeze(2).to(x.dtype)) + e[0].squeeze(2).to(x.dtype)
        else:
            # For regular training, use float
            attn_input = norm_x.float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
        
        y = self.self_attn(attn_input, seq_lens, grid_sizes, freqs)
        if x.dtype == torch.bfloat16:
            # For mixed precision, ensure consistent dtype for residual connection
            x = x + y * e[2].squeeze(2).to(x.dtype)
        else:
            # For regular training, use float with autocast
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[2].squeeze(2)

        # cross-attention (from Wan2.2)
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        # cross attn of audio
        # x_a = self.audio_cross_attn(self.norm_audio(x), encoder_hidden_states=audio_embedding, shape=grid_sizes[0])
        # x = x + x_a * ref_target_masks  # 需适配mask

        # Audio cross-attention (from model_audio.py, adapted for Wan2.2)
        if audio_embedding is not None:
            x_audio_norm = self.norm_audio(x)
            
            # AudioProjModel output: [1, 21, 32, 768] - frame-by-frame audio tokens
            # Pass directly to SingleStreamAttention for proper frame alignment
            shape_tuple = (grid_sizes[0][0].item(), grid_sizes[0][1].item(), grid_sizes[0][2].item())
            audio_out = self.audio_cross_attn(
                x_audio_norm, 
                encoder_hidden_states=audio_embedding,  # [1, N_t, 32, 768]
                shape=shape_tuple  # (N_t, N_h, N_w) tuple
            )
            
            # Apply face mask if provided
            if ref_target_masks is not None:
                # Ensure face mask has correct dimensions
                if ref_target_masks.dim() == 3 and audio_out.dim() == 3:
                    # Both should be [B, seq_len, dim]
                    if ref_target_masks.size(1) != audio_out.size(1):
                        # Resize face mask to match audio output sequence length
                        ref_target_masks = F.interpolate(
                            ref_target_masks.transpose(1, 2), 
                            size=audio_out.size(1), 
                            mode='nearest'
                        ).transpose(1, 2)
                    audio_out = audio_out * ref_target_masks
            
            x = x + audio_out

        # FFN (from Wan2.2)
        # Keep consistent dtype for DeepSpeed mixed precision
        norm_x2 = self.norm2(x)
        if x.dtype == torch.bfloat16:
            # For mixed precision training, keep BFloat16
            # Convert modulation tensors to BFloat16 to match input dtype
            ffn_input = norm_x2 * (1 + e[4].squeeze(2).to(x.dtype)) + e[3].squeeze(2).to(x.dtype)
        else:
            # For regular training, use float
            ffn_input = norm_x2.float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
        
        y = self.ffn(ffn_input)
        if x.dtype == torch.bfloat16:
            # For mixed precision, ensure consistent dtype for residual connection
            x = x + y * e[5].squeeze(2).to(x.dtype)
        else:
            # For regular training, use float with autocast
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)

        return x

class AudioProjModel(nn.Module):
    """
    Audio projection model from model_audio.py
    Projects audio embeddings to format suitable for video-audio attention
    """
    
    def __init__(
        self,
        seq_len=5,
        seq_len_vf=12,
        blocks=12,  
        channels=768, 
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        norm_output_audio=False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels  
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds)) 
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf)) 
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1) 
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c*N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c*N_t, self.context_tokens, self.output_dim)

        # normalization and reshape
        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens

class Head_audio(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x

class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos
def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x

class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, conv_layers=4, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text, seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = torch.nn.functional.pad(text, (0, seq_len - text_len), value=0)
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        return text
import torch.nn.functional as F
class FeatureMapUpsampler(nn.Module):
    def __init__(self, in_channels):
        """
        初始化特征图上采样网络。

        Args:
            in_channels (int): 输入特征图的通道数 (即形状中的 't' 维度)。
        """
        super(FeatureMapUpsampler, self).__init__()

        # 1. 卷积精炼层
        # 在插值之后，使用一个标准卷积层来学习如何生成更高分辨率的特征，
        # 并且可以平滑插值可能引入的伪影。
        # 这里使用 kernel_size=3, padding=1 确保空间维度在卷积后保持不变。
        # 输入和输出通道数都设为 in_channels (即 't')，以保持通道数不变。
        self.conv_refine = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU() # 可以选择其他激活函数，如 LeakyReLU 或 SiLU

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入特征图，形状为 (d, t, w, h)。
                              通常情况下，d 是批量大小 (Batch Size)，t 是通道数 (Channels)，
                              w 是高度 (Height)，h 是宽度 (Width)。
        Returns:
            torch.Tensor: 输出特征图，形状为 (d, t, 1.375w, 1.375h)。
        """
        # Step 1: 空间上采样
        # 使用双线性插值 (bilinear) 来进行上采样，它通常适用于图像数据。
        # scale_factor=(1.375, 1.375) 表示对最后两个维度 (w 和 h) 进行 1.375 倍的缩放。
        # align_corners=False 对于下采样然后上采样的场景通常是推荐的，以避免不对称。
        # 因为您保证了 1.375w 和 1.375h 是整数，所以输出尺寸会是精确的整数。
        upsampled_x = F.interpolate(x, scale_factor=(1.375, 1.375), mode='bilinear', align_corners=False)

        # Step 2: 卷积精炼
        # 对上采样后的特征图应用卷积层和激活函数进行精炼。
        out = self.conv_refine(upsampled_x)
        out = self.relu(out)

        return out
class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='ti2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=48,
                 dim=3072,
                 audio_dim=1536,
                 ffn_dim=14336,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=48,
                 num_heads=24,
                 num_layers=30,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 back_append_frame=1,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.audio_dim=audio_dim

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlockHybrid(dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params_cond(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],dim=1)

        self.motion_patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.motion_text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        self.motion_time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.motion_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.motion_head = Head(dim, out_dim, patch_size, eps)
        self.motion_blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps,audio_dim)
            for _ in range(num_layers)
        ])
        self.zero_motion_proj_blocks = nn.ModuleList([
            nn.Linear(self.dim, self.dim, bias=False)
            for _ in range(num_layers)
        ])
        # self.motion_conv = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
        # self.motion_proj = nn.Linear(1536, self.dim, bias=False)
        # self.zero_motion_proj =  nn.Linear(self.dim, self.dim, bias=False)

        # init audio adapter
        audio_window=5
        intermediate_dim=512
        output_dim=768
        context_tokens=32
        vae_scale=4
        norm_output_audio=True
        self.audio_proj = AudioProjModel(
                    seq_len=audio_window,
                    seq_len_vf=audio_window+vae_scale-1,
                    intermediate_dim=intermediate_dim,
                    output_dim=output_dim,
                    context_tokens=context_tokens,
                    norm_output_audio=norm_output_audio,
                )
        # initialize weights
        self.enable_teacache = False
        self.init_weights()
        self.back_append_frame = back_append_frame

    def forward(
        self,
        x=None,
        motion=None,
        t=None,
        motion_t=None,
        prefix_len=1,
        mode=None,
        context_list=None,
        seq_len=None,
        seq_len_motion=None,
        audio_embedding=None,
        face_mask=None,
        skip_block=False,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        cond_flag=False,
        gpu_manager=None,
        up_scale=2.75,
        **kwargs
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # if self.model_type == 'i2v' or self.model_type == 'flf2v':
        #     assert clip_fea is not None and y is not None
        # params
        
        if motion_t is None:
            motion_t=t
        back_append_frame = self.back_append_frame
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if x is not None:
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]  # [b, 1, dim, t, h/2, w/2] -> [b,seq_len,dim 1536]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])  # [1, 3], thw
            frame_l = x[0].shape[-1] * x[0].shape[-2]
            x = [u.flatten(2).transpose(1, 2) for u in x]  # [b,  dim, thw/4] => [b, thw/4, dim]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long) 
            #print(seq_lens,"seq lens")  #tensor([10368]) seq lens
            if seq_len!=0:
                assert seq_lens.max() <= seq_len
                x = torch.cat([
                    torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                              dim=1) for u in x
                ])
        
        if motion is not None:
            motion = [self.motion_patch_embedding(u.unsqueeze(0)) for u in motion]  # [b, 1, dim, t, h/2, w/2]
            motion_grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in motion])  # [1, 3], thw
            frame_m = motion[0].shape[-1] * motion[0].shape[-2]
            _, dimm, ff, hh, ww = motion[0].shape
            bb = len(motion)
            motion = [u.flatten(2).transpose(1, 2) for u in motion]  # [b,  dim, thw/4] => [b, thw/4, dim]
            motion_seq_lens = torch.tensor([u.size(1) for u in motion], dtype=torch.long)
            if seq_len_motion!=0:
                assert motion_seq_lens.max() <= seq_len_motion
                motion = torch.cat([
                    torch.cat([u, u.new_zeros(1, seq_len_motion - u.size(1), u.size(2))],
                              dim=1) for u in motion
                ])

        seq_lens_video = torch.tensor([u.size(0) for u in x], dtype=torch.long)
        seq_lens_motion = torch.tensor([u.size(0) for u in motion], dtype=torch.long)
        seq_len2 = seq_lens_video.item() # TODO: 默认batch_size = 1，所以这里seq_len = seq_lens_video
        seq_len2_motion = seq_lens_motion.item() 
        la = 1
        # time embeddings (from Wan2.2)
        if t.dim() == 1:
            tv = t.view(-1,1).repeat(t.size(0), seq_len2)   # (b f)
            tm = motion_t.view(-1,1).repeat(motion_t.size(0), seq_len2_motion)   # (b f)
            tv[:, :prefix_len*frame_l] = 0.
            tm[:, :prefix_len*frame_m] = 0.
            tv[:, -back_append_frame*frame_l:] = 0.
            tm[:, -back_append_frame*frame_m:] = 0.
        with torch.amp.autocast('cuda',dtype=torch.float32):
            bt = t.size(0)
            tv = tv.flatten()
            tm = tm.flatten()

            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, tv).unflatten(0, (bt, seq_len2)).float())  # (b f c)
            e0 = self.time_projection(e).unflatten(2, (6, self.dim)) # (b f 6 c)

            e_m = self.motion_time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, tm).unflatten(0, (bt, seq_len2_motion)).float())  # (b f c)
            e0_m = self.motion_time_projection(e_m).unflatten(2, (6, self.dim)) # (b f 6 c)

            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = [context_list[0]] #,context_list[1]
        motion_context = [context_list[0]]


        if context is not None and x is not None:
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))
        if motion_context is not None and motion is not None:
            motion_context = self.motion_text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in motion_context
                ]))
            
        # Process audio embedding if provided (from model_audio.py, adapted)
        processed_audio = None
        if audio_embedding is not None:
            # Audio preprocess (based on model_audio.py lines 762-775)
            audio_cond = audio_embedding.to(device=x.device, dtype=x.dtype)
            first_frame_audio_emb_s = audio_cond[:, :1, ...] 
            latter_frame_audio_emb = audio_cond[:, 1:, ...] 
            latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=4)
            middle_index = 5 // 2
            latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...]
            latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
            latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
            latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
            latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...]
            latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
            latter_frame_audio_emb_s = torch.concat(
                [latter_first_frame_audio_emb,latter_middle_frame_audio_emb,latter_last_frame_audio_emb], 
                dim=2
            ) 
            # Use AudioProjModel for processing
            processed_audio = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
            zero_audio = torch.zeros_like(processed_audio[:,:1]).to(device=x.device, dtype=x.dtype)
            if back_append_frame == 1:
                processed_audio = torch.concat([processed_audio, zero_audio], dim=1)
            elif back_append_frame == 2:
                processed_audio = torch.concat([processed_audio, zero_audio, zero_audio], dim=1)


        # Face mask processing (based on model_audio.py implementation)
        processed_face_mask = None
        if face_mask is not None:
            if face_mask.dim() == 4:  # [1, 1, 512, 512]
                processed_face_mask = face_mask.squeeze(1).to(torch.float32)  # [1, 512, 512]
            elif face_mask.dim() == 3:  # [1, 512, 512]
                processed_face_mask = face_mask.to(torch.float32)
            else:
                processed_face_mask = face_mask.squeeze(0).to(torch.float32)
            
            processed_face_mask = F.interpolate(
                processed_face_mask.unsqueeze(1) if processed_face_mask.dim() == 3 else processed_face_mask, 
                size=(grid_sizes[0][-2].item(), grid_sizes[0][-1].item()), 
                mode='nearest'
            )
            processed_face_mask = processed_face_mask.repeat(
                1, grid_sizes[0][-3].item(), 1, 1
            ).view(x.shape[0], -1, 1).repeat(1, 1, x.shape[-1])
            processed_face_mask = processed_face_mask.to(device=x.device, dtype=x.dtype)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        if x is not None:
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                audio_embedding=processed_audio,
                ref_target_masks=processed_face_mask
            )
        if motion is not None:
            motion_kwargs = dict(
                e=e0_m,
                seq_lens=motion_seq_lens,
                grid_sizes=motion_grid_sizes,
                grid_sizes_cond=None,
                freqs=self.freqs,
                context=motion_context,
                context_lens=context_lens,
                ref_target_masks=processed_face_mask,
            )
        iii = 0
        pre_motion = torch.zeros_like(motion, requires_grad=True)
        for idx,block in enumerate(self.blocks):
            if gpu_manager is not None:
                if idx < len(self.blocks):
                    module = gpu_manager.managed_modules[idx]
                    if hasattr(module, 'to'):
                        module.to(gpu_manager.device)
                if idx > 0 and (idx - 1) < len(self.blocks):
                    prev_module = gpu_manager.managed_modules[idx - 1]
                    if hasattr(prev_module, 'to'):
                        prev_module.to('cpu')
            if skip_block and idx==11:
                continue
            residual_motion = motion.detach() - pre_motion.detach()
            pre_motion = motion.detach()
            residual_motion = residual_motion.transpose(1, 2).reshape(bb, -1, ff, hh, ww)
            residual_motion = residual_motion.permute(0,2,1,3,4).reshape(bb*ff, -1, hh, ww)
            residual_motion_up = F.interpolate(
                residual_motion, scale_factor=(up_scale, up_scale),  #2.75 = 704/256,2.0=512/256
                mode='bilinear', align_corners=False
                ).to(motion.dtype)
            H_out, W_out = residual_motion_up.shape[-2], residual_motion_up.shape[-1]
            last_part_len = H_out * W_out
   
            residual_motion_up = residual_motion_up.reshape(bb, ff, -1, H_out, W_out).permute(0,2,1,3,4)

            if gpu_manager is not None:
                if idx < len(self.zero_motion_proj_blocks):
                    module_ = gpu_manager.managed_motion_proj_modules[idx]
                    if hasattr(module_, 'to'):
                        module_.to(gpu_manager.device)
                if idx > 0 and (idx - 1) < len(self.zero_motion_proj_blocks):
                    prev_module_ = gpu_manager.managed_motion_proj_modules[idx - 1]
                    if hasattr(prev_module_, 'to'):
                        prev_module_.to('cpu')
   
            value_to_add = self.zero_motion_proj_blocks[idx](residual_motion_up.flatten(2).transpose(1, 2))
            #print(f"value_to_add shape: {value_to_add.shape}",f"x shape: {x.shape}",f"last_part_len: {last_part_len}") 
            #value_to_add shape: torch.Size([1, 19602, 3072]) x shape: torch.Size([1, 10368, 3072]) last_part_len: 726 # if use short_side=512 got error when upscale=2.75

            x[:, :-last_part_len, :] = x[:, :-last_part_len, :] + value_to_add[:, :-last_part_len, :]


            if gpu_manager is not None:
                if idx < len(self.motion_blocks):
                    module_1 = gpu_manager.managed_motion_modules[idx]
                    if hasattr(module_1, 'to'):
                        module_1.to(gpu_manager.device)
                if idx > 0 and (idx - 1) < len(self.motion_blocks):
                    prev_module_1 = gpu_manager.managed_motion_modules[idx - 1]
                    if hasattr(prev_module_1, 'to'):
                        prev_module_1.to('cpu')

            motion_block=self.motion_blocks[idx]
           

            iii += 1
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    if x is not None:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, e0, seq_lens, grid_sizes,
                            self.freqs,context, context_lens,
                            processed_audio, processed_face_mask,
                            use_reentrant=False,
                        )
                    if motion is not None:
                        motion = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(motion_block),
                            motion, e0_m, motion_seq_lens, motion_grid_sizes, None, self.freqs,
                            motion_context, context_lens,
                            face_mask, False,
                            use_reentrant=False,
                        )
            elif use_gradient_checkpointing:
                if x is not None:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, e0, seq_lens, grid_sizes,
                        self.freqs,context, 
                        context_lens,processed_audio, 
                        processed_face_mask,
                        use_reentrant=False,
                    )
                if motion is not None:
                    motion = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(motion_block),
                            motion, e0_m, motion_seq_lens, motion_grid_sizes, None, self.freqs,
                            motion_context, context_lens,
                            face_mask, False,
                            use_reentrant=False,
                        )
            else:
                if x is not None:
                    x = block(x, **kwargs)
                if motion is not None:
                    motion = motion_block(motion, **motion_kwargs)

        # head
        if x is not None:
            x = self.head(x, e)
            x = self.unpatchify(x, grid_sizes)

        if motion is not None:
            motion = self.motion_head(motion, e_m)
            motion = self.unpatchify(motion, motion_grid_sizes)
        if x is not None:
            if motion is not None:
                return x[0],motion[0]
            else:
                return x[0],None
        if motion is not None:
            return None,motion[0]
        return None,None

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

