import os
from einops import rearrange

import torch
import torch.nn as nn

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from einops import rearrange, repeat
from functools import lru_cache
import imageio
import uuid
from tqdm import tqdm
import numpy as np
import subprocess
import soundfile as sf


VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
ASPECT_RATIO_627 = {
     '0.26': ([320, 1216], 1), '0.38': ([384, 1024], 1), '0.50': ([448, 896], 1), '0.67': ([512, 768], 1), 
     '0.82': ([576, 704], 1),  '1.00': ([640, 640], 1),  '1.22': ([704, 576], 1), '1.50': ([768, 512], 1), 
     '1.86': ([832, 448], 1),  '2.00': ([896, 448], 1),  '2.50': ([960, 384], 1), '2.83': ([1088, 384], 1), 
     '3.60': ([1152, 320], 1), '3.80': ([1216, 320], 1), '4.00': ([1280, 320], 1)}


ASPECT_RATIO_960 = {
     '0.22': ([448, 2048], 1), '0.29': ([512, 1792], 1), '0.36': ([576, 1600], 1), '0.45': ([640, 1408], 1), 
     '0.55': ([704, 1280], 1), '0.63': ([768, 1216], 1), '0.76': ([832, 1088], 1), '0.88': ([896, 1024], 1), 
     '1.00': ([960, 960], 1), '1.14': ([1024, 896], 1), '1.31': ([1088, 832], 1), '1.50': ([1152, 768], 1), 
     '1.58': ([1216, 768], 1), '1.82': ([1280, 704], 1), '1.91': ([1344, 704], 1), '2.20': ([1408, 640], 1), 
     '2.30': ([1472, 640], 1), '2.67': ([1536, 576], 1), '2.89': ([1664, 576], 1), '3.62': ([1856, 512], 1), 
     '3.75': ([1920, 512], 1)}



def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# 确保所有必要的库都已导入
import torch
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
import textwrap
from tqdm import tqdm
import subprocess
import os

# 1. 确保所有必要的库都已导入
import torch
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
import textwrap
from tqdm import tqdm
import subprocess
import os

def merge_audio_and_video(video_path, audio_path, output_path):
    """
    使用 FFmpeg 将指定的音频和视频合并。
    
    参数:
    video_path (str): 原始视频文件路径。
    audio_path (str): 要合并的音频文件路径。
    output_path (str): 合成后视频的保存路径。
    """
    # 构建 FFmpeg 命令
    # -i video.mp4: 输入视频文件
    # -i audio.wav: 输入音频文件
    # -c:v copy: 直接复制视频流，不重新编码，速度快且无损画质
    # -c:a aac: 将音频编码为 AAC 格式，这是 mp4 容器的常用格式
    # -map 0:v:0: 映射第一个输入文件（视频）的视频流
    # -map 1:a:0: 映射第二个输入文件（音频）的音频流
    # -shortest: 当最短的输入流结束时，完成编码。这确保了如果音频比视频长，音频会被截断以匹配视频长度。
    # -y: 如果输出文件已存在，则自动覆盖
    new_path = '/apdcephfs_cq10/share_1367250/raylanzhang/ffmpeg2/ffmpeg/ffmpeg-7.0.2-i686-static'

    # 获取当前的 PATH 环境变量
    # 使用 os.environ.get('PATH', '') 来避免在 PATH 不存在时出错
    current_path = os.environ.get('PATH', '')

    # 构建新的 PATH，将新路径添加到最前面
    # os.pathsep 是系统特定的路径分隔符 (在 Linux 和 macOS 上是 ':'，在 Windows 上是 ';')
    new_path_value = new_path + os.pathsep + current_path

    # 设置新的 PATH 环境变量
    os.environ['PATH'] = new_path_value

    command = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',  # <--- 关键参数
        '-y',
        output_path
    ]
    
    try:
        print(f"正在处理: {os.path.basename(video_path)} 和 {audio_path}")
        # 执行命令，并隐藏 FFmpeg 的输出信息
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"成功 -> {os.path.basename(output_path)}")
    except subprocess.CalledProcessError as e:
        print(f"处理失败: {os.path.basename(video_path)}")
        # 打印错误信息以便调试
        print("FFmpeg 错误信息:", e.stderr.decode())
    except FileNotFoundError:
        print("错误：找不到 'ffmpeg' 命令。请确保 FFmpeg 已正确安装并已添加到系统 PATH 环境变量中。")
        return

def save_composite_video_with_audio(
    main_video_tensor,
    save_path,
    motion_video_tensor=None,
    vocal_audio_path=None,
    text=None,
    fps=25,
    quality=7, # 默认质量稍作提高
    font_path=None,
    bg_color='black',
    text_color='white'
):
    """
    将一或两个视频与可选的文本和音频合并，并保存为一个最终的 MP4 文件。

    该函数整合了视觉合成和音频添加的功能，并开启了调试模式：
    1.  它首先将主视频、可选的第二个视频和可选的文本面板横向拼接，
        生成一个无声的视觉合成视频。
    2.  如果提供了音频文件，它会使用 FFmpeg 将音频剪辑到与视频等长，
        然后将其混入视频中。FFmpeg 的所有输出都会被打印，以便调试。
    3.  最后清理所有临时文件，只留下最终的带音频的视频。

    参数:
        main_video_tensor (torch.Tensor): 形状为 (C, T, H, W) 的主视频张量，像素值范围 [-1, 1]。
        save_path (str): 最终视频的保存路径（.mp4 后缀会自动处理）。
        motion_video_tensor (torch.Tensor, optional): 形状与主视频相同的第二个视频张量。
        vocal_audio_path (str or list, optional): 输入的音频文件路径。如果为 None，则只输出无声视频。
        text (str, optional): 要显示在视频右侧的文字。
        fps (int): 视频的帧率。
        quality (int): 视频质量，数值越高质量越好。
        font_path (str, optional): .ttf 或 .otf 字体文件的路径。
        bg_color (str): 文字区域的背景颜色。
        text_color (str): 文字的颜色。
    """
    # ------------------ 设置文件路径 ------------------
    base_save_path = os.path.splitext(save_path)[0]
    final_video_path = base_save_path + ".mp4"
    temp_video_path = base_save_path + "_temp_video.mp4"
    temp_audio_path = base_save_path + "_temp_audio.wav"

    # =========================================================================
    # PART 1: 视觉合成 (生成无声视频)
    # =========================================================================
    print("步骤 1/3: 正在生成视觉合成视频...")

    # 内部辅助函数
    def _save_video_from_frames(frames, path, fps, quality):
        # 使用 imageio V3 API
        writer = imageio.get_writer(path, fps=fps, quality=quality, macro_block_size=1)
        for frame in tqdm(frames, desc=f"正在保存临时视频到 {path}"):
            writer.append_data(np.array(frame))
        writer.close()

    def _process_video_tensor(video_tensor):
        video_frames = (video_tensor + 1) / 2
        video_frames = video_frames.permute(1, 2, 3, 0).cpu().numpy()
        return np.clip(video_frames * 255, 0, 255).astype(np.uint8)

    # 1.1 转换视频张量
    video_frames_uint8 = _process_video_tensor(main_video_tensor)
    _, T, H, W = main_video_tensor.shape

    motion_frames_uint8 = None
    if motion_video_tensor is not None:
        motion_frames_uint8 = _process_video_tensor(motion_video_tensor)

    # 1.2 创建文本面板
    text_panel = None
    if text:
        margin = int(W * 0.1)
        max_text_width, max_text_height = W - margin, H - margin
        font_size = max(15, int(H / 15))
        font = None
        wrapped_text = text
        while font_size > 8:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except (IOError, TypeError):
                if font is None: font = ImageFont.load_default()
            avg_char_width = font_size * 0.6
            wrap_width = max(10, int(W / avg_char_width))
            wrapped_text = textwrap.fill(text, width=wrap_width)
            temp_draw = ImageDraw.Draw(Image.new('RGB', (W, H)))
            bbox = temp_draw.multiline_textbbox((0, 0), wrapped_text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if text_width < max_text_width and text_height < max_text_height: break
            font_size -= 1
        text_panel = Image.new('RGB', (W, H), color=bg_color)
        draw = ImageDraw.Draw(text_panel)
        text_x, text_y = (W - text_width) / 2, (H - text_height) / 2
        draw.multiline_text((text_x, text_y), wrapped_text, font=font, fill=text_color, align='center')

    # 1.3 合成每一帧
    num_columns = 1 + (motion_frames_uint8 is not None) + (text_panel is not None)
    final_width = W * num_columns
    composite_frames = []
    for i in range(len(video_frames_uint8)):
        composite_frame = Image.new('RGB', (final_width, H))
        current_x = 0
        composite_frame.paste(Image.fromarray(video_frames_uint8[i]), (current_x, 0))
        current_x += W
        if motion_frames_uint8 is not None:
            composite_frame.paste(Image.fromarray(motion_frames_uint8[i]), (current_x, 0))
            current_x += W
        if text_panel is not None:
            composite_frame.paste(text_panel, (current_x, 0))
        composite_frames.append(composite_frame)

    # 1.4 保存无声视频
    _save_video_from_frames(composite_frames, temp_video_path, fps, quality)
    print("无声视频已成功生成。")

    # =========================================================================
    # PART 2: 添加音频 (使用 FFmpeg)
    # =========================================================================
    if vocal_audio_path is None:
        print("未提供音频文件，将无声视频重命名为最终文件。")
        os.rename(temp_video_path, final_video_path)
        print(f"最终视频已保存到: {final_video_path}")
        return

    print("步骤 2/3: 正在处理并添加音频...")

    # 2.1 智能处理音频路径输入 (str 或 list)
    actual_audio_file = None
    if isinstance(vocal_audio_path, list):
        if len(vocal_audio_path) > 0:
            actual_audio_file = vocal_audio_path[0]
            if len(vocal_audio_path) > 1:
                print(f"警告: 提供了 {len(vocal_audio_path)} 个音频文件，只会使用第一个: {actual_audio_file}")
    elif isinstance(vocal_audio_path, str):
        actual_audio_file = vocal_audio_path

    if not actual_audio_file:
        print("警告: 提供的音频路径为空或格式不正确，将生成无声视频。")
        os.rename(temp_video_path, final_video_path)
        return

    # 2.2 剪辑音频
    duration = T / fps
    try:
        crop_command = [
            "ffmpeg", "-y",
            "-i", actual_audio_file,
            "-t", f'{duration}',
            "-acodec", "pcm_s16le", # 使用标准的WAV编码，兼容性好
            temp_audio_path,
        ]
        print("\n[调试信息] 将要执行音频剪辑命令:")
        print(" ".join(crop_command))
        
        # 执行命令并显示所有输出
        subprocess.run(crop_command, check=True)
        
        print(f"音频已成功剪辑到 {duration:.2f} 秒。")

    except FileNotFoundError:
        print("\n错误: 'ffmpeg' 命令未找到。请确保 FFmpeg 已正确安装并已添加到系统的 PATH 环境变量中。")
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        return
    except subprocess.CalledProcessError as e:
        print(f"\n错误：剪辑音频失败。FFmpeg 返回了非零退出码。请检查上面的 FFmpeg 输出信息以了解详情。")
        print(f"Python 错误详情: {e}")
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        return

    # 2.3 合并视频和音频
    try:
        merge_command = [
            "ffmpeg", "-y",
            "-i", temp_video_path,
            "-i", temp_audio_path,
            "-c:v", "copy",       # 直接复制视频流，速度快
            "-c:a", "aac",        # 编码音频流
            "-shortest",          # 以最短的流为准结束
            final_video_path,
        ]
        print("\n[调试信息] 将要执行音视频合并命令:")
        print(" ".join(merge_command))

        # 执行命令并显示所有输出
        subprocess.run(merge_command, check=True)

        print("视频和音频合并成功。")

    except FileNotFoundError:
        print("\n错误: 'ffmpeg' 命令未找到。请确保 FFmpeg 已正确安装并已添加到系统的 PATH 环境变量中。")
    except subprocess.CalledProcessError as e:
        print(f"\n错误：合并视频和音频失败。FFmpeg 返回了非零退出码。请检查上面的 FFmpeg 输出信息以了解详情。")
        print(f"Python 错误详情: {e}")
    finally:
        # =========================================================================
        # PART 3: 清理临时文件
        # =========================================================================
        print("步骤 3/3: 正在清理临时文件...")
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        if os.path.exists(final_video_path):
            print(f"任务完成！最终视频已成功保存到: {final_video_path}")
        else:
            print("任务失败，未生成最终文件。请检查上面的日志输出。")

def split_token_counts_and_frame_ids(T, token_frame, world_size, rank):

    S = T * token_frame
    split_sizes = [S // world_size + (1 if i < S % world_size else 0) for i in range(world_size)]
    start = sum(split_sizes[:rank])
    end = start + split_sizes[rank]
    counts = [0] * T
    for idx in range(start, end):
        t = idx // token_frame
        counts[t] += 1

    counts_filtered = []
    frame_ids = []
    for t, c in enumerate(counts):
        if c > 0:
            counts_filtered.append(c)
            frame_ids.append(t)
    return counts_filtered, frame_ids


def normalize_and_scale(column, source_range, target_range, epsilon=1e-8):

    source_min, source_max = source_range
    new_min, new_max = target_range
 
    normalized = (column - source_min) / (source_max - source_min + epsilon)
    scaled = normalized * (new_max - new_min) + new_min
    return scaled


@torch.compile
def calculate_x_ref_attn_map(visual_q, ref_k, ref_target_masks, mode='mean', attn_bias=None):
    
    ref_k = ref_k.to(visual_q.dtype).to(visual_q.device)
    scale = 1.0 / visual_q.shape[-1] ** 0.5
    visual_q = visual_q * scale
    visual_q = visual_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = visual_q @ ref_k.transpose(-2, -1)

    if attn_bias is not None:
        attn = attn + attn_bias

    x_ref_attn_map_source = attn.softmax(-1) # B, H, x_seqlens, ref_seqlens


    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.to(visual_q.dtype)
    x_ref_attn_map_source = x_ref_attn_map_source.to(visual_q.dtype)

    for class_idx, ref_target_mask in enumerate(ref_target_masks):
        torch_gc()
        ref_target_mask = ref_target_mask[None, None, None, ...]
        x_ref_attnmap = x_ref_attn_map_source * ref_target_mask
        x_ref_attnmap = x_ref_attnmap.sum(-1) / ref_target_mask.sum() # B, H, x_seqlens, ref_seqlens --> B, H, x_seqlens
        x_ref_attnmap = x_ref_attnmap.permute(0, 2, 1) # B, x_seqlens, H
       
        if mode == 'mean':
            x_ref_attnmap = x_ref_attnmap.mean(-1) # B, x_seqlens
        elif mode == 'max':
            x_ref_attnmap = x_ref_attnmap.max(-1) # B, x_seqlens
        
        x_ref_attn_maps.append(x_ref_attnmap)
    
    del attn
    del x_ref_attn_map_source
    torch_gc()

    return torch.concat(x_ref_attn_maps, dim=0)


def get_attn_map_with_target(visual_q, ref_k, shape, ref_target_masks=None, split_num=2, enable_sp=False):
    """Args:
        query (torch.tensor): B M H K
        key (torch.tensor): B M H K
        shape (tuple): (N_t, N_h, N_w)
        ref_target_masks: [B, N_h * N_w]
    """

    N_t, N_h, N_w = shape
    if enable_sp:
        ref_k = get_sp_group().all_gather(ref_k, dim=1)
    
    x_seqlens = N_h * N_w
    ref_k     = ref_k[:, :x_seqlens]
    _, seq_lens, heads, _ = visual_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = torch.zeros(class_num, seq_lens).to(visual_q.device).to(visual_q.dtype)

    split_chunk = heads // split_num
    
    for i in range(split_num):
        x_ref_attn_maps_perhead = calculate_x_ref_attn_map(
            visual_q[:, :, i*split_chunk:(i+1)*split_chunk, :], 
            ref_k[:, :, i*split_chunk:(i+1)*split_chunk, :], 
            ref_target_masks
        )
        x_ref_attn_maps += x_ref_attn_maps_perhead
    
    return x_ref_attn_maps / split_num


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RotaryPositionalEmbedding1D(nn.Module):

    def __init__(self,
                 head_dim,
                 ):
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000


    @lru_cache(maxsize=32)
    def precompute_freqs_cis_1d(self, pos_indices):

        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))
        freqs = freqs.to(pos_indices.device)
        freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def forward(self, x, pos_indices):
        """1D RoPE.

        Args:
            query (torch.tensor): [B, head, seq, head_dim]
            pos_indices (torch.tensor): [seq,]
        Returns:
            query with the same shape as input.
        """
        freqs_cis = self.precompute_freqs_cis_1d(pos_indices)

        x_ = x.float()

        freqs_cis = freqs_cis.float().to(x.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')
        x_ = (x_ * cos) + (rotate_half(x_) * sin)

        return x_.type_as(x)
    

def save_video_ffmpeg(gen_video_samples, save_path, vocal_audio_list, fps=25, quality=5):
    
    def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
        writer = imageio.get_writer(
            save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
        )
        for frame in tqdm(frames, desc="Saving video"):
            frame = np.array(frame)
            writer.append_data(frame)
        writer.close()
    save_path_tmp = save_path + "-temp.mp4"

    video_audio = (gen_video_samples+1)/2 # C T H W
    video_audio = video_audio.permute(1, 2, 3, 0).cpu().numpy()
    video_audio = np.clip(video_audio * 255, 0, 255).astype(np.uint8) 
    save_video(video_audio, save_path_tmp, fps=fps, quality=quality)


    # crop audio according to video length
    _, T, _, _ = gen_video_samples.shape
    duration = T / fps
    save_path_crop_audio = save_path + "-cropaudio.wav"
    final_command = [
        "ffmpeg",
        "-i",
        vocal_audio_list[0],
        "-t",
        f'{duration}',
        save_path_crop_audio,
    ]
    subprocess.run(final_command, check=True)


    # generate video with audio
    save_path = save_path + ".mp4"
    final_command = [
        "ffmpeg",
        "-y",
        "-i",
        save_path_tmp,
        "-i",
        save_path_crop_audio,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        save_path,
    ]
    subprocess.run(final_command, check=True)
    os.remove(save_path_tmp)
    os.remove(save_path_crop_audio)



class MomentumBuffer:
    def __init__(self, momentum: float): 
        self.momentum = momentum 
        self.running_average = 0 
    
    def update(self, update_value: torch.Tensor): 
        new_average = self.momentum * self.running_average 
        self.running_average = update_value + new_average
    


def project( 
        v0: torch.Tensor, # [B, C, T, H, W] 
        v1: torch.Tensor, # [B, C, T, H, W] 
        ): 
    dtype = v0.dtype 
    v0, v1 = v0.double(), v1.double() 
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3, -4]) 
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3, -4], keepdim=True) * v1 
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def adaptive_projected_guidance( 
          diff: torch.Tensor, # [B, C, T, H, W] 
          pred_cond: torch.Tensor, # [B, C, T, H, W] 
          momentum_buffer: MomentumBuffer = None, 
          eta: float = 0.0,
          norm_threshold: float = 55,
          ): 
    if momentum_buffer is not None: 
        momentum_buffer.update(diff) 
        diff = momentum_buffer.running_average
    if norm_threshold > 0: 
        ones = torch.ones_like(diff) 
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3, -4], keepdim=True) 
        print(f"diff_norm: {diff_norm}")
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm) 
        diff = diff * scale_factor 
    diff_parallel, diff_orthogonal = project(diff, pred_cond) 
    normalized_update = diff_orthogonal + eta * diff_parallel
    
    return normalized_update
