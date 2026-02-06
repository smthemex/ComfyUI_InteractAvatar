# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
from contextlib import contextmanager,nullcontext
from functools import partial
from safetensors.torch import safe_open
import torch
import torch.distributed as dist
from tqdm import tqdm
from .modules.model_tia2mv_rope_back import WanModel,BlockGPUManager
from .distributed.fsdp import shard_model
from accelerate import init_empty_weights
from diffusers.utils import is_accelerate_available
from .modules.t5 import T5EncoderModel
from .modules.vae2_2 import Wan2_2_VAE
import torchvision.transforms.functional as TF
def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t
import random

def distribute_prompts_with_gaps(prompt_dict, num_loops):
    """
    将 prompt_dict 中的字符串填充到长度为 num_loops 的列表中，
    保留元素原始顺序，但位置随机，且保证相邻非空元素间至少间隔一个空字符串。
    """
    k = len(prompt_dict)
    
    # 边界情况处理：如果是空列表，直接返回全空列表
    if k == 0:
        return [""] * num_loops
    
    # 1. 计算硬性限制需要的最小长度
    # K个元素本身占用 K 个位置
    # K个元素之间至少需要 K-1 个空格
    min_required_len = k + (k - 1)
    
    if num_loops < min_required_len:
        raise ValueError(
            f"num_loops ({num_loops}) 太小，无法容纳 {k} 个元素并保持间隔。"
            f"至少需要 {min_required_len} 的长度。"
        )
    
    # 2. 计算剩余可自由分配的空格数 (Slack)
    # 这里的 slack 不包含那些必须存在的间隔符
    slack = num_loops - min_required_len
    
    # 3. 将 slack 随机分配到 k+1 个区域中
    # 区域 0: 第1个元素之前
    # 区域 1: 第1个和第2个元素之间 (在强制间隔的基础上额外增加)
    # ...
    # 区域 k: 最后一个元素之后
    # 方法：在 0 到 slack 之间随机切 k 刀，形成 k+1 段
    cut_points = sorted([random.randint(0, slack) for _ in range(k)])
    
    # 计算每段分配到的空格数
    # 为了方便计算，首尾加上边界
    extended_cuts = [0] + cut_points + [slack]
    gaps_distribution = [extended_cuts[i+1] - extended_cuts[i] for i in range(k + 1)]
    
    # 4. 构建最终列表
    result = []
    for i in range(k):
        # A. 插入当前区域分配到的游离空格
        result.extend([""] * gaps_distribution[i])
        
        # B. 插入当前有意义的字符串
        result.append(prompt_dict[i])
        
        # C. 如果不是最后一个元素，强制插入一个必要的间隔空格
        if i < k - 1:
            result.append("")
            
    # D. 补齐最后区域的游离空格
    result.extend([""] * gaps_distribution[k])
    
    return result

import numpy as np
class WanTIA2MVRefBackIDPrefix:

    def __init__(
        self,
        config,
        checkpoint_dir,
        transformer_dir,
        lora_path=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        load_from_merged_model=None,
        back_append_frame=1,
        short_side=512,
        offload=False,
        origin_mode=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """

        self.device = torch.device(f"cuda:{device_id}") 
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.origin_mode = origin_mode
        self.offload = offload
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        if self.origin_mode:
            shard_fn = partial(shard_model, device_id=device_id)
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
                tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
                shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size

        if self.origin_mode:
            self.vae = Wan2_2_VAE(
                vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
                device=self.device
            )
        logging.info(f"Creating WanModel from {checkpoint_dir}")
        model_cfg = config
        model_type = 'ti2v' if model_cfg.__name__ == 'Config: Wan TI2V 5B' else 'i2v'
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx(): 
            unet = WanModel(
                model_type=model_type,
                patch_size=model_cfg.patch_size,
                text_len=model_cfg.text_len,
                in_dim=model_cfg.in_dim,  # Use config value: 48 dims for Wan2.2
                dim=model_cfg.dim,
                audio_dim=model_cfg.audio_dim,
                ffn_dim=model_cfg.ffn_dim,
                freq_dim=model_cfg.freq_dim,
                text_dim=model_cfg.text_dim,  # Use config value: 4096
                out_dim=model_cfg.out_dim,  # Use config value: 48 dims for Wan2.2
                num_heads=model_cfg.num_heads,
                num_layers=model_cfg.num_layers,
                back_append_frame=back_append_frame,
            )
        model_tensor={}
        checkpoint_file=transformer_dir
        #checkpoint_file = os.path.join(transformer_dir, 'diffusion_pytorch_model.safetensors')
        # if load_from_merged_model is not None:
        #     checkpoint_file = os.path.join(load_from_merged_model, 'diffusion_pytorch_model.safetensors')
        
        # Check if single file exists
        #if os.path.exists(checkpoint_file):
            
        with safe_open(checkpoint_file, framework="pt") as f:
            for k in f.keys():
                model_tensor[k] = f.get_tensor(k)
        # else:
        #     # Try to load from sharded files
        #     index_file = os.path.join(transformer_dir, 'diffusion_pytorch_model.safetensors.index.json')
        #     if os.path.exists(index_file):
        #         print(f"Loading sharded model from {index_file}")
        #         import json
        #         with open(index_file, 'r') as f:
        #             index_data = json.load(f)

        #         # Get list of shard files
        #         shard_files = set(index_data['weight_map'].values())
        #         print(f"Found {len(shard_files)} shard files: {shard_files}")

        #         # Load all shards
        #         for shard_file in shard_files:
        #             shard_path = os.path.join(transformer_dir, shard_file)
        #             print(f"Loading shard: {shard_path}")
        #             with safe_open(shard_path, framework="pt") as f:
        #                 for k in f.keys():
        #                     model_tensor[k] = f.get_tensor(k)
        #     else:
        #         raise FileNotFoundError(f"Neither {checkpoint_file} nor {index_file} found!")
        
        unet.load_state_dict(model_tensor, strict=False,assign=True)
        del model_tensor
        gc.collect()
        #if load_from_merged_model is not None:
        import torch.nn as nn
        with torch.no_grad():
            for name, param in unet.named_parameters():
                if 'zero_motion_proj' in name and 'weight' in name:
                    nn.init.zeros_(param)
                    #print('init zero motion proj:',name)
        #dtype = model_cfg.param_dtype
        # dtype=torch.bfloat16
        #device = torch.device(f"cuda:{device_id}")

        self.model = unet.eval().requires_grad_(False)

        #self.model.eval().requires_grad_(False)

        # if use_usp:
        #     from xfuser.core.distributed import \
        #         get_sequence_parallel_world_size

        #     from .distributed.xdit_context_parallel import (usp_attn_forward,
        #                                                     usp_dit_forward)
        #     for block in self.model.blocks:
        #         block.self_attn.forward = types.MethodType(
        #             usp_attn_forward, block.self_attn)
        #     self.model.forward = types.MethodType(usp_dit_forward, self.model)
        #     self.sp_size = get_sequence_parallel_world_size()
        # else:
        self.sp_size = 1

        # if dist.is_initialized():
        #     dist.barrier()
        # if dit_fsdp:
        #     self.model = shard_fn(self.model)
        # else:
        if offload:
            self.model.to(torch.device('cpu'))
        else:
            self.model.to(self.device)

        self.num_timesteps = 1000
        self.use_timestep_transform=True
        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 img,
                 pose_ref_img,
                 obj_img,
                 small_img,
                 pose_sequence=None,
                 audio_embedding=None,
                 id_img = None,
                 id_small_img = None,
                 zero_audio_embedding=None,
                 mode='a2v',
                 n_prompt="",
                 frame_num=81,
                 shift=5.0,
                 sampling_steps=40,
                 text_guide_scale=5.0,
                 audio_guide_scale=5.0,
                 seed=-1,
                 bad_cfg=False,
                 three_cfg=False,
                 bad_thres=0,
                 offload_model=True,
                 motion_frame=25,
                 max_frames_num=1000,
                 **kwargs

                 ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.
        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """

        if self.origin_mode:
            cond_image = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
            cond_image = cond_image[None, :, None, :, :]
            obj_image = TF.to_tensor(obj_img).sub_(0.5).div_(0.5).to(self.device)
            obj_image = obj_image[None, :, None, :, :]
            pose_ref_img = TF.to_tensor(pose_ref_img).sub_(0.5).div_(0.5).to(self.device)
            pose_ref_img = pose_ref_img[None, :, None, :, :] # Shape: B, C, 1, H, W
            small_img = TF.to_tensor(small_img).sub_(0.5).div_(0.5).to(self.device)
            small_img = small_img[None, :, None, :, :]

            # --- 2. Preprocess pose sequence ---
            # Ensure pose_sequence is a tensor and normalize it.
            if pose_sequence is not None:
                if isinstance(pose_sequence, list): # If input is a list of PIL Images
                    processed_poses = [TF.to_tensor(p).sub_(0.5).div_(0.5) for p in pose_sequence]
                    cond_pose_sequence = torch.stack(processed_poses).to(self.device)
                else: # If input is already a tensor
                    cond_pose_sequence = pose_sequence.to(self.device)
                dwpose_len = cond_pose_sequence.shape[0]
                dwpose_len = (dwpose_len - 1) // self.vae_stride[0] * self.vae_stride[0] + 1
                frame_num = min(frame_num, dwpose_len)
                cond_pose_sequence = cond_pose_sequence[:frame_num]
                # Add batch dimension and align dimensions: (N, C, H, W) -> (1, C, N, H, W)
                cond_pose_sequence = cond_pose_sequence.permute(1, 0, 2, 3).unsqueeze(0)
            else:
                cond_pose_sequence = torch.zeros_like(pose_ref_img).repeat(1, 1, frame_num, 1, 1)
            zero_cond_pose_sequence = torch.zeros_like(cond_pose_sequence)
        
        full_audio_embs = [audio_embedding]
        zero_full_audio_embs = [zero_audio_embedding]

        # preprocess
        if self.origin_mode:
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                context_null = self.text_encoder([n_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
        else:
            context=kwargs.get('context', None)
            context_null=kwargs.get('context_null', None)

        indices = (torch.arange(2 * 2 + 1) - 2) * 1 
        clip_length = frame_num
        is_first_clip = True
        arrive_last_frame = False
        cur_motion_frames_num = 1
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + clip_length
        gen_video_list = []
        # set random seed and init noise
        seed = seed if seed >= 0 else random.randint(0, 99999999)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # audio
        audio_embs = []
        # split audio with window size
        for human_idx in range(len(full_audio_embs)):   
            center_indices = torch.arange(
                audio_start_idx,
                audio_end_idx,
                1,
            ).unsqueeze(
                1
            ) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=full_audio_embs[human_idx].shape[0]-1)
            audio_emb = full_audio_embs[human_idx][center_indices][None,...].to(self.device)
            audio_embs.append(audio_emb)
        audio_embs = torch.concat(audio_embs, dim=0).to(self.param_dtype)


        zero_audio_embs = []
        # split audio with window size
        for human_idx in range(len(zero_full_audio_embs)):   
            center_indices = torch.arange(
                audio_start_idx,
                audio_end_idx,
                1,
            ).unsqueeze(
                1
            ) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=zero_full_audio_embs[human_idx].shape[0]-1)
            zero_audio_emb = zero_full_audio_embs[human_idx][center_indices][None,...].to(self.device)
            zero_audio_embs.append(zero_audio_emb)
        zero_audio_embs = torch.concat(zero_audio_embs, dim=0).to(self.param_dtype)

        if self.origin_mode:
            if mode in ['a2v','a2mv','mv','i2v']:
                cond_pose_sequence = torch.zeros_like(cond_pose_sequence)

        if mode in ['p2v','mv','i2v']:
            audio_embs = zero_audio_embs
        if self.origin_mode:
            h, w = cond_image.shape[-2], cond_image.shape[-1]
            lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
            max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
                self.patch_size[1] * self.patch_size[2])
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        else:
            lat_h=kwargs.get('lat_h', None)
            lat_w=kwargs.get('lat_w', None)
            max_seq_len=kwargs.get('max_seq_len', None)
        noise = torch.randn(
            1, 48, (frame_num - 1) // 4 + 1 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            device=self.device) 
        if self.origin_mode:
            cond_image = self.vae.encode([cond_image.squeeze(0).to(torch.float32)])[0].unsqueeze(0)
            obj_image = self.vae.encode([obj_image.repeat(1, 1, 4, 1, 1).squeeze(0).to(torch.float32)])[0].unsqueeze(0)
            small_img = self.vae.encode([small_img.repeat(1, 1, 4, 1, 1).squeeze(0).to(torch.float32)])[0].unsqueeze(0)
        else:
            cond_image = kwargs.get('cond_image', None)
            obj_image = kwargs.get('obj_image', None)
            small_img = kwargs.get('small_image', None)

        if self.origin_mode:
            cond_dw_img = pose_ref_img # Shape: B, C, 1, H, W. (1, C, N, H, W)
            motion_h, motion_w = cond_dw_img.shape[-2], cond_dw_img.shape[-1]
            motion_lat_h, motion_lat_w = motion_h // self.vae_stride[1], motion_w // self.vae_stride[2]
            motion_max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * motion_lat_h * motion_lat_w // (
                    self.patch_size[1] * self.patch_size[2])
            motion_max_seq_len = int(math.ceil(motion_max_seq_len / self.sp_size)) * self.sp_size
        else:
            motion_lat_h=kwargs.get('motion_lat_h', None)
            motion_lat_w=kwargs.get('motion_lat_w', None)
            motion_max_seq_len=kwargs.get('motion_max_seq_len', None)
            
        motion_noise = torch.randn(
            1, 48, (frame_num - 1) // 4 + 1 + 1,
            motion_lat_h,
            motion_lat_w,
            dtype=torch.float32,
            device=self.device) 
        if self.origin_mode:
            cond_dw_img = self.vae.encode([cond_dw_img.squeeze(0).to(torch.float32)])[0].unsqueeze(0)
            cond_pose_sequence = self.vae.encode([cond_pose_sequence.squeeze(0).to(torch.float32)])[0].unsqueeze(0)
        else:
            cond_dw_img = kwargs.get('cond_dw_img', None)
            cond_pose_sequence = kwargs.get('cond_pose_sequence', None)
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        self.input_dtype = noise.dtype
        # evaluation mode
        with torch.no_grad(), no_sync():
            
            # prepare timesteps
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]
            
            # sample videos
            latent = noise
            latent_motion = motion_noise

            cur_dtype=torch.bfloat16
            torch_gc()
            if not self.offload:
                if self.model.device != self.device:
                    self.model.to(self.device)
                gpu_manager=None
            else:
                gpu_manager = BlockGPUManager(device="cuda")
                gpu_manager.setup_for_inference(self.model)
            #print(latent.shape,cond_image.shape,obj_image.shape) #torch.Size([1, 48, 22, 48, 32]) torch.Size([1, 48, 1, 48, 32]) torch.Size([1, 48, 1, 48, 32])
            latent[:, :, 0:1] = cond_image 
            latent[:, :, -1:] = obj_image
            #print(latent_motion.shape,cond_dw_img.shape,small_img.shape) #torch.Size([1, 48, 22, 24, 16]) torch.Size([1, 48, 1, 24, 16]) torch.Size([1, 48, 1, 24, 16])
            latent_motion[:, :, 0:1] = cond_dw_img
            latent_motion[:, :, -1:] = small_img 
            #print(cond_pose_sequence.shape) #torch.Size([1, 48, 21, 24, 16])
            cond_pose_sequence = torch.concat([cond_pose_sequence, small_img], dim=2)
            cond_pose_sequence = cond_pose_sequence.to(latent_motion.dtype).to(self.device,cur_dtype)
            audio_embs = audio_embs.to(latent_motion.dtype).to(self.device,cur_dtype)
            # zero_audio_embs = zero_audio_embs.to(latent_motion.dtype).to(self.device)
            
            # prepare condition and uncondition configs
            arg_c = {
                'context_list': [context[0].to(cur_dtype)],
                'seq_len': max_seq_len + lat_h * lat_w //4,
                'seq_len_motion': motion_max_seq_len + motion_lat_h * motion_lat_w //4,
                'mode': mode,
                'skip_block': False,
                'audio_embedding': audio_embs,
            }
            arg_null_text = {
                'context_list': [context_null[0].to(cur_dtype)],
                'seq_len': max_seq_len + lat_h * lat_w //4,
                'seq_len_motion': motion_max_seq_len + motion_lat_h * motion_lat_w //4,
                'mode': mode,
                'skip_block': bad_cfg,
                'audio_embedding': audio_embs * 0,
            }
            arg_pure_audio = {
                'context_list': [context_null[0].to(cur_dtype)],
                'seq_len': max_seq_len + lat_h * lat_w //4,
                'seq_len_motion': motion_max_seq_len + motion_lat_h * motion_lat_w //4,
                'mode': mode,
                'skip_block': False,
                'audio_embedding': audio_embs,
            }

            progress_wrap = partial(tqdm, total=len(timesteps)-1)
            for i in progress_wrap(range(len(timesteps)-1)):
                timestep = timesteps[i]
                latent_model_input = latent.to(self.device,torch.bfloat16) # bf16
                latent_model_input_motion = latent_motion.to(self.device,torch.bfloat16)
                zero_timestep = torch.zeros_like(timestep).to(dtype=timestep.dtype, device=timestep.device)

                if timestep[0] > bad_thres:
                    arg_null_text['skip_block'] = bad_cfg
                    #print('bad_cfg', timestep, bad_cfg) #bad_cfg tensor([1000.], device='cuda:0') True
                else:
                    arg_null_text['skip_block'] = False
                    #print('no bad_cfg', timestep, bad_cfg)


                if mode not in ['mv','a2mv']:
                    noise_pred_cond,  noise_pred_cond_motion = self.model(
                        x=latent_model_input,motion=cond_pose_sequence,
                        t=timestep,motion_t=zero_timestep, **arg_c,gpu_manager=gpu_manager,
                        )
                    torch_gc()
                    noise_pred_drop_text, noise_pred_drop_text_motion = self.model(
                        x=latent_model_input,
                        motion=cond_pose_sequence, 
                        t=timestep,motion_t=zero_timestep, 
                        **arg_null_text,
                        gpu_manager=gpu_manager,
                    )
                    torch_gc()
                else:
                    # inference with CFG strategy
                    noise_pred_cond,  noise_pred_cond_motion = self.model(
                        x=latent_model_input,motion=latent_model_input_motion,
                          t=timestep,motion_t=timestep, **arg_c,gpu_manager=gpu_manager,
                          )
                    torch_gc()
                    noise_pred_drop_text, noise_pred_drop_text_motion = self.model(
                        x=latent_model_input,
                        motion=latent_model_input_motion, 
                        t=timestep,
                        motion_t=timestep, 
                        **arg_null_text,
                        gpu_manager=gpu_manager,

                    )
                    torch_gc()
                # vanilla CFG strategy
                noise_pred = noise_pred_drop_text + text_guide_scale * (
                    noise_pred_cond - noise_pred_drop_text)
                noise_pred = -noise_pred  

                noise_pred_motion = noise_pred_drop_text_motion + text_guide_scale * (
                    noise_pred_cond_motion - noise_pred_drop_text_motion)
                noise_pred_motion = -noise_pred_motion  

                # update latent
                dt = timesteps[i] - timesteps[i + 1]
                dt = dt / self.num_timesteps
                latent = latent + noise_pred * dt[:, None, None]
                latent_motion = latent_motion + noise_pred_motion * dt[:, None, None]

                latent[:, :, 0:1] = cond_image 
                latent[:, :, -1:] = obj_image
                latent_motion[:, :, 0:1] = cond_dw_img
                latent_motion[:, :, -1:] = small_img

                x0 = [latent[:, :, :-1].to(self.device)] 
                x0_motion = [latent_motion[:, :, :-1].to(self.device)] 
            if self.offload:
                gpu_manager.unload_all_blocks_to_cpu()
                torch_gc()
            else:
                if offload_model: 
                    self.model.cpu()
                torch_gc()
        if self.origin_mode:
            videos = self.vae.decode(x0[0])[0]
            if mode in ['mv','a2mv']:
                videos_motion = self.vae.decode(x0_motion[0])[0]
            else:
                videos_motion = self.vae.decode(cond_pose_sequence[:, :, :-1])[0]
        else:
            videos=x0[0] 
            if mode in ['mv','a2mv']:
                videos_motion=x0_motion[0]
            else:
                videos_motion=cond_pose_sequence[:, :, :-1]
        torch_gc()
        if offload_model:    
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        
        if dist.is_initialized():
            dist.barrier()

        del noise, latent
        torch_gc()
        #print(videos.shape, videos_motion.shape) #torch.Size([1, 48, 26, 48, 32]) torch.Size([1, 48, 26, 24, 16])
        return videos, videos_motion if self.rank == 0 else None

    def generate_long(self,
                 prompt_dict,
                 img,
                 pose_ref_img,
                 obj_img,
                 small_img,
                 pose_sequence = None,
                 audio_embedding = None,
                 id_img = None,
                 id_small_img = None,
                 zero_audio_embedding=None,
                 mode='a2v',
                 n_prompt="",
                 frame_num=101, 
                 shift=5.0,
                 sampling_steps=40,
                 text_guide_scale=5.0,
                 audio_guide_scale=5.0,
                 seed=-1,
                 bad_cfg=False,
                 three_cfg=False,
                 bad_thres=0,
                 offload_model=True,
                 motion_frame=25,
                 max_frames_num=1000,
                 **kwargs
                 ):
        r"""
        Generates video frames autoregressively with Latent Prefix Loop.
        Segment Length: 101 frames (26 temporal latents + 2 condition latents).
        """
        if self.origin_mode:
            # --- 1. 输入预处理 (Input Preprocessing) ---
            # 初始首帧 (Start Frame) - 仅用于第一段
            curr_cond_image = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(0)
            curr_pose_ref_img = TF.to_tensor(pose_ref_img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(0)
        
            # 固定的尾帧条件 (Fixed End Condition)
            obj_image = TF.to_tensor(obj_img).sub_(0.5).div_(0.5).to(self.device)
            obj_image = obj_image[None, :, None, :, :] 
            small_img_tensor = TF.to_tensor(small_img).sub_(0.5).div_(0.5).to(self.device)
            small_img_tensor = small_img_tensor[None, :, None, :, :]
            id_img = TF.to_tensor(id_img).sub_(0.5).div_(0.5).to(self.device)
            id_img = id_img[None, :, None, :, :]
            id_small_img = TF.to_tensor(id_small_img).sub_(0.5).div_(0.5).to(self.device)
            id_small_img = id_small_img[None, :, None, :, :]

            # Pose Sequence 处理
            if pose_sequence is not None:
                if isinstance(pose_sequence, list): 
                    processed_poses = [TF.to_tensor(p).sub_(0.5).div_(0.5) for p in pose_sequence]
                    cond_pose_sequence_full = torch.stack(processed_poses).to(self.device)
                else: 
                    cond_pose_sequence_full = pose_sequence.to(self.device)
                # 确保维度是 (1, C, N, H, W)
                if cond_pose_sequence_full.ndim == 4:
                    cond_pose_sequence_full = cond_pose_sequence_full.permute(1, 0, 2, 3).unsqueeze(0)
            else:
                cond_pose_sequence_full = torch.zeros(
                    1, 3, max_frames_num + 2000, 
                    pose_ref_img.size[1], pose_ref_img.size[0]
                ).to(self.device)

            if mode in ['a2v', 'a2mv', 'mv', 'i2v']:
                cond_pose_sequence_full = torch.zeros_like(cond_pose_sequence_full)
        else:
            curr_cond_image = kwargs.get('curr_cond_image',None)
            curr_pose_ref_img = kwargs.get('curr_pose_ref_img',None)
            cond_pose_sequence_full = kwargs.get('cond_pose_sequence_full',None)
        # 音频处理
        full_audio_embs = [audio_embedding]
        if zero_audio_embedding is not None:
            zero_full_audio_embs = [zero_audio_embedding]
        else:
            zero_full_audio_embs = [torch.zeros_like(audio_embedding)]

        # --- 2. 辅助变量与参数计算 ---
        indices = (torch.arange(2 * 2 + 1) - 2) * 1 
        if self.origin_mode:
            h, w = img.size[1], img.size[0]
            lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
            motion_h, motion_w = pose_ref_img.size[1], pose_ref_img.size[0]
            motion_lat_h, motion_lat_w = motion_h // self.vae_stride[1], motion_w // self.vae_stride[2]
        else:
            lat_h=kwargs.get('lat_h',None)
            lat_w=kwargs.get('lat_w',None)
            motion_lat_h=kwargs.get('motion_lat_h',None)
            motion_lat_w=kwargs.get('motion_lat_w',None)
        
        if self.origin_mode:
            # 预先编码固定的尾部条件
            with torch.no_grad():
                encoded_obj_image = self.vae.encode(
                    [obj_image.repeat(1, 1, 4, 1, 1).squeeze(0).to(torch.float32)]
                )[0].unsqueeze(0)
                encoded_small_img = self.vae.encode(
                    [small_img_tensor.repeat(1, 1, 4, 1, 1).squeeze(0).to(torch.float32)]
                )[0].unsqueeze(0)
                encoded_id_img = self.vae.encode([id_img.squeeze(0).to(torch.float32)])[0].unsqueeze(0)
                encoded_id_small_img = self.vae.encode([id_small_img.squeeze(0).to(torch.float32)])[0].unsqueeze(0)
        else:
            encoded_obj_image = kwargs.get('encoded_obj_image',None)
            encoded_small_img = kwargs.get('encoded_small_img',None)
            encoded_id_img = kwargs.get('encoded_id_img',None)
            encoded_id_small_img = kwargs.get('encoded_id_small_img',None)

        # 循环参数设定
        seg_frames = 101  
        target_latent_len = (seg_frames - 1) // self.vae_stride[0] + 1  # 26 latents
        total_model_seq_len = target_latent_len + 1 + 1 # 26 + 1(ID) + 1(Obj) = 28

        # --- 设置 Loop Logic 参数 ---
        prefix_latents_num = 5   # 后续段复用的 Latent 数量
        vae_temp_stride = 4      # VAE 时间维度的下采样率
        
        effective_new_latents = target_latent_len - prefix_latents_num 
        stride_frames = effective_new_latents * vae_temp_stride # 84帧
        
        total_video_len = min((audio_embedding.shape[0]-1)//4*4+1, max_frames_num//4*4+1)

        if total_video_len <= seg_frames:
            num_loops = 1
        else:
            num_loops = 1 + int(math.ceil((total_video_len - seg_frames) / stride_frames))
        
        print(f"Total Video Length: {total_video_len}, Num Loops: {num_loops}, Stride: {stride_frames}")
        if num_loops == 0: num_loops = 1
        if self.origin_mode:
            num_action = len(prompt_dict)
            print('before prompt_dict:', prompt_dict)
            prompt_dict = distribute_prompts_with_gaps(prompt_dict, num_loops)
            print('after prompt_dict:', prompt_dict)

            context_loop = []
            context_null_loop = []
            for loop_idx in range(num_loops):
                # start_idx = loop_idx * base + min(loop_idx, remainder)
                # end_idx = (loop_idx + 1) * base + min(loop_idx + 1, remainder)
                # current_chunk = struct_list[start_idx:end_idx]
                if loop_idx < num_loops:
                    input_prompt = prompt_dict[loop_idx]
                else:
                    input_prompt = prompt_dict[-1]
                # input_prompt = short_prompt + " ".join(current_chunk) + clear_state
                # 文本编码
                if not self.t5_cpu:
                    self.text_encoder.model.to(self.device)
                    context = self.text_encoder([input_prompt], self.device)
                    context_null = self.text_encoder([n_prompt], self.device)
                    if offload_model:
                        self.text_encoder.model.cpu()
                else:
                    context = self.text_encoder([input_prompt], torch.device('cpu'))
                    context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                    context = [t.to(self.device) for t in context]
                    context_null = [t.to(self.device) for t in context_null]
                context_loop.append(context)
                context_null_loop.append(context_null)
        else:
            context_loop=kwargs.get('context_loop',None)
            context_null_loop=kwargs.get('context_null_loop',None)

        max_seq_len = total_model_seq_len * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        
        motion_max_seq_len = total_model_seq_len * motion_lat_h * motion_lat_w \
                     // (self.patch_size[1] * self.patch_size[2])
        motion_max_seq_len = int(math.ceil(motion_max_seq_len / self.sp_size)) * self.sp_size

        # 随机种子
        seed = seed if seed >= 0 else random.randint(0, 99999999)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        gen_video_list = []
        gen_motion_list = []
        
        # 用于存储上一段的 Latent 尾部，作为下一段的前缀
        prev_loop_latent_tail = None
        prev_loop_motion_tail = None

        # --- 3. 自回归生成循环 (Autoregressive Loop) ---
        @contextmanager
        def noop_no_sync():
            yield
        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        dummy_noise = torch.randn(1, 48, 1, 1, 1, dtype=torch.float32, device=self.device)
        self.input_dtype = dummy_noise.dtype
        del dummy_noise

        current_start_frame = 0
        if not self.offload:
            gpu_manager=None
        else:
            gpu_manager = BlockGPUManager(device="cuda")
            gpu_manager.setup_for_inference(self.model)
        for loop_idx in range(num_loops):
            start_frame_idx = current_start_frame
            end_frame_idx = start_frame_idx + seg_frames
            
            # --- A. 准备前缀 Latent (Prefix) ---
            if loop_idx == 0:
                current_prefix_len = 1
                with torch.no_grad():
                    if self.origin_mode:
                        curr_img_in = curr_cond_image.unsqueeze(2)
                        encoded_prefix_img = self.vae.encode([curr_img_in.squeeze(0).to(torch.float32)])[0].unsqueeze(0)
                    else:
                        #print('before curr_cond_image:', curr_cond_image.shape)  #before curr_cond_image: torch.Size([1, 768, 512, 3])
                        encoded_prefix_img = self.vae.encode(curr_cond_image[:, :, :, :3])
                        encoded_prefix_img=get_mu_scale(encoded_prefix_img)[0].unsqueeze(0)
                        #print('after encoded_prefix_img:', encoded_prefix_img.shape)   ##after encoded_prefix_img: torch.Size([1, 48, 1, 48, 32])
                    
                    if self.origin_mode:
                        curr_pose_in = curr_pose_ref_img.unsqueeze(2)
                        encoded_prefix_motion = self.vae.encode([curr_pose_in.squeeze(0).to(torch.float32)])[0].unsqueeze(0)
                    else:
                        encoded_prefix_motion = self.vae.encode(curr_pose_ref_img[:, :, :, :3])
                        encoded_prefix_motion=get_mu_scale(encoded_prefix_motion)[0].unsqueeze(0)
           
            else:
                # 使用上一段 **像素回灌** 重编码后的 Latent
                current_prefix_len = prefix_latents_num
                encoded_prefix_img = prev_loop_latent_tail
                encoded_prefix_motion = prev_loop_motion_tail
            
            # --- B. 切片 Pose Sequence ---
            if self.origin_mode:
                cur_pose_seq_slice = cond_pose_sequence_full[:, :, start_frame_idx : end_frame_idx, :, :]
                if cur_pose_seq_slice.shape[2] < seg_frames:
                    pad_len = seg_frames - cur_pose_seq_slice.shape[2]
                    last_pose_frame = cur_pose_seq_slice[:, :, -1:, :, :]
                    cur_pose_seq_slice = torch.cat([cur_pose_seq_slice, last_pose_frame.repeat(1, 1, pad_len, 1, 1)], dim=2)
            else:
                cur_pose_seq_slice = cond_pose_sequence_full[start_frame_idx : end_frame_idx, :, :, :] #BHWC
                if cur_pose_seq_slice.shape[0] < seg_frames: #BHWC
                    pad_len = seg_frames - cur_pose_seq_slice.shape[0]
                    last_pose_frame = cur_pose_seq_slice[-1:, :, :, :]
                    cur_pose_seq_slice = torch.cat([cur_pose_seq_slice, last_pose_frame.repeat(pad_len, 1, 1, 1)], dim=0)   

            with torch.no_grad():
                if self.origin_mode:
                    encoded_pose_seq = self.vae.encode([cur_pose_seq_slice.squeeze(0).to(torch.float32)])[0].unsqueeze(0)
                else:
                    #print('before cur_pose_seq_slice:', cur_pose_seq_slice.shape) #before cur_pose_seq_slice: torch.Size([101, 384, 256, 3])
                    encoded_pose_seq = self.vae.encode(cur_pose_seq_slice[:, :, :, :3])
                    encoded_pose_seq=get_mu_scale(encoded_pose_seq)[0].unsqueeze(0)
                    #print('after cur_pose_seq_slice:', cur_pose_seq_slice.shape)

            cond_pose_sequence_loop = torch.cat([encoded_pose_seq, encoded_id_small_img, encoded_small_img], dim=2)

            # --- C. 切片 Audio Embeddings ---
            audio_indices_start = start_frame_idx 
            audio_indices_end = start_frame_idx + seg_frames 
            
            loop_audio_embs = []
            loop_zero_audio_embs = []
            
            for human_idx in range(len(full_audio_embs)):
                center_indices = torch.arange(
                    audio_indices_start, audio_indices_end, 1
                ).unsqueeze(1) + indices.unsqueeze(0)
                center_indices = torch.clamp(center_indices, min=0, max=full_audio_embs[human_idx].shape[0]-1)
                emb = full_audio_embs[human_idx][center_indices][None,...].to(self.device)
                loop_audio_embs.append(emb)
            loop_audio_embs = torch.cat(loop_audio_embs, dim=0).to(self.param_dtype)
            
            for human_idx in range(len(zero_full_audio_embs)):
                center_indices = torch.arange(
                    audio_indices_start, audio_indices_end, 1
                ).unsqueeze(1) + indices.unsqueeze(0)
                center_indices = torch.clamp(center_indices, min=0, max=zero_full_audio_embs[human_idx].shape[0]-1)
                emb = zero_full_audio_embs[human_idx][center_indices][None,...].to(self.device)
                loop_zero_audio_embs.append(emb)
            loop_zero_audio_embs = torch.cat(loop_zero_audio_embs, dim=0).to(self.param_dtype)

            if mode in ['p2v', 'mv', 'i2v']:
                loop_audio_embs = loop_zero_audio_embs

            # --- D. 初始化噪声 (Latents) ---
            noise = torch.randn(
                1, 48, total_model_seq_len,
                lat_h, lat_w,
                dtype=torch.float32, device=self.device)
            
            motion_noise = torch.randn(
                1, 48, total_model_seq_len,
                motion_lat_h, motion_lat_w,
                dtype=torch.float32, device=self.device)

            self.input_dtype = noise.dtype
            cur_dtype=torch.bfloat16
            
            # --- E. 准备扩散模型输入 ---
            if not self.offload:
                if self.model.device != self.device:
                    self.model.to(self.device)
            latent = noise
            latent_motion = motion_noise
            
            latent[:, :, :current_prefix_len] = encoded_prefix_img
            latent_motion[:, :, :current_prefix_len] = encoded_prefix_motion
            
            latent[:, :, -2:-1] = encoded_id_img
            latent[:, :, -1:] = encoded_obj_image
            
            latent_motion[:, :, -2:-1] = encoded_id_small_img
            latent_motion[:, :, -1:] = encoded_small_img

            cond_pose_sequence_loop = cond_pose_sequence_loop.to(latent_motion.dtype).to(self.device)
            loop_audio_embs = loop_audio_embs.to(latent_motion.dtype).to(self.device)

            arg_c = {
                'context_list': [context_loop[loop_idx][0].to(cur_dtype)],
                'seq_len': max_seq_len,
                'seq_len_motion': motion_max_seq_len,
                'mode': mode,
                'skip_block': False,
                'audio_embedding': loop_audio_embs,
                'prefix_len': current_prefix_len
            }
            arg_null_text = {
                'context_list': [context_null_loop[loop_idx][0].to(cur_dtype)],
                'seq_len': max_seq_len ,
                'seq_len_motion': motion_max_seq_len,
                'mode': mode,
                'skip_block': bad_cfg,
                'audio_embedding': loop_audio_embs * 0,
                'prefix_len': current_prefix_len
            }
            arg_pure_audio = {
                'context_list': [context_null_loop[loop_idx][0].to(cur_dtype)], 
                'seq_len': max_seq_len,
                'seq_len_motion': motion_max_seq_len,
                'mode': mode,
                'skip_block': False,
                'audio_embedding': loop_audio_embs,
                'prefix_len': current_prefix_len
            }

            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]

            # --- F. 采样循环 ---
            with torch.no_grad(), no_sync():
                progress_wrap = partial(
                    tqdm, 
                    total=len(timesteps)-1, 
                    desc=f"Loop {loop_idx} (Prefix={current_prefix_len})"
                )
                for i in progress_wrap(range(len(timesteps)-1)):
                    timestep = timesteps[i]
                    zero_timestep = torch.zeros_like(timestep).to(dtype=timestep.dtype, device=timestep.device)
                    
                    latent_model_input = latent.to(self.device, dtype=cur_dtype)
                    latent_model_input_motion = latent_motion.to(self.device,dtype=cur_dtype)

                    if timestep[0] > bad_thres:
                        arg_null_text['skip_block'] = bad_cfg
                    else:
                        arg_null_text['skip_block'] = False
                    
                    if three_cfg and timestep[0] > bad_thres:
                        arg_pure_audio['skip_block'] = bad_cfg

                    motion_input = latent_model_input_motion if mode in ['mv','a2mv'] else cond_pose_sequence_loop
                    motion_time = timestep if mode in ['mv','a2mv'] else zero_timestep
                    
                    # Model Forward
                    if three_cfg:
                        noise_pred_cond, noise_pred_cond_motion = self.model(
                            x=latent_model_input, motion=motion_input, t=timestep, motion_t=motion_time, **arg_c,gpu_manager=gpu_manager)
                        torch_gc()
                        noise_pred_drop_text, noise_pred_drop_text_motion = self.model(
                            x=latent_model_input, 
                            motion=motion_input, 
                            t=timestep, 
                            motion_t=motion_time, 
                            **arg_null_text,gpu_manager=gpu_manager
                        )
                        torch_gc()
                        noise_pred_pure_audio, noise_pred_pure_audio_motion = self.model(
                            x=latent_model_input, 
                            motion=motion_input, 
                            t=timestep, 
                            motion_t=motion_time, 
                            **arg_pure_audio,gpu_manager=gpu_manager
                        )
                        torch_gc()
                        
                        noise_pred = noise_pred_drop_text + \
                                    text_guide_scale * (noise_pred_cond - noise_pred_pure_audio) + \
                                    audio_guide_scale * (noise_pred_pure_audio - noise_pred_drop_text)
                        noise_pred_motion = noise_pred_drop_text_motion + text_guide_scale * (
                            noise_pred_cond_motion - noise_pred_drop_text_motion
                        )
                    else:
                        noise_pred_cond, noise_pred_cond_motion = self.model(
                            x=latent_model_input, motion=motion_input, t=timestep, motion_t=motion_time, **arg_c,gpu_manager=gpu_manager)
                        torch_gc()
                        noise_pred_drop_text, noise_pred_drop_text_motion = self.model(
                            x=latent_model_input, motion=motion_input, t=timestep, 
                            motion_t=motion_time, **arg_null_text,gpu_manager=gpu_manager)
                        torch_gc()
                        
                        noise_pred = noise_pred_drop_text + text_guide_scale * (noise_pred_cond - noise_pred_drop_text)
                        noise_pred_motion = noise_pred_drop_text_motion + text_guide_scale * (
                            noise_pred_cond_motion - noise_pred_drop_text_motion
                        )

                    noise_pred = -noise_pred
                    noise_pred_motion = -noise_pred_motion

                    dt = timesteps[i] - timesteps[i + 1]
                    dt = dt / self.num_timesteps
                    latent = latent + noise_pred * dt[:, None, None]
                    latent_motion = latent_motion + noise_pred_motion * dt[:, None, None]

                    # 强约束：重置 Prefix 和 End Conditions
                    latent[:, :, :current_prefix_len] = encoded_prefix_img
                    latent_motion[:, :, :current_prefix_len] = encoded_prefix_motion
                    
                    latent[:, :, -2:-1] = encoded_id_img
                    latent[:, :, -1:] = encoded_obj_image
                    latent_motion[:, :, -2:-1] = encoded_id_small_img
                    latent_motion[:, :, -1:] = encoded_small_img

   
            torch_gc()

            latent_to_decode = latent[:, :, :-2].to(self.device)
            
            if mode not in ['mv','a2mv']:
                latent_motion_to_decode = cond_pose_sequence_loop[:, :, :-2].to(self.device)
            else:
                latent_motion_to_decode = latent_motion[:, :, :-2].to(self.device)
            with torch.no_grad():
                if self.origin_mode:
                    decoded_video_seg = self.vae.decode(latent_to_decode)[0] 
                    decoded_motion_seg = self.vae.decode(latent_motion_to_decode)[0] 
                else:
                    #print(latent_to_decode.shape) #torch.Size([1, 48, 26, 48, 32])

                    latent_to_decode=get_z_scale(latent_to_decode) # apply scale
                    latent_motion_to_decode=get_z_scale(latent_motion_to_decode)
    
                    decoded_video_seg = self.vae.decode(latent_to_decode)[0]  # need check
                    decoded_motion_seg = self.vae.decode(latent_motion_to_decode)[0] 
                    #print(decoded_video_seg.shape) #torch.Size([101, 768, 512, 3])

            # --- I. [修复] 正确的 Prefix 准备：像素切片 -> 重编码 ---
            # 只有当不是最后一个 Loop 时才需要准备
            if prefix_latents_num > 0 and loop_idx < num_loops - 1:
                # 5 latents * 4 stride -> 覆盖范围约 17 帧 ( (17-1)//4 + 1 = 5 )
                overlap_pixel_len = 17 
                
                # 取出生成的最后 17 帧 (decoded_video_seg 形状为 [C, F, H, W])
                # 注意增加 batch 维度以适配 encode 接口
                if  self.origin_mode:
                    pixel_tail_video = decoded_video_seg[:, -overlap_pixel_len:, :, :].unsqueeze(0)
                    pixel_tail_motion = decoded_motion_seg[:, -overlap_pixel_len:, :, :].unsqueeze(0)
                    first_pixel_tail_video = decoded_video_seg[:, -17:-16, :, :].unsqueeze(0)
                    first_pixel_tail_motion = decoded_motion_seg[:, -17:-16, :, :].unsqueeze(0)
                else:
                    pixel_tail_video = decoded_video_seg[-overlap_pixel_len:, :, :,:]
                    pixel_tail_motion = decoded_motion_seg[ -overlap_pixel_len:, :, :,:]
                    first_pixel_tail_video = decoded_video_seg[ -17:-16, :, :,:]
                    first_pixel_tail_motion = decoded_motion_seg[ -17:-16, :, :,:]
                    #print(pixel_tail_video.shape, pixel_tail_motion.shape, first_pixel_tail_video.shape, first_pixel_tail_motion.shape) #torch.Size([17, 768, 512, 3]) torch.Size([17, 384, 256, 3]) torch.Size([1, 768, 512, 3]) torch.Size([1, 384, 256, 3])
                with torch.no_grad():
                    # 重新编码，结果赋值给 prev_loop_... 供下一次循环使用
                    # encode 返回 list[Tensor], 取 [0] -> Tensor[C, T, H, W], unsqueeze(0) -> [1, C, T, H, W]
                    if self.origin_mode:
                        prev_loop_latent_tail = self.vae.encode(
                            [pixel_tail_video.squeeze(0).to(torch.float32)]
                        )[0].unsqueeze(0)
                        prev_loop_motion_tail = self.vae.encode(
                            [pixel_tail_motion.squeeze(0).to(torch.float32)]
                        )[0].unsqueeze(0)
                        prev_loop_first_latent_tail = self.vae.encode(
                            [first_pixel_tail_video.squeeze(0).to(torch.float32)]
                        )[0].unsqueeze(0)
                        prev_loop_first_motion_tail = self.vae.encode(
                            [first_pixel_tail_motion.squeeze(0).to(torch.float32)]
                        )[0].unsqueeze(0)
                    else:
                        prev_loop_latent_tail = self.vae.encode(pixel_tail_video)
                        prev_loop_latent_tail=get_mu_scale(prev_loop_latent_tail)[0].unsqueeze(0)
                        #print(prev_loop_latent_tail.shape) #torch.Size([1, 48, 5, 48, 32])

                        prev_loop_motion_tail = self.vae.encode(pixel_tail_motion) # scale
                        prev_loop_motion_tail=get_mu_scale(prev_loop_motion_tail)[0].unsqueeze(0)
                        #print(prev_loop_motion_tail.shape)#torch.Size([1, 48, 5, 24, 16])

                        prev_loop_first_latent_tail = self.vae.encode(first_pixel_tail_video)
                        prev_loop_first_latent_tail=get_mu_scale(prev_loop_first_latent_tail)[0].unsqueeze(0)
                        #print(prev_loop_first_latent_tail.shape) #torch.Size([1, 48, 1, 48, 32])

                        prev_loop_first_motion_tail = self.vae.encode(first_pixel_tail_motion)
                        prev_loop_motion_tail=get_mu_scale(prev_loop_motion_tail)[0].unsqueeze(0)
                        #print(prev_loop_motion_tail.shape) #torch.Size([1, 48, 5, 24, 16])

                    prev_loop_latent_tail[:, :, 0:1] = prev_loop_first_latent_tail
                    prev_loop_motion_tail[:, :, 0:1] = prev_loop_first_motion_tail

            # --- 结果拼接逻辑 ---
            if loop_idx == 0:
                gen_video_list.append(decoded_video_seg.cpu()) #torch.Size([101, 768, 512, 3])
                gen_motion_list.append(decoded_motion_seg.cpu())
                current_start_frame += stride_frames 
            else:
                frames_to_keep = stride_frames
                if self.origin_mode:
                    keep_start_idx = max(0, decoded_video_seg.shape[1] - frames_to_keep)
                    gen_video_list.append(decoded_video_seg[:, keep_start_idx:, :, :].cpu())
                    gen_motion_list.append(decoded_motion_seg[:, keep_start_idx:, :, :].cpu())
                else:
                    keep_start_idx = max(0, decoded_video_seg.shape[0] - frames_to_keep)
                    gen_video_list.append(decoded_video_seg[keep_start_idx:, :, :, :].cpu())
                    gen_motion_list.append(decoded_motion_seg[keep_start_idx:, :, :, :].cpu())
                
                current_start_frame += stride_frames

            del latent, latent_motion, noise, motion_noise
            torch_gc()
        if self.offload:
            gpu_manager.unload_all_blocks_to_cpu()
            torch_gc()
        else:
            if offload_model: 
                self.model.cpu()
        # --- 4. 结果拼接 (Concatenation) ---
        if self.origin_mode:
            full_video = torch.cat(gen_video_list, dim=1)
            full_motion = torch.cat(gen_motion_list, dim=1)
        else:
            full_video = torch.cat(gen_video_list, dim=0)
            full_motion = torch.cat(gen_motion_list, dim=0)
        #print(f"Generation Finished. Output Shape: {full_video.shape}") 
        #print(f"Generation Finished. Output Shape: {full_motion.shape}") 
        if offload_model:    
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        
        return full_video, full_motion if self.rank == 0 else None




mean = torch.tensor(
            [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ],
            device=torch.device('cuda'),
        )
std = torch.tensor(
            [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ],
            device=torch.device('cuda'),
        )
wanvae_scale = [mean, 1.0 / std]

def get_mu_scale(mu):
    mu = (mu - wanvae_scale[0].view(1, 48, 1, 1, 1).to(mu.device,mu.dtype)) * wanvae_scale[1].view(
                        1, 48, 1, 1, 1).to(mu.device,mu.dtype)
    return mu
def get_z_scale(z):
    z = z / wanvae_scale[1].view(1, 48, 1, 1, 1).to(z.device,z.dtype) + wanvae_scale[0].view(
                    1, 48, 1, 1, 1).to(z.device,z.dtype)
    return z