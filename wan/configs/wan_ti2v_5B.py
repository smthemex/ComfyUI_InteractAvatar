# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict
from .shared_config import wan_shared_cfg

#------------------------ Wan TI2V 5B ------------------------#

ti2v_5B = EasyDict(__name__='Config: Wan TI2V 5B')
ti2v_5B.update(wan_shared_cfg)

# t5
ti2v_5B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
ti2v_5B.t5_tokenizer = 'google/umt5-xxl'

# vae
ti2v_5B.vae_checkpoint = 'Wan2.2_VAE.pth'
ti2v_5B.motion_vae_checkpoint = 'Wan2.1_VAE.pth'
ti2v_5B.vae_stride = (4, 16, 16)

# transformer
ti2v_5B.patch_size = (1, 2, 2)
ti2v_5B.dim = 3072
ti2v_5B.ffn_dim = 14336
ti2v_5B.freq_dim = 256
ti2v_5B.text_dim = 4096
ti2v_5B.in_dim = 48  # Wan2.2 TI2V 5B uses 48-dim input (from Wan2.2_VAE)
ti2v_5B.out_dim = 48  # Wan2.2 TI2V 5B uses 48-dim output (to Wan2.2_VAE)
ti2v_5B.num_heads = 24
ti2v_5B.num_layers = 30
ti2v_5B.window_size = (-1, -1)
ti2v_5B.qk_norm = True
ti2v_5B.cross_attn_norm = True
ti2v_5B.eps = 1e-6
# Wan2.2 specific features
ti2v_5B.seperated_timestep = True
ti2v_5B.fuse_vae_embedding_in_latents = True
ti2v_5B.require_clip_embedding = False
ti2v_5B.require_vae_embedding = False

ti2v_5B.audio_dim = 1536
ti2v_5B.audio_ffn_dim = 8960
ti2v_5B.audio_num_heads = 12
ti2v_5B.audio_num_layers = 30

# inference
ti2v_5B.sample_fps = 24
ti2v_5B.sample_shift = 5.0
ti2v_5B.sample_steps = 50
ti2v_5B.sample_guide_scale = 5.0
ti2v_5B.frame_num = 121

ti2v_5B.audio_embedder = "audio/fc_model.safetensors"
ti2v_5B.audio_encoder = "audio/whisper-tiny"
ti2v_5B.bigvgan = "audio/code2wav_bigvgan_model"
ti2v_5B.audio_mean = -5.1753
ti2v_5B.audio_std = 2.1544
ti2v_5B.sampling_rate = 24000

