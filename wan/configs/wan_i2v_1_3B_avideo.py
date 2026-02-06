# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan T2V 1.3B ------------------------#

i2v_1_3B_avideo = EasyDict(__name__='Config: Wan I2V 1.3B Audio')
i2v_1_3B_avideo.update(wan_shared_cfg)

# t5
i2v_1_3B_avideo.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
i2v_1_3B_avideo.t5_tokenizer = 'google/umt5-xxl'

# clip
i2v_1_3B_avideo.clip_model = 'clip_xlm_roberta_vit_h_14'
i2v_1_3B_avideo.clip_dtype = torch.float16
i2v_1_3B_avideo.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
i2v_1_3B_avideo.clip_tokenizer = 'xlm-roberta-large'

# vae
i2v_1_3B_avideo.vae_checkpoint = 'Wan2.1_VAE.pth'
i2v_1_3B_avideo.vae_stride = (4, 8, 8)

# audio
i2v_1_3B_avideo.audio_embedder = "audio/fc_model.safetensors"
i2v_1_3B_avideo.audio_encoder = "audio/whisper-tiny"
i2v_1_3B_avideo.bigvgan = "audio/code2wav_bigvgan_model"
i2v_1_3B_avideo.audio_mean = -5.1753
i2v_1_3B_avideo.audio_std = 2.1544
i2v_1_3B_avideo.sampling_rate = 24000

# transformer
i2v_1_3B_avideo.patch_size = (1, 2, 2)
i2v_1_3B_avideo.dim = 1536
i2v_1_3B_avideo.ffn_dim = 8960
i2v_1_3B_avideo.freq_dim = 256
i2v_1_3B_avideo.num_heads = 12
i2v_1_3B_avideo.num_layers = 30
i2v_1_3B_avideo.window_size = (-1, -1)
i2v_1_3B_avideo.qk_norm = True
i2v_1_3B_avideo.cross_attn_norm = True
i2v_1_3B_avideo.eps = 1e-6
i2v_1_3B_avideo.text_len = 512
