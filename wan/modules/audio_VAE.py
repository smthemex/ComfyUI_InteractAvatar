import sys
sys.path.append('/apdcephfs_cq10/share_1367250/terohu/code/ARwan')
import logging
import os
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path
from utils.wanx_audio_dataset import VideoAudioTextLoader
import pandas as pd
import tensordict as td
import torch
import torch.distributed as distributed
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from mmaudio.ext.autoencoder import AutoEncoderModule
from mmaudio.ext.mel_converter import get_mel_converter
import torchaudio

log = logging.getLogger()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 16k
SAMPLE_RATE = 16_000
NUM_SAMPLES = 16_000 * 8
tod_vae_ckpt = '/apdcephfs_cq10/share_1367250/terohu/code/MMAudio/ext_weights/v1-16.pth'
bigvgan_vocoder_ckpt = '/apdcephfs_cq10/share_1367250/terohu/code/MMAudio/ext_weights/best_netG.pt'
mode = '16k'

# 44k
"""
NOTE: 352800 (8*44100) is not divisible by (STFT hop size * VAE downsampling ratio) which is 1024.
353280 is the next integer divisible by 1024.
"""



@torch.inference_mode()
def main():

    # 16k
    tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                            vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                            mode=mode).eval().cuda()
    # 44k
    # mel_converter = get_mel_converter(mode).eval().cuda()
    # waveforms=1
    # mel = mel_converter(waveforms)
    # dist = tod.encode(mel)

    a_mean = dist.mean.detach().cpu().transpose(1, 2)
    a_std = dist.std.detach().cpu().transpose(1, 2)
    latent=a_mean+a_std*torch.randn_like(a_mean).to(a_mean)
    mel=tod.decode(latent.transpose(1, 2))
    audios=tod.vocode(mel)
    audio = audios.float().cpu()[0]
    torchaudio.save('tmp.wav', audio, 16000)




