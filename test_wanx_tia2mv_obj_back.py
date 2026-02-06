
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import sys
import warnings
import pyloudnorm as pyln
import librosa
warnings.filterwarnings('ignore')
import numpy as np
import torch
from PIL import Image
import librosa
import re
import math
#import wan
from .wan.tia2mv_obj_back_id_prefix import WanTIA2MVRefBackIDPrefix,distribute_prompts_with_gaps
from einops import rearrange
from .utils.audio_analysis.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
from .wan.configs import WAN_CONFIGS
from .utils.img_utils import process_images_final,resize_images,resize_short_side
from .model_loader_utils import covert_obj_img,clear_comfyui_cache

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

def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio
def align_floor_to(value, alignment):
    return int(math.floor(value / alignment) * alignment)
def align_ceil_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)
def custom_init(device, wav2vec):    
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder


def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25 # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    return audio_emb

def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def load_model(checkpoint_path,dit_path,lora_path,short_side=512,back_append_frame=1,offload=False):
    logging.info("Creating WanI2V pipeline.")
    cfg = WAN_CONFIGS['ti2v-5B']
    wan_a2v = WanTIA2MVRefBackIDPrefix(
        config=cfg,
        checkpoint_dir=checkpoint_path,
        transformer_dir=dit_path,
        lora_path=lora_path,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,#(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=True,
        load_from_merged_model=None,
        short_side=short_side,
        back_append_frame=back_append_frame,
        offload=offload,
    )
    return wan_a2v


def generate_video(wan_a2v,gen_kwargs):
    logging.info("Generating video ...")

    if gen_kwargs["back_append_frame"] == 1:
        video, _ = wan_a2v.generate(
            None, None, None, None, None, **gen_kwargs
        )
    else:
        wan_a2v.vae=gen_kwargs["vae"]
        video, _ = wan_a2v.generate_long(
            None, None, None, None, None, **gen_kwargs
        )

    return video


def perdata( clip,vae,images,dw_iamges,object_images,object_mask,audio_path,mode,prompt,negative_prompt,structured_prompt,frame_num,short_side,back_append_frame,
            wav2vec_dir,device,all_text=False,clips=None,max_frames_num=1000,
            ):
 
    img=images[0]
    dw_img=dw_iamges[0]

    if len(dw_iamges) >1:
        dw_seqs = dw_iamges
        if clips is not None:
            #clips = batch['clips']
            dw_seqs = dw_iamges[clips[0]:clips[1]]
    else:
        dwpose_frame_num = frame_num
        dw_seqs = None
  

    # pre size
    w, h = img.size
    img = resize_short_side(img, short_side)
    small_img = resize_short_side(img, 256)
    dw_img = resize_short_side(dw_img, 256)
    if short_side == 512:
        _, small_img = resize_images(img, small_img)
        img, dw_img = resize_images(img, dw_img)
    else:
        _, small_img = process_images_final(img, small_img)
        img, dw_img = process_images_final(img, dw_img)

    dw_w, dw_h = dw_img.size
    if small_img.size != dw_img.size:
        small_img = img.resize(dw_img.size, Image.LANCZOS)
    if dw_seqs is not None:
        dw_seqs = [dw.resize(dw_img.size, Image.LANCZOS) for dw in dw_seqs]
        dwpose_len = len(dw_seqs)
        dwpose_frame_num = (dwpose_len - 1) // 4 * 4 + 1

    # pre audio
    if audio_path is not None:
        audio_input, sampling_rate = librosa.load(audio_path, sr=16000)
        audio_frames_clip = loudness_norm(audio_input, sampling_rate)
        audio_frame_len = int(len(audio_frames_clip) / sampling_rate * 25)
        audio_frame_num = (audio_frame_len - 1) // 4 * 4 + 1
    else:
        audio_frames_clip = np.zeros((int(dwpose_frame_num / 25 * 16000)))
        audio_frame_num = dwpose_frame_num
        sampling_rate = 16000

    if dw_seqs is not None:
        dw_seqs = dw_seqs[:frame_num]
        if audio_frame_num > dwpose_frame_num:
            padding_dwpose = dw_seqs[-1]
            dw_seqs = dw_seqs + [padding_dwpose] * (audio_frame_num - dwpose_frame_num)
            dwpose_frame_num = audio_frame_num
        if audio_frame_num < dwpose_frame_num:
            padding_audio = np.zeros((int(dwpose_frame_num / 25 * 16000)))
            audio_frames_clip = np.concatenate([audio_frames_clip, padding_audio], axis=0)
            audio_frame_num = dwpose_frame_num
    if dw_seqs is None and mode == 'ap2v':
        dw_seqs = [dw_img] * audio_frame_num

    frame_num = min(frame_num,min(audio_frame_num,dwpose_frame_num))
    audio_frames_clip = audio_frames_clip[:int(frame_num * sampling_rate / 25)]

    wav2vec_feature_extractor, audio_encoder= custom_init(device, wav2vec_dir)

    zero_audio_embedding = get_embedding(np.zeros_like(audio_frames_clip), wav2vec_feature_extractor, audio_encoder, device=device) 
    audio_embedding = get_embedding(audio_frames_clip, wav2vec_feature_extractor, audio_encoder, device=device)

    # obj_img_path = obj_img
    obj_img = covert_obj_img(object_images,object_mask,target_size=img.size) #RGBA

    caption_prompt_list = []
    clear_state = " clear hands and face, objects are clear and stable. human movements are slow and steady, with a strong sense of reality."
    if back_append_frame == 1: 
        segments = re.findall(r'\(.*?\)', structured_prompt)
        if all_text:
            first_three_segments = segments
        else:
            first_three_segments = segments[:3]
        print(first_three_segments)
        result_string = "".join(first_three_segments)
        caption_prompt = prompt + clear_state + result_string.lower().replace(')(',') (')
    else:
        for i in range(len(structured_prompt)):
            segments = re.findall(r'\(.*?\)', structured_prompt[i])
            if all_text:
                first_three_segments = segments
            else:
                first_three_segments = segments[:3]
            print(first_three_segments)
            result_string = "".join(first_three_segments)
            caption_prompt = prompt[i] + clear_state + result_string.lower().replace(')(',') (')
            caption_prompt_list.append(caption_prompt)
    if mode in ['a2v','a2mv','mv','i2v']:
        dw_seqs = None

    vae_stride=WAN_CONFIGS['ti2v-5B'].vae_stride 
    patch_size=WAN_CONFIGS['ti2v-5B'].patch_size 
    sp_size=1

    if  back_append_frame==1:

        cond_image=phi2narry(img) #BHWC
        obj_image=phi2narry(obj_img)
        pose_ref_img=phi2narry(dw_img)
        small_img=phi2narry(small_img)
        #print(cond_image.shape, obj_image.shape, pose_ref_img.shape, small_img.shape) #torch.Size([1, 768, 512, 3]) torch.Size([1, 768, 512, 3]) torch.Size([1, 384, 256, 3]) torch.Size([1, 384, 256, 3])
        # 2
        
        if dw_seqs is not None:
            if isinstance(dw_seqs, list): # If input is a list of PIL Images
                processed_poses = [phi2narry(p) for p in dw_seqs]
                cond_pose_sequence = torch.stack(processed_poses).to(device)
            else: # If input is already a tensor
                cond_pose_sequence = dw_seqs.to(device)
            dwpose_len = cond_pose_sequence.shape[0]
            dwpose_len = (dwpose_len - 1) // vae_stride[0] * vae_stride[0] + 1
            frame_num = min(frame_num, dwpose_len)
            cond_pose_sequence = cond_pose_sequence[:frame_num]

        else:
            cond_pose_sequence = torch.ones((frame_num, pose_ref_img.shape[1], pose_ref_img.shape[2], pose_ref_img.shape[3]), device=pose_ref_img.device, dtype=pose_ref_img.dtype) * 0.5

        if mode in ['a2v','a2mv','mv','i2v']:
            if dw_seqs is not None:
                cond_pose_sequence = torch.ones((frame_num, cond_pose_sequence.shape[1], cond_pose_sequence.shape[2], cond_pose_sequence.shape[-1]), device=cond_pose_sequence.device, dtype=cond_pose_sequence.dtype) * 0.5
        # 3 
        if back_append_frame==1:
            tokens_p = clip.tokenize(caption_prompt)
            context = clip.encode_from_tokens_scheduled(tokens_p)[0][0].to(device,torch.bfloat16) #use bf16 #torch.Size([1, 512, 4096])
            tokens_n = clip.tokenize(negative_prompt)
            context_null =  clip.encode_from_tokens_scheduled(tokens_n)[0][0].to(device,torch.bfloat16)
        else:
            pass 
        clear_comfyui_cache()   
        # 4
        h, w = cond_image.shape[1], cond_image.shape[2]  #cf BHWC
        lat_h, lat_w = h // vae_stride[1], w // vae_stride[2]
        max_seq_len = ((frame_num - 1) // vae_stride[0] + 1) * lat_h * lat_w // (
            patch_size[1] * patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / sp_size)) * sp_size

        
        with torch.no_grad():
            cond_image =vae.encode(cond_image[:, :, :, :3]) 
            cond_image = get_mu_scale(cond_image)[0].unsqueeze(0) #torch.Size([1, 48, 1, 48, 32]) 
            obj_image=vae.encode(obj_image.repeat( 4, 1,1,1)[:, :, :, :3])
            obj_image = get_mu_scale(obj_image)[0].unsqueeze(0)
            small_img=vae.encode(small_img.repeat( 4, 1, 1,1)[:, :, :, :3]) 
            small_img = get_mu_scale(small_img)[0].unsqueeze(0)
        # 5
        cond_dw_img = pose_ref_img # Shape: B, C, 1, H, W. (1, C, N, H, W)

        motion_h, motion_w = cond_dw_img.shape[1], cond_dw_img.shape[2] #cf BHWC

        motion_lat_h, motion_lat_w = motion_h // vae_stride[1], motion_w // vae_stride[2]
        motion_max_seq_len = ((frame_num - 1) // vae_stride[0] + 1) * motion_lat_h * motion_lat_w // (
                        patch_size[1] * patch_size[2])
        motion_max_seq_len = int(math.ceil(motion_max_seq_len / sp_size)) * sp_size
        with torch.no_grad():
            cond_dw_img=vae.encode(cond_dw_img[:, :, :, :3])
            cond_dw_img = get_mu_scale(cond_dw_img)[0].unsqueeze(0)
            cond_pose_sequence=vae.encode(cond_pose_sequence[:, :, :, :3])
            cond_pose_sequence = get_mu_scale(cond_pose_sequence)[0].unsqueeze(0)

        data_dict=dict(
            context_null=context_null,
            context=context,
            lat_h=lat_h,
            lat_w=lat_w,
            max_seq_len=max_seq_len,
            motion_lat_h=motion_lat_h,
            motion_lat_w=motion_lat_w,
            motion_max_seq_len=motion_max_seq_len,
            cond_image=cond_image,
            obj_image=obj_image,
            small_image=small_img,
            cond_dw_img=cond_dw_img,
            cond_pose_sequence=cond_pose_sequence,
            audio_embedding=audio_embedding,
            zero_audio_embedding=zero_audio_embedding,
            back_append_frame=back_append_frame,
            mode=mode,
            frame_num=frame_num,
            max_frames_num=frame_num,
            )   
    else:
        curr_cond_image=phi2narry(img) #BHWC
        curr_pose_ref_img=phi2narry(dw_img) #BCHW

        # 固定的尾帧条件 (Fixed End Condition)
        obj_image=phi2narry(obj_img)
        small_img_tensor=phi2narry(small_img)
        id_img=phi2narry(img)#BHWC use img as id_img
        id_small_img=phi2narry(small_img) #BHWC


        # Pose Sequence 处理
        if dw_seqs is not None:
            if isinstance(dw_seqs, list): 
                processed_poses = [phi2narry(p) for p in dw_seqs]
                cond_pose_sequence_full = torch.stack(processed_poses).to(device)
            else: 
                cond_pose_sequence_full = dw_seqs.to(device)
        else:
            cond_pose_sequence_full = torch.ones((frame_num+ 2000, curr_pose_ref_img.shape[1], curr_pose_ref_img.shape[2], curr_pose_ref_img.shape[3]), device=curr_pose_ref_img.device, dtype=curr_pose_ref_img.dtype) * 0.5


        if mode in ['a2v', 'a2mv', 'mv', 'i2v']:
            if dw_seqs is not None:
                cond_pose_sequence_full = torch.ones((cond_pose_sequence_full.shape[0], cond_pose_sequence_full.shape[1], cond_pose_sequence_full.shape[2], cond_pose_sequence_full.shape[3]), device=cond_pose_sequence_full.device, dtype=cond_pose_sequence_full.dtype) * 0.5

        h, w = curr_cond_image.shape[1], curr_cond_image.shape[2]  #cf BHWC
        lat_h, lat_w = h // vae_stride[1], w // vae_stride[2]

        motion_h, motion_w = curr_pose_ref_img.shape[1], curr_pose_ref_img.shape[2]

        motion_lat_h, motion_lat_w = motion_h // vae_stride[1], motion_w // vae_stride[2]

        with torch.no_grad():

            encoded_obj_image=vae.encode(obj_image.repeat(4, 1, 1,1)[:, :, :, :3])
            encoded_obj_image = get_mu_scale(encoded_obj_image)[0].unsqueeze(0)

            encoded_small_img=vae.encode(small_img_tensor.repeat(4, 1, 1,1)[:, :, :, :3])
            encoded_small_img = get_mu_scale(encoded_small_img)[0].unsqueeze(0) # torch.Size([1, 48, 1, 24, 16]) encoded_obj_image.shape: torch.Size([1, 48, 1, 48, 32])

            encoded_id_img=vae.encode(id_img[:, :, :, :3])
            encoded_id_img = get_mu_scale(encoded_id_img)[0].unsqueeze(0)

            encoded_id_small_img=vae.encode(id_small_img[:, :, :, :3])
            encoded_id_small_img = get_mu_scale(encoded_id_small_img)[0].unsqueeze(0) #torch.Size([1, 48, 1, 48, 32]) encoded_id_small_img.shape: torch.Size([1, 48, 1, 24, 16])

        # 循环参数设定
        seg_frames = 101  
        target_latent_len = (seg_frames - 1) // vae_stride[0] + 1  # 26 latents
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

        prompt_dict=caption_prompt_list

        num_action = len(prompt_dict) #TODO: 
        #print('before prompt_dict:', prompt_dict)
        prompt_dict = distribute_prompts_with_gaps(prompt_dict, num_loops)
        print('after prompt_dict:', prompt_dict)

        context_loop = []
        context_null_loop = []
        for loop_idx in range(num_loops):

            if loop_idx < num_loops:
                input_prompt = prompt_dict[loop_idx]
            else:
                input_prompt = prompt_dict[-1]
            # input_prompt = short_prompt + " ".join(current_chunk) + clear_state
            # 文本编码
            tokens_p = clip.tokenize(input_prompt)
            context = clip.encode_from_tokens_scheduled(tokens_p)[0][0].to(device,torch.bfloat16) 
            tokens_n = clip.tokenize(negative_prompt)
            context_null =  clip.encode_from_tokens_scheduled(tokens_n)[0][0].to(device,torch.bfloat16)

            context_loop.append(context)
            context_null_loop.append(context_null)

        data_dict=dict(
            context_loop=context_loop,
            context_null_loop=context_null_loop,
            lat_h=lat_h,
            lat_w=lat_w,
            motion_lat_h=motion_lat_h,
            motion_lat_w=motion_lat_w,
            encoded_small_img=encoded_small_img,
            encoded_id_img=encoded_id_img,
            encoded_id_small_img=encoded_id_small_img,
            encoded_obj_image=encoded_obj_image,
            curr_cond_image=curr_cond_image,
            curr_pose_ref_img=curr_pose_ref_img,
            cond_pose_sequence_full=cond_pose_sequence_full,
            audio_embedding=audio_embedding,
            zero_audio_embedding=zero_audio_embedding,
            back_append_frame=back_append_frame,
            mode=mode,
            frame_num=frame_num,
            vae=vae,
            max_frames_num=frame_num,
            )   
    return data_dict
