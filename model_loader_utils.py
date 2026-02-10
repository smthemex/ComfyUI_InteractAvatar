# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
from PIL import Image
import numpy as np
import math
import comfy.utils
import cv2
import random
import torchaudio
import folder_paths
from comfy.utils import common_upscale,ProgressBar
from safetensors.torch import load_file
import soundfile as sf 
import comfy.model_management as mm
from pathlib import PureWindowsPath
cur_path = os.path.dirname(os.path.abspath(__file__))


def covert_obj_img(images, masks,target_size):
    background_color=(255, 255, 255)
    if images is not None  and  masks is not None:
        output_list = []

        # Convert to numpy for easier processing
        images_np = images.cpu().numpy()
        
        # Handle mask shapes
        if masks.dim() == 2:
            # Single mask HW, expand to BHW
            masks_np = masks.unsqueeze(0).cpu().numpy()
        elif masks.dim() == 3:
            # Batch masks BHW
            masks_np = masks.cpu().numpy()
        else:
            raise ValueError("Mask must be of shape HW or BHW")
        
        assert masks_np.shape[0] == images_np.shape[0] , "Masks  and images must had same batch size"
        
        batch_size = images_np.shape[0]
        
        for i in range(batch_size):
            # Get image and corresponding mask
            img = images_np[i]  # HWC
            mask = masks_np[i] 
            
            # Ensure image is in range [0, 255]
            if img.max() <= 1.0:
                img = img * 255
            img = img.astype(np.uint8)
            
            # Create RGBA image using mask as alpha channel
            rgba_img = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            rgba_img[:, :, :3] = img
            rgba_img[:, :, 3] = ((1 - mask) * 255).astype(np.uint8) # 
            
            # Convert to PIL Image
            pil_img = Image.fromarray(rgba_img, 'RGBA') 
            pil_img.save(f"{i}temp.png")          
            output_list.append(pil_img)
        img=output_list[0]
        original_width, original_height = img.size
        target_width, target_height = target_size

        # 1. 计算缩放比例，确保图片能完整放入目标框内
        ratio = min(target_width / original_width, target_height / original_height)
        
        # 2. 计算缩放后的新尺寸
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # 3. 高质量缩放图片
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 4. 创建一个新的纯色背景画布
        # 注意：颜色需要是RGBA格式，所以白色是 (255, 255, 255, 255)
        # 最后一个值255代表完全不透明
        final_background_color = background_color + (255,)
        background = Image.new('RGBA', target_size, final_background_color)
        
        # 5. 计算粘贴位置，使其居中
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # 6. 将缩放后的图片粘贴到背景画布上
        # 第三个参数 `resized_img` 作为蒙版，可以正确处理PNG的透明通道
        background.paste(resized_img, (paste_x, paste_y), resized_img)
    else:
        final_background_color = background_color + (255,)
        background = Image.new('RGBA', target_size, final_background_color)
    background.save("temp.png")
    return background.convert('RGB')

def get_wav2vec_repo(repo):
    required_files = ["chinese-wav2vec2-base-fairseq-ckpt.pt", "config.json", "model.safetensors", "preprocessor_config.json"]
    if not repo:
        wav2vec_repo=os.path.join(folder_paths.models_dir, "wav2vec2-base")
        if not os.path.exists(wav2vec_repo):
            os.makedirs(wav2vec_repo)
            download_file(folder_paths.models_dir, required_files)
        else:
            if not check_files_exist(wav2vec_repo,required_files):
                download_file(folder_paths.models_dir, required_files)  
    else:
        wav2vec_repo=PureWindowsPath(repo).as_posix()
    return wav2vec_repo

def download_file(local_dir, required_files):
    from huggingface_hub import hf_hub_download
    for i in required_files:
        hf_hub_download(
            repo_id="youliang1233214/InteractAvatar",
            subfolder="wav2vec2-base",
            filename=i,
            local_dir = local_dir,
        )   

def check_files_exist(folder_path, required_files):
    for file_name in required_files:
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            return False
    return True

def clear_comfyui_cache():
    cf_models=mm.loaded_models()
    try:
        for pipe in cf_models:
            pipe.unpatch_model(device_to=torch.device("cpu"))
            print(f"Unpatching models.{pipe}")
    except: pass
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")

# def trans2path(audio):
#     if audio is None:
#         return None
#     import io as io_base
#     audio_file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
#     audio_file = os.path.join(folder_paths.get_input_directory(), f"audio_{audio_file_prefix}_temp.wav")
#     buff = io_base.BytesIO()

#     torchaudio.save(buff, audio["waveform"].squeeze(0), audio["sample_rate"], format="FLAC")
#     with open(audio_file, 'wb') as f:
#         f.write(buff.getbuffer())
#     return audio_file
def trans2path(audio):
    """
    修正版：使用 soundfile 代替 torchaudio.save 以避开 torchcodec 的环境报错。
    """
    if audio is None:
        return None
    import io as io_base
    audio_file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
    audio_file = os.path.join(folder_paths.get_input_directory(), f"audio_{audio_file_prefix}_temp.wav")
    buff = io_base.BytesIO()

    # --- 修正逻辑开始 ---
    # ComfyUI 音频格式通常为 [Batch, Channels, Samples] -> [1, C, S]
    # 我们需要将其转换为 NumPy，并调整维度为 soundfile 要求的 [Samples, Channels]
    waveform = audio["waveform"].squeeze(0).cpu().numpy()  # 结果为 [C, S]
    sample_rate = audio["sample_rate"]

    if waveform.ndim == 2:
        waveform = waveform.T  # 转置为 [S, C]
    
    # 使用 soundfile 直接写入内存流，不触发 torchaudio 的后端检测
    sf.write(buff, waveform, sample_rate, format="FLAC")
    # --- 修正逻辑结束 ---

    with open(audio_file, 'wb') as f:
        f.write(buff.getbuffer())
    return audio_file





def encode_image( image, vae):
    if image is None: 
        return None
    ref_latents=None
    samples = image.movedim(-1, 1)
    total = int(1024 * 1024)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)

    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    image = s.movedim(1, -1)
    if vae is not None:
        ref_latents = vae.encode(image[:, :, :, :3])
    return ref_latents

def add_mean(latents):
    vae_config={"latents_mean": [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921
        ],
        "latents_std": [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.916
        ],}
    latents_mean = (torch.tensor(vae_config["latents_mean"]).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype))
    latents_std = 1.0 / torch.tensor(vae_config["latents_std"]).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    image_latent_height, image_latent_width = latents.shape[3:]
    image_latents = pack_latents_(
        latents, 1, 16, image_latent_height, image_latent_width)
    return image_latents

def pack_latents_(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents



def load_lora(model, lora_1, lora_2, lora_scale1, lora_scale2):
    lora_path_1=folder_paths.get_full_path("loras", lora_1) if lora_1 != "none" else None
    lora_path_2=folder_paths.get_full_path("loras", lora_2) if lora_2 != "none" else None
    # lora_list=[i for i in [lora_path_1,lora_path_2] if i is not None]
    # lora_scales=[lora_scale1,lora_scale2]
    all_adapters = model.get_list_adapters()
    dit_list=[]
    if all_adapters:
        dit_list= all_adapters.get('transformer',[])+all_adapters.get('transformer_2',[])
    if lora_path_1 is not None:
        adapter_name=os.path.splitext(os.path.basename(lora_path_1))[0].replace(".", "_")
        dit_list2=all_adapters.get('transformer_2',[])
        if dit_list2:
            if adapter_name in dit_list: #dit_list
                pass
            else: 
                for i in dit_list2:
                    model.delete_adapters(i)
                    print(f"去除dit中未加载的lora: {i}")   
                try:  
                    model.load_lora_weights(lora_path_1, adapter_name=adapter_name,**{"load_into_transformer_2": True})
                    model.set_adapters([adapter_name], adapter_weights=lora_scale1)
                except KeyError as e:
                    try:
                        print(f"检测到特殊的 LoRA 格式，尝试手动处理: {lora_path_1}")
                        state_dict = torch.load(lora_path_1, map_location="cpu",weights_only=False) if not lora_path_1.endswith(".safetensors") else load_file(lora_path_1,)
                        processed_state_dict = preprocess_lora_state_dict(state_dict)
                        model.load_lora_weights(processed_state_dict, adapter_name=adapter_name,**{"load_into_transformer_2": True})
                        model.set_adapters([adapter_name], adapter_weights=lora_scale1)
                    except:
                        print(f"加载LoRA权重失败: {e}")
                        pass
        else: 
            try:  
                model.load_lora_weights(lora_path_1, adapter_name=adapter_name,**{"load_into_transformer_2": True})
                model.set_adapters([adapter_name], adapter_weights=lora_scale1)
            except KeyError as e:
                try:
                    print(f"检测到特殊的 LoRA 格式，尝试手动处理: {lora_path_1}")
                    state_dict = torch.load(lora_path_1, map_location="cpu",weights_only=False) if not lora_path_1.endswith(".safetensors") else load_file(lora_path_1,)
                    processed_state_dict = preprocess_lora_state_dict(state_dict)
                    model.load_lora_weights(processed_state_dict, adapter_name=adapter_name,**{"load_into_transformer_2": True})
                    model.set_adapters([adapter_name], adapter_weights=lora_scale1)
                    del processed_state_dict
                except:
                    print(f"加载LoRA权重失败: {e}")
                    pass      
    if lora_path_2 is not None:
        adapter_name=os.path.splitext(os.path.basename(lora_path_2))[0].replace(".", "_")
        dit_list=all_adapters.get('transformer',[])
        if dit_list:
            if adapter_name in dit_list: #dit_list
                pass
            else: 
                for i in dit_list:
                    model.delete_adapters(i)
                    print(f"去除dit中未加载的lora: {i}")
                try:
                    model.load_lora_weights(lora_path_2, adapter_name=adapter_name,**{"load_into_transformer_2": False})
                    model.set_adapters([adapter_name], adapter_weights=lora_scale2)
                except KeyError as e:
                    try:
                        print(f"检测到特殊的 LoRA 格式，尝试手动处理: {lora_path_2}")
                        state_dict = torch.load(lora_path_2, map_location="cpu",weights_only=False) if not lora_path_2.endswith(".safetensors") else load_file(lora_path_2,)
                        processed_state_dict = preprocess_lora_state_dict(state_dict)
                        model.load_lora_weights(processed_state_dict, adapter_name=adapter_name,**{"load_into_transformer_2": False})
                        model.set_adapters([adapter_name], adapter_weights=lora_scale2)
                        del processed_state_dict
                    except:
                        print(f"加载LoRA权重失败: {e}")
                        pass
        else:
            try:
                model.load_lora_weights(lora_path_2, adapter_name=adapter_name,**{"load_into_transformer_2": False})
                model.set_adapters([adapter_name], adapter_weights=lora_scale2)
            except KeyError as e:
                try:
                    print(f"检测到特殊的 LoRA 格式，尝试手动处理: {lora_path_2}")
                    state_dict = torch.load(lora_path_2, map_location="cpu",weights_only=False) if not lora_path_2.endswith(".safetensors") else load_file(lora_path_2,)
                    processed_state_dict = preprocess_lora_state_dict(state_dict)
                    model.load_lora_weights(processed_state_dict, adapter_name=adapter_name,**{"load_into_transformer_2": False})
                    model.set_adapters([adapter_name], adapter_weights=lora_scale2)
                    del processed_state_dict
                except:
                    print(f"加载LoRA权重失败: {e}")
                    pass

    return model


def preprocess_lora_state_dict(state_dict):

    processed_dict = state_dict.copy()
    keys_to_remove = [
        'head.head.diff_b',
        'head.head.diff_m',
        'head.head.diff',
        'patch_embedding.diff',
        'patch_embedding.diff_b',
        'blocks.*.diff_m',  # 匹配所有blocks的diff_m
        'head.head.lora_down'
        'diffusion_model.head.head.diff'
        'diffusion_model.head.head.diff_b'
        'diffusion_model.head.lora_down'
        
    ]
    keys_to_delete = []
    for key in processed_dict.keys():
        if key.endswith('.diff_m'):
            keys_to_delete.append(key)
    for key in keys_to_delete:
        processed_dict.pop(key, None)
        print(f"移除键: {key}")    
    for key in keys_to_remove:
        if key in processed_dict:
            processed_dict.pop(key, None)
            print(f"移除键: {key}")
    return processed_dict

def gc_cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def tensor2cv(tensor_image):
    if len(tensor_image.shape)==4:# b hwc to hwc
        tensor_image=tensor_image.squeeze(0)
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu()
    tensor_image=tensor_image.numpy()
    #反归一化
    maxValue=tensor_image.max()
    tensor_image=tensor_image*255/maxValue
    img_cv2=np.uint8(tensor_image)#32 to uint8
    img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    return img_cv2

def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def tensor2image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pillist(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor2image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor2image(i) for i in tensor_list]
    return img_list

def tensor2pillist_upscale(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [nomarl_upscale(tensor_in,width,height)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[nomarl_upscale(i,width,height) for i in tensor_list]
    return img_list

def tensor2list(tensor_in,width,height):
    if tensor_in is None:
        return None
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        tensor_list = [tensor_upscale(tensor_in,width,height)]
    else:
        tensor_list_ = torch.chunk(tensor_in, chunks=d1)
        tensor_list=[tensor_upscale(i,width,height) for i in tensor_list_]
    return tensor_list


def tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "bilinear", "center")
    samples = samples.movedim(1, -1)
    return samples

def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "bilinear", "center")
    samples = img.movedim(1, -1)
    img = tensor2image(samples)
    return img



def cv2tensor(img,bgr2rgb=True):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).permute(1, 2, 0).unsqueeze(0)  



def images_generator(img_list: list, ):
    # get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_, Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_, np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise "unsupport image list,must be pil or cv2!!!"
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]
    
    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in = img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in, np.ndarray):
            i = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            print(i.shape)
            return i
        else:
            raise "unsupport image list,must be pil,cv2 or tensor!!!"
    
    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image


def load_images_list(img_list: list, ):
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images

def get_video_files(directory, extensions=None):
    if extensions is None:
        extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov']
    extensions = [ext.lower() for ext in extensions]
    video_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file)
            ext = ext.lower()[1:] 
            if ext in extensions:
                full_path = os.path.join(root, file)
                video_files.append(full_path)             
    return video_files
