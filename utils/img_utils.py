import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, random
import torch.distributed as dist
from PIL import Image
import pandas as pd
import librosa
import json
import cv2
def get_image_from_path(image_path):
    """
    读取图片文件并返回一个 PIL Image 对象。

    :param image_path: 图片文件的路径 (例如 PNG, JPG 等)
    :return: PIL Image 对象，如果读取失败则返回 None
    """
    try:
        # 使用 Pillow 的 Image.open() 直接打开图片文件
        img = Image.open(image_path)

        # 为了确保输出格式与原函数一致 (RGB)，我们进行转换。
        # 这么做可以处理带透明度通道的 PNG 图片 (RGBA -> RGB)
        # 或者其他色彩模式的图片。
        return img.convert('RGB')

    except FileNotFoundError:
        print(f"错误: 无法找到文件 {image_path}")
        return None
    except Exception as e:
        print(f"错误: 打开图片文件时发生未知错误 {image_path}: {e}")
        return None

def _center_crop_to_aspect_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    """
    辅助函数：将图片通过中心裁剪的方式调整到目标长宽比。
    target_ratio: 宽度 / 高度
    """
    w, h = img.size
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 1e-4: # 如果比例已经很接近，则无需裁剪
        return img

    if current_ratio > target_ratio: # 当前图片比目标更“宽”，需要裁掉左右两边
        new_w = int(target_ratio * h)
        left = (w - new_w) // 2
        right = left + new_w
        top, bottom = 0, h
    else: # 当前图片比目标更“高”，需要裁掉上下两边
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        bottom = top + new_h
        left, right = 0, w
        
    img_cropped = img.crop((left, top, right, bottom))
    print(f"INFO: 图片从 {img.size} 裁剪到 {img_cropped.size} 以匹配目标长宽比 {target_ratio:.3f}")
    return img_cropped

def process_images_final(img1, img2, reference_aspect_from = 'small'):
    """
    处理两张图片，严格满足以下所有条件：
    1. 所有最终图片的宽高都是32的倍数。
    2. 一张图的短边固定为256px，另一张固定为704px。
    3. 两张图最终的长宽比【完全相等】。
    4. 通过中心裁剪来统一长宽比。
    
    :param img1: 第一个Pillow图片对象。
    :param img2: 第二个Pillow图片对象。
    :param reference_aspect_from: 以哪张图的长宽比为基准, 'small' 或 'large'。
    :return: 一个包含两张处理后图片的元组 (img1_processed, img2_processed)。
    """
    # 1. 识别小图和大图
    w1, h1 = img1.size
    w2, h2 = img2.size
    short1, short2 = min(w1, h1), min(w2, h2)

    if short1 < short2:
        img_small_orig, img_large_orig, is_img1_small = img1, img2, True
    else:
        img_small_orig, img_large_orig, is_img1_small = img2, img1, False
        
    # 2. 确定基准长宽比并裁剪非基准图
    if reference_aspect_from == 'small':
        ref_img, other_img = img_small_orig, img_large_orig
    else:
        ref_img, other_img = img_large_orig, img_small_orig

    ref_w, ref_h = ref_img.size
    target_aspect_ratio = ref_w / ref_h
    other_img_cropped = _center_crop_to_aspect_ratio(other_img, target_aspect_ratio)

    if reference_aspect_from == 'small':
        img_small_src, img_large_src = ref_img, other_img_cropped
    else:
        img_small_src, img_large_src = other_img_cropped, ref_img
        
    # 3. 核心计算：基于严格比例推导尺寸
    is_horizontal = target_aspect_ratio > 1
    
    # 3.1 计算小图的理想长边
    if is_horizontal:
        ideal_long_small = 256 * target_aspect_ratio
    else:
        ideal_long_small = 256 / target_aspect_ratio
        
    # 3.2 【关键修正】将小图理想长边调整到最接近的128的倍数
    # 这是为了确保大图的长边也能成为32的倍数 (128 = 32 * 4)
    final_long_small = int(round(ideal_long_small / 128.0) * 128)
    final_long_small = max(128, final_long_small) # 确保不为0
    
    # 3.3 计算大图的最终长边，它现在必然是32的倍数
    final_long_large = int(final_long_small * (704 / 256.0))

    # 4. 组合最终尺寸
    if is_horizontal:
        final_size_small = (final_long_small, 256)
        final_size_large = (final_long_large, 704)
    else:
        final_size_small = (256, final_long_small)
        final_size_large = (704, final_long_large)

    print(f"INFO: 小图理想长边 {ideal_long_small:.2f} -> 调整到128的倍数 -> {final_long_small}")
    print(f"FINAL: 小图尺寸: {final_size_small}, 大图尺寸: {final_size_large}")
    
    # 验证长宽比是否相等
    final_ratio_small = final_size_small[0] / final_size_small[1]
    final_ratio_large = final_size_large[0] / final_size_large[1]

    # 5. 缩放图片到最终尺寸
    img_small_final = img_small_src.resize(final_size_small, Image.Resampling.LANCZOS)
    img_large_final = img_large_src.resize(final_size_large, Image.Resampling.LANCZOS)

    # 6. 按原始输入顺序返回结果
    if is_img1_small:
        return img_small_final, img_large_final
    else:
        return img_large_final, img_small_final
def resize_images(img1: Image.Image, img2: Image.Image):
    """
    根据特定规则缩放两个Pillow图片对象。（已修正版本）
    """
    w1, h1 = img1.size
    w2, h2 = img2.size
    short1, long1 = (w1, h1) if w1 < h1 else (h1, w1)
    short2, long2 = (w2, h2) if w2 < h2 else (h2, w2)

    # 情况一：两个短边都是256 (此部分逻辑正确，无需修改)
    if short1 == 256 and short2 == 256:
        print("检测到情况一：两张图片的短边均为256。")
        if long1 < long2:
            target_size = img1.size
            img2_resized = img2.resize(target_size, Image.Resampling.LANCZOS)
            return img1, img2_resized
        elif long2 < long1:
            target_size = img2.size
            img1_resized = img1.resize(target_size, Image.Resampling.LANCZOS)
            return img1_resized, img2
        else:
            return img1, img2

    # 情况二：一个短边是256，另一个是512 (此部分已修正)
    elif (short1 == 256 and short2 == 512) or (short1 == 512 and short2 == 256):
        print("检测到情况二：一张图片短边为256，另一张为512。")
        if short1 == 256:
            img_small, img_large = img1, img2
            long_small, long_large_orig = long1, long2
        else:
            img_small, img_large = img2, img1
            long_small, long_large_orig = long2, long1
            
        # 计算大图的目标长边
        target_long_large = long_small * 2
        
        # 如果大图当前的长边已经是目标长度，则无需处理
        if target_long_large == long_large_orig:
            return img1, img2

        w_large, h_large = img_large.size

        # --- 核心修正逻辑 ---
        # 直接将新的长边与固定的短边(512)组合，而不是通过比例计算
        if w_large > h_large:  # 如果宽度是长边, 高度就是固定的512
            target_size_large = (target_long_large, h_large)
        else:  # 如果高度是长边, 宽度就是固定的512
            target_size_large = (w_large, target_long_large)
        # --- 修正结束 ---
        
        print(f"小图尺寸: {img_small.size}, 大图将从 {img_large.size} 缩放到 {target_size_large}")
        img_large_resized = img_large.resize(target_size_large, Image.Resampling.LANCZOS)

        if short1 == 256:
            return img_small, img_large_resized
        else:
            return img_large_resized, img_small

    else:
        raise ValueError("输入的图片尺寸不符合预设的两种情况。")
def resize_short_side(img, short_side=512):
    w, h = img.size
    if h < w:
        new_h = short_side
        new_w = int(w * short_side / h)
    else:
        new_w = short_side
        new_h = int(h * short_side / w)
    new_h = int(new_h // 32 * 32)
    new_w = int(new_w // 32 * 32)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return img
def get_first_frame_from_video(video_path):
    """
    读取视频文件的第一帧并返回一个 PIL Image 对象。

    :param video_path: 视频文件的路径
    :return: PIL Image 对象，如果读取失败则返回 None
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return None

    # 读取第一帧
    # cap.read() 返回一个元组 (布尔值, 帧)
    # 布尔值表示是否成功读取，帧是图像数据
    ret, frame = cap.read()

    # 释放视频捕获对象，这很重要
    cap.release()

    if ret:
        # OpenCV 读取的图像格式是 BGR (蓝, 绿, 红)
        # PIL 和大多数其他库使用的格式是 RGB (红, 绿, 蓝)
        # 所以我们需要进行颜色空间转换
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 将 NumPy 数组格式的帧转换为 PIL Image 对象
        return Image.fromarray(frame_rgb)
    else:
        print("错误: 无法从视频中读取第一帧")
        return None

def get_all_frames_from_video(video_path: str):
    """
    读取视频文件的所有帧并返回一个 PIL Image 对象的列表。

    :param video_path: 视频文件的路径
    :return: PIL Image 对象的列表，如果读取失败或视频为空则返回 None
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return None

    frames = []
    while True:
        # 读取一帧
        # cap.read() 返回一个元组 (布尔值, 帧)
        # 布尔值表示是否成功读取，帧是图像数据
        ret, frame = cap.read()

        # 如果 ret 为 False，表示视频已结束或读取时发生错误
        if not ret:
            break

        # OpenCV 读取的图像格式是 BGR (蓝, 绿, 红)
        # PIL 和大多数其他库使用的格式是 RGB (红, 绿, 蓝)
        # 所以我们需要进行颜色空间转换
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 将 NumPy 数组格式的帧转换为 PIL Image 对象并添加到列表中
        frames.append(Image.fromarray(frame_rgb))

    # 释放视频捕获对象，这很重要
    cap.release()

    if not frames:
        print(f"错误: 未能从视频 {video_path} 中读取任何帧。")
        return None

    return frames
import torchvision.transforms.functional as TF

def read_obj_tensor_from_path(input_path=None, target_size=None, background_color=(255, 255, 255)):
    if input_path is None:
        final_background_color = background_color + (255,)
        background = Image.new('RGBA', target_size, final_background_color)
    elif not os.path.exists(input_path):
        final_background_color = background_color + (255,)
        background = Image.new('RGBA', target_size, final_background_color)
    else:
        with Image.open(input_path) as img:
            # 确保图片是RGBA模式，以更好地处理透明度
            img = img.convert("RGBA")
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
    return background.convert('RGB')

