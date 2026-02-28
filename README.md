<div align="center">
  <img src="assets/logo.svg" height=100 width="100%"> 

# Making Avatars Interact <br> Towards Text-Driven Human-Object Interaction for Controllable Talking Avatars
<!-- # Towards Text-Driven Human-Object Interaction for Controllable Talking Avatars -->

</div>

**InteractAvatar** is a novel dual-stream DiT framework that enables talking avatars to perform **Grounded Human-Object Interaction (GHOI)**. Unlike previous methods restricted to simple gestures, our model can perceive the environment from a static reference image and generate complex, text-guided interactions with objects while maintaining high-fidelity lip synchronization.

<div align="center">
  <a href="https://github.com/angzong/InteractAvatar"><img src="https://img.shields.io/static/v1?label=InteractAvatar%20Code&message=Github&color=blue"></a> &ensp;
  <a href="https://interactavatar.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://arxiv.org/abs/2602.01538"><img src="https://img.shields.io/badge/ArXiv-2602.01538-red"></a> &ensp;
  <a href="https://huggingface.co/youliang1233214/InteractAvatar"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow"></a>
</div>

<br>

# ComfyUI_InteractAvatar
InteractAvatar is a novel dual-stream DiT framework that enables talking avatars to perform Grounded Human-Object Interaction (GHOI)

# Update
* Add Dwpose node to easy use
* 新增dwpose节点，模型为none时会自动下载，简化object物体输入流程，可以之间用mask获取

# Previous
* fix bug ,now  output video short side muse be 512 or 704
* If your Vram <24G,turn on 'offload', ActionAndSong mode use 'long model' and need chocie '2' mode;example img\video\ audio in "InterDemo" dir
* test env 64G RAM, 12G VRAM,win11
* The prompt words for the singing mode and the action prompt words must have the same number of lines；
* 小显存开启offload，唱歌用带long的dit，模式选'2'，否则用常规的,示例图片音频等在InterDemo文件内; 基本上40G加8G能跑普通模式，长时长唱歌可能有难度，唱歌模式的提示词和动作提示词必须要有相同的行数


# 1. Installation

In the ./ComfyUI /custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_InteractAvatar.git
```
---
# 2. Requirements  
```
pip install -r requirements.txt
```
# 3. Models
* wan 2.2 vae/clip [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files)
* InteractAvatar dit  [youliang1233214/InteractAvatar](https://huggingface.co/youliang1233214/InteractAvatar/tree/main)  

```
--  ComfyUI/models/vae
    |-- wan2.2_vae.safetensors # or Wan2.2_VAE.pth origin
--  ComfyUI/models/clip
    |-- umt5_xxl_fp8_e4m3fn_scaled.safetensors  # or fp16
--  ComfyUI/models/diffusion_models
     |--interact-avatar-long.safetensors  #  rename from diffusion_pytorch_model.safetensors  long or normal
```

# 4 Example
* song long
![](https://github.com/smthemex/ComfyUI_InteractAvatar/blob/main/example_workflows/example-song.png)
* object 
![](https://github.com/smthemex/ComfyUI_InteractAvatar/blob/main/example_workflows/example.png)
* ap2v audio and pose driver
![](https://github.com/smthemex/ComfyUI_InteractAvatar/blob/main/example_workflows/example_ap2v.png)

# 5 Citation
```
@article{zhang2026making,
  title={Making Avatars Interact: Towards Text-Driven Human-Object Interaction for Controllable Talking Avatars},
  author={Zhang, Youliang and Zhou, Zhengguang and Yu, Zhentao and Huang, Ziyao and Hu, Teng and Liang, Sen and Zhang, Guozhen and Peng, Ziqiao and Li, Shunkai and Chen, Yi and Zhou, Zixiang and Zhou, Yuan and Lu, Qinglin and Li, Xiu},
  journal={arXiv preprint arXiv:2602.01538},
  year={2026}
}
```

## 🙏 Acknowledgements

We sincerely thank the contributors to the following projects:
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
- [Diffusers](https://github.com/huggingface/diffusers)
- [HuggingFace](https://huggingface.co)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)

