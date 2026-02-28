 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .test_wanx_tia2mv_obj_back import generate_video,perdata,load_model,get_z_scale
from .model_loader_utils import trans2path,tensor2pillist,clear_comfyui_cache,get_wav2vec_repo,tensor_upscale,merge_mask_with_pose,tensor2image
import nodes
MAX_SEED = np.iinfo(np.int32).max

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")

node_cr_path = os.path.dirname(os.path.abspath(__file__))

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir

weigths_dwpose_current_path = os.path.join(folder_paths.models_dir, "dwpose")
if not os.path.exists(weigths_dwpose_current_path):
    os.makedirs(weigths_dwpose_current_path)
folder_paths.add_model_folder_path("dwpose", weigths_dwpose_current_path) #  dwpose dir

class InteractAvatar_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):     
        return io.Schema(
            node_id="InteractAvatar_SM_Model",
            display_name="InteractAvatar_SM_Model",
            category="InteractAvatar",
            inputs=[
                io.Combo.Input("dit",options= ["none"] +folder_paths.get_filename_list("diffusion_models") ), 
                io.Combo.Input("gguf",options= ["none"] +folder_paths.get_filename_list("gguf") ),  
                io.Combo.Input("lora",options= ["none"] + folder_paths.get_filename_list("loras") ),
                io.Combo.Input("back_append_frame",options= [1,2]),
                io.Boolean.Input("offload",default=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, dit,gguf,lora,back_append_frame,offload) -> io.NodeOutput:
        dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        lora_path=folder_paths.get_full_path("loras", lora) if lora != "none" else None
        assert dit_path != None , "Please select a model"
        clear_comfyui_cache()
        model = load_model(
                checkpoint_path=dit_path if dit_path != None else gguf_path if gguf_path != None else None,
                dit_path=dit_path if dit_path != None else gguf_path if gguf_path != None else None,
                lora_path=lora_path,
                back_append_frame=int(back_append_frame),
                offload=offload
            )
        return io.NodeOutput(model)
    
class InteractAvatar_SM_Predata(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="InteractAvatar_SM_Predata",
            display_name="InteractAvatar_SM_Predata",
            category="InteractAvatar",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Image.Input("images"), # image or video
                io.Image.Input("pose_images"), # image or video
                io.Combo.Input("short_side",options= [704,512]),
                io.String.Input("prompt",multiline=True, default="两只手打招呼 伸出大拇指点赞"),
                io.String.Input("negative_prompt",multiline=True, default="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"),
                io.String.Input("structured_prompt",multiline=True, default=" (Raise both hands slowly) (Wave both hands side to side for greeting),\n (Make a fist) (Extend the thumb upwards)"),
                io.Int.Input("num_frames", default=81, min=8, max=10000,step=1,display_mode=io.NumberDisplay.number),
                io.Combo.Input("mode",options=['a2mv','ap2v','mv','a2v','p2v'], default="a2mv"),
                io.Combo.Input("back_append_frame",options= [1,2]),
                io.Boolean.Input("all_text",default=True),
                io.String.Input("wav2vec_repo",multiline=False, default=""),
                io.String.Input("object_name",multiline=False, default=""),
                io.Audio.Input("audio",optional=True),
                io.Image.Input("object_images",optional=True), 
                ],
            outputs=[
                io.Conditioning.Output(display_name="data_dict"),
            ],
        ) 
    @classmethod
    def execute(cls, clip,vae,images,pose_images,short_side,prompt,negative_prompt,structured_prompt,num_frames,mode,back_append_frame,all_text,wav2vec_repo,object_name,audio=None,object_images=None,) -> io.NodeOutput: 
        
        if back_append_frame==2:
            prompt=[ii for ii in prompt.splitlines() if ii]
            structured_prompt=[ii for ii in structured_prompt.splitlines() if ii]
            assert len(prompt)==len(structured_prompt), "Please make sure the number of prompts and structured prompts are the same,人物提示词和动作提示词的行数要一致"

        img=tensor2pillist(images)
        dw_img=tensor2pillist(pose_images)
        audio_path=trans2path(audio)
        object_images=tensor2image(object_images)
        data_dict = perdata(
            clip,
            vae,
            img,
            dw_img,
            object_images,
            audio_path,
            mode,
            prompt=prompt,
            negative_prompt=negative_prompt,
            structured_prompt=structured_prompt,
            frame_num=num_frames,
            short_side=short_side,
            back_append_frame=back_append_frame,
            wav2vec_dir=get_wav2vec_repo(wav2vec_repo),
            device=device,
            all_text=all_text,
        )
        
        return io.NodeOutput(data_dict)

class InteractAvatar_SM_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="InteractAvatar_SM_Sampler",
            display_name="InteractAvatar_SM_Sampler",
            category="InteractAvatar",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("data_dict"),
                io.Int.Input("steps", default=20, min=1, max=10000,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Float.Input("sample_shift", default=5.0, min=0.0, max=10.0,step=0.1,display_mode=io.NumberDisplay.number),
                io.Float.Input("text_guide_scale", default=5.0, min=0.1, max=10.0,step=0.1,display_mode=io.NumberDisplay.number),
                io.Float.Input("audio_guide_scale", default=7.5, min=0.1, max=10.0,step=0.1,display_mode=io.NumberDisplay.number),
                io.Int.Input("bad_thres", default=800, min=0, max=100000,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("motion_frame",default=25, min=8, max=10000,step=1,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("bad_cfg",default=True),
                io.Boolean.Input("three_cfg",default=False),
                ],
            outputs=[
                io.Latent.Output(display_name="Latent"),
            ],
        ) 
    @classmethod
    def execute(cls, model,data_dict,steps,seed,sample_shift,text_guide_scale,audio_guide_scale,bad_thres,motion_frame,bad_cfg,three_cfg) -> io.NodeOutput: 
        clear_comfyui_cache()
        param_dict = dict(
            shift=sample_shift,
            sampling_steps=steps,
            text_guide_scale=text_guide_scale,
            audio_guide_scale=audio_guide_scale,
            seed=seed,
            bad_cfg=bad_cfg,
            three_cfg=three_cfg,
            bad_thres=bad_thres,
            offload_model=True,
            motion_frame=motion_frame,
        )
        data_dict.update(param_dict)

        video = generate_video(model,data_dict)
        clear_comfyui_cache() # clear cache
        if data_dict["back_append_frame"]==2: # don't need  decoder image  # B H W C
            # import time
            # pre_fix=time.strftime("%Y%m%d-%H%M%S")
            # save_files=os.path.join(folder_paths.get_output_directory(), f"output_{pre_fix}.pt")
            # torch.save(video, save_files) #本地解码
            video=model.vae.encode(video) # 统一输出接口
        else:
            video=get_z_scale(video) #torch.Size([1, 16, 21, 90, 136])
        output={"samples":video} 
        return io.NodeOutput(output)

class InteractAvatar_SM_Pose(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="InteractAvatar_SM_Pose",
            display_name="InteractAvatar_SM_Pose",
            category="InteractAvatar",
            inputs=[
                io.Image.Input("images"), # image or video
                io.Combo.Input("dw",options= ["none"] +folder_paths.get_filename_list("dwpose") ), 
                io.Combo.Input("yolo",options= ["none"] +folder_paths.get_filename_list("dwpose") ),
                io.Boolean.Input("short_side",default=True),
                io.Mask.Input("mask",optional=True),
                ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Image.Output(display_name="obj_images"),
            ],
        ) 
    @classmethod
    def execute(cls, images,dw,yolo,short_side,mask=None) -> io.NodeOutput: 
        dw_path=folder_paths.get_full_path("dwpose", dw) if dw != "none" else None
        yolo_path=folder_paths.get_full_path("dwpose", yolo) if yolo != "none" else None
        from .model_loader_utils import get_pose_normal
        origin_images=images
        images=get_pose_normal(images,os.path.join(folder_paths.base_path,"dwpose"),device,dw_path,yolo_path)
        obj_images=torch.ones((1, 256, 256,3))
        if mask is not None:
            images,obj_images = merge_mask_with_pose(mask, images,origin_images)
         
        if short_side: # 短边为512
            scale_factor=256/min(images.shape[1],images.shape[2])
            if images.shape[1]>images.shape[2]: #H>W   
                h_new=int((images.shape[1]*scale_factor)//32*32)
                images=tensor_upscale(images,256,h_new)
            else:
                w_new=int((images.shape[2]*scale_factor)//32*32)
                images=tensor_upscale(images,w_new,256)
        return io.NodeOutput(images,obj_images)


class InteractAvatar_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            InteractAvatar_SM_Model,
            InteractAvatar_SM_Predata,
            InteractAvatar_SM_Sampler,
            InteractAvatar_SM_Pose,
        ]

async def comfy_entrypoint() -> InteractAvatar_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return InteractAvatar_SM_Extension()
