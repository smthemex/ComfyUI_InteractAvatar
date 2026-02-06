#!/bin/bash
# ======== 基本配置 (保持不变) ========
GPUS_NUM=1
inference_steps=40
model_path="./model_pretrain/Wan2.2-TI2V-5B"
exp_name="interact-demo"
caption="struct"
GPU_ID_LIST=(0 1 2)
GPU_ID=0
TOTAL_GPUS=8
wav2vec_path="./ckpt/wav2vec2-base/"

# for a2mv
mode="a2mv"
transformer_path="./ckpt/interact-avatar/"
test_data_path="./InterDemo/TIA2MV/demo_tia2mv.json"
back_append_frame=1
frame_num=101

# for pose dirven with audio
# mode="ap2v"
# transformer_path="./ckpt/interact-avatar/"
# test_data_path="./InterDemo/TIAP2V/demo_ap2v.json"
# back_append_frame=1
# frame_num=133

# for song with action
# mode="a2mv"
# transformer_path="./ckpt/interact-avatar-long/"
# test_data_path="./InterDemo/ActionAndSong/demo_song_action.json"
# back_append_frame=2
# frame_num=1000

base_save_path="./output/"$mode"_"$back_append_frame
for GPU_ID in "${GPU_ID_LIST[@]}"; do
    echo "======> Checkpoint on GPU $GPU_ID with mode $mode cfg <======"
    save_path=${base_save_path}
    CUDA_VISIBLE_DEVICES=$GPU_ID python test_wanx_tia2mv_obj_back.py \
         --task "ti2v-5B" \
         --ckpt_dir $model_path \
         --ulysses_size $GPUS_NUM \
         --frame_num $frame_num \
         --sample_shift 5.0 \
         --text_guide_scale 5.0 \
         --audio_guide_scale 7.5 \
         --transformer_dir $transformer_path \
         --sample_steps $inference_steps \
         --save_path $save_path \
         --test_data_path $test_data_path\
         --base_seed 2025 \
         --mode $mode \
         --start $(($GPU_ID)) \
         --end $(($GPU_ID + 1)) \
         --caption $caption \
         --short_side 704 \
         --bad_cfg \
         --bad_thres 800 \
         --back_append_frame $back_append_frame \
         --wav2vec_dir $wav2vec_path &
done
echo "all task started..."
wait
echo "all task done"