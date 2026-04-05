#!/usr/bin/env bash

# const
CKPT_PATH=/path/to/Uni-ViGU-CKPT
wan_model_path=/path/to/wan_2.1_1.3B_ckpt

python inference_unified_wan.py \
    --mode t2v \
    --checkpoint "$CKPT_PATH" \
    --wan_model_path "$wan_model_path" \
    --prompt "A golden retriever runs across a sunlit grassy field filled with tiny wildflowers, ears bouncing naturally, dust glowing in the warm afternoon light, camera follows at low angle in slow motion, then cuts to a close-up of the dog happily looking toward the viewer, vivid natural colors, joyful mood, realistic fur motion, cinematic family film style." \
    --output_dir ./outputs \
        --max_length 256 \
        --num_inference_steps 100 \
        --seed 42 \
        --cfg_scale 5 \
        --save_video --video_fps 16 --video_format mp4
