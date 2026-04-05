#!/usr/bin/env bash

# const
CKPT_PATH=/inspire/ssd/project/sais-mtm/public/qlz/code/flow_matching/examples/text/results/unified-flow-vbench-64-128-TO-128-256-sampled-2k-balanced-alpha-t-cfg-0.2-lr-5e-5/checkpoints/step-00068000.pth
wan_model_path=/inspire/ssd/project/sais-mtm/public/qlz/pretrained_models/Wan2.1-T2V-1.3B-Diffusers

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
