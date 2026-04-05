# Uni-ViGU: Towards Unified Video Generation and Understanding via A Diffusion-Based Video Generator [Preview]

Unified multimodal models integrating visual understanding and generation face a fundamental challenge: visual generation incurs substantially higher computational costs than understanding, particularly for video. This imbalance motivates us to invert the conventional paradigm: rather than extending understanding-centric MLLMs to support generation, we propose **Uni-ViGU**, a framework that unifies video generation and understanding by extending a video generator as the foundation.

We introduce a **unified flow method** that performs continuous flow matching for video and discrete flow matching for text within a single process, enabling coherent multimodal generation. We further propose a **modality-driven MoE-based framework** that augments Transformer blocks with lightweight layers for text generation while preserving generative priors. To repurpose generation knowledge for understanding, we design a **bidirectional training mechanism** with two stages: *Knowledge Recall* reconstructs input prompts to leverage learned text-video correspondences, while *Capability Refinement* fine-tunes on detailed captions to establish discriminative shared representations.

Experiments demonstrate that Uni-ViGU achieves competitive performance on both video generation and understanding, validating generation-centric architectures as a scalable path toward unified multimodal intelligence.

## 📦 Install

### 1. Clone the Repository

```bash
git clone https://github.com/Fr0zenCrane/Uni-ViGU.git
cd ./Uni-ViGU
```

### 2. Create Conda Environment

```bash
conda create -n uni-vigu python=3.11
conda activate uni-vigu
```

### 3. Install Flow Matching

```bash
git clone https://github.com/facebookresearch/flow_matching.git
cd flow_matching
python setup.py install
cd ..
```

### 4. Install Dependencies

```bash
pip install torch==2.4.0 hydra-core hydra-submitit-launcher datasets wandb einops accelerate
pip install transformers==4.57.6 sentencepiece
pip install imageio
```

## 🚀 Inference

Run the inference script `run.sh`:

```bash
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
```

> **Note:** Replace `/path/to/Uni-ViGU-CKPT` and `/path/to/wan_2.1_1.3B_ckpt` with the actual paths to your model checkpoints before running.

## 🙏 Thanks

We appreciate the [Diffusers](https://github.com/huggingface/diffusers) and the [Flow Matching](https://github.com/facebookresearch/flow_matching) — the two codebases this project is built upon.
