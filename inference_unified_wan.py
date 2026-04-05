"""
Inference script for WanUnifiedTransformer (T2V, T2VT, and V2T).

This script loads a checkpoint produced by train_unified_wan.py and supports:
  1. T2V (Text-to-Video): Generate video frames from text prompt
  2. T2VT (Text-to-Video-Text): Generate video AND text jointly in one pass
  3. V2T (Video-to-Text): Generate text caption from video frames

==============================================================================
USAGE EXAMPLES (run on Linux server with GPU):
==============================================================================

# 1. T2V: Generate video from text prompt (outputs pixel-space video frames)
python inference_unified_wan.py \
    --mode t2v \
    --checkpoint /path/to/checkpoint/step-00010000.pth \
    --wan_model_path /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cat playing with a ball in the garden" \
    --output_dir ./outputs \
    --num_inference_steps 50 \
    --seed 42

# 1b. T2V with video export (saves as MP4 file)
python inference_unified_wan.py \
    --mode t2v \
    --checkpoint /path/to/checkpoint/step-00010000.pth \
    --wan_model_path /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cat playing with a ball in the garden" \
    --output_dir ./outputs \
    --save_video --video_fps 24 --video_format mp4

# 1c. T2V with Classifier-Free Guidance (CFG) for better prompt adherence
python inference_unified_wan.py \
    --mode t2v \
    --checkpoint /path/to/checkpoint/step-00010000.pth \
    --wan_model_path /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cat playing with a ball in the garden" \
    --output_dir ./outputs \
    --cfg_scale 7.5 \
    --null_cond_path /path/to/null_cond_t5_emb.pt

# 1d. T2V with CFG (auto-generate null condition from empty string)
python inference_unified_wan.py \
    --mode t2v \
    --checkpoint /path/to/checkpoint/step-00010000.pth \
    --wan_model_path /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cat playing with a ball in the garden" \
    --output_dir ./outputs \
    --cfg_scale 7.5

# 2. T2VT: Generate video AND text jointly (single pass, not two-stage)
python inference_unified_wan.py \
    --mode t2vt \
    --checkpoint /path/to/checkpoint/step-00010000.pth \
    --wan_model_path /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cat playing with a ball in the garden" \
    --output_dir ./outputs \
    --num_inference_steps 50 \
    --seed 42

# 2b. T2VT with video export
python inference_unified_wan.py \
    --mode t2vt \
    --checkpoint /path/to/checkpoint/step-00010000.pth \
    --wan_model_path /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cat playing with a ball in the garden" \
    --output_dir ./outputs \
    --save_video --video_fps 16

# 3. V2T: Generate text from video frames (input is pixel-space video tensor)
python inference_unified_wan.py \
    --mode v2t \
    --checkpoint /path/to/checkpoint/step-00010000.pth \
    --wan_model_path /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --video_path /path/to/video_frames.pt \
    --output_dir ./outputs \
    --num_inference_steps 50 \
    --seed 42

# 3b. V2T: Generate text from video file directly (.mp4, .avi, etc.)
python inference_unified_wan.py \
    --mode v2t \
    --checkpoint /path/to/checkpoint/step-00010000.pth \
    --wan_model_path /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --video_path /path/to/video.mp4 \
    --output_dir ./outputs \
    --input_fps 16 --max_input_frames 81 --crop_mode center

# 4. Dry run (validate checkpoint loads and forward pass works)
python inference_unified_wan.py \
    --mode t2v \
    --checkpoint /path/to/checkpoint/step-00010000.pth \
    --wan_model_path /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --dry_run

==============================================================================
NOTES:
==============================================================================
- T2V outputs video frames in pixel space as a .pt tensor [B, C, T, H, W]
  with values in range [-1, 1]. Also saves latent for debugging.
- T2VT generates both video and text in a single joint inference pass.
  The text is generated alongside video (not as a two-stage pipeline).
  Outputs: video frames + generated text description.
- V2T accepts either:
  1. A .pt tensor file [B, C, T, H, W] with values in [-1, 1] or [0, 1]
  2. A video file (.mp4, .avi, .mov, etc.) - will be auto-converted to 480x832
- Video files are center-cropped to 480x832 (no stretching). Use --crop_mode to
  change crop position. Use --input_fps and --max_input_frames to control sampling.
- The VAE is used internally to encode/decode between pixel and latent space.
- The --wan_model_path should point to a diffusers-format Wan model directory
  containing subfolders: transformer/, vae/, text_encoder/, tokenizer/
- Use --save_video to export frames as MP4 or GIF. Requires imageio/imageio-ffmpeg.
- Classifier-Free Guidance (CFG): Use --cfg_scale > 1.0 to enable CFG for T2V/T2VT.
  Typical values are 5.0-15.0. Higher values = stronger prompt adherence.
  Optionally provide --null_cond_path with pre-computed null T5 embedding,
  otherwise the empty string will be encoded as the null condition.

==============================================================================
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Add parent directory to path for local imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Local imports (from the training codebase)
try:
    from model import WanUnifiedTransformer, FlowMatchScheduler
    from logic.flow import (
        get_source_distribution,
        get_path,
        SourceDistribution,
        UniformSourceDistribution,
        MaskedSourceDistribution,
    )
    from utils.video_export import tensor_to_video, tensor_to_gif
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    logger.error("Make sure you're running this script from the examples/text directory")
    sys.exit(1)

# External imports
try:
    from transformers import T5Tokenizer, T5EncoderModel
    from diffusers import AutoencoderKLWan
    from flow_matching.path import MixtureDiscreteProbPath, ProbPath
    from flow_matching.path.scheduler import PolynomialConvexScheduler
    from flow_matching.solver import MixtureDiscreteEulerSolver
    from flow_matching.utils import ModelWrapper
except ImportError as e:
    logger.error(f"Missing required packages: {e}")
    logger.error("Install with: pip install transformers diffusers flow_matching")
    sys.exit(1)


# ==============================================================================
# Model Wrapper for Text Flow Matching
# ==============================================================================

class TextModelWrapper(ModelWrapper):
    """Wrapper for text generation with the unified model."""

    def __init__(
        self,
        model: torch.nn.Module,
        video_latent: torch.Tensor,
        cond_t: torch.Tensor,
        num_train_timesteps: int = 1000,
        device: torch.device = None,
    ):
        super().__init__(model)
        self.video_latent = video_latent
        self.cond_t = cond_t
        self.num_train_timesteps = num_train_timesteps
        self.device = device or video_latent.device

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        """
        Forward pass for text generation.

        Args:
            x: Current text tokens [batch, seq_len]
            t: Current timestep (0 to 1)

        Returns:
            Probability distribution over vocabulary [batch, seq_len, vocab_size]
        """
        batch_size = x.shape[0]

        # Convert flow matching time (0 to 1) to Wan-style timestep
        # In training: time_t = round((1.0 - t) * num_train_timesteps)
        time_t = torch.round((1.0 - t) * self.num_train_timesteps)
        time_t = torch.clamp(time_t, min=0, max=self.num_train_timesteps)
        time_t = time_t.expand(batch_size).to(self.device, dtype=self.video_latent.dtype)

        # Use fixed video (timestep 0 = fully denoised)
        time_v = torch.zeros(batch_size, device=self.device, dtype=self.video_latent.dtype)

        # Forward through model
        _, logits = self.model(
            x_v=self.video_latent.expand(batch_size, -1, -1, -1, -1),
            x_t=x,
            time_v=time_v,
            time_t=time_t,
            cond_t=self.cond_t.expand(batch_size, -1, -1),
        )

        # Return softmax probabilities
        return torch.softmax(logits.float(), dim=-1)


# ==============================================================================
# Utility Functions
# ==============================================================================

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU available, using CPU (inference will be slow)")
    return device


def save_video_frames_as_images(
    video_frames: torch.Tensor,
    output_dir: str,
    image_format: str = "png",
    prefix: str = "frame",
) -> None:
    """
    Save video frames as individual image files.

    Args:
        video_frames: Video tensor [B, C, T, H, W] in range [-1, 1]
        output_dir: Directory to save images
        image_format: Image format ("png" or "jpg")
        prefix: Filename prefix for saved images
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("PIL (Pillow) is required for saving images. Install with: pip install Pillow")
        return

    # Create output directory
    frames_dir = Path(output_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Get frames from tensor [B, C, T, H, W] -> process first batch
    frames = video_frames[0]  # [C, T, H, W]

    # Rearrange to [T, H, W, C] for saving
    frames = frames.permute(1, 2, 3, 0)  # [T, H, W, C]

    # Convert from [-1, 1] to [0, 255]
    frames = ((frames + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    frames = frames.cpu().numpy()

    num_frames = frames.shape[0]
    logger.info(f"Saving {num_frames} frames to {frames_dir}...")

    for i in range(num_frames):
        frame = frames[i]  # [H, W, C]
        img = Image.fromarray(frame)
        filename = f"{prefix}_{i:04d}.{image_format}"
        img.save(frames_dir / filename)

    logger.info(f"Saved {num_frames} frames to {frames_dir}")


# ==============================================================================
# Video File Loading (for V2T with direct video input)
# ==============================================================================

# Target resolution for video input (height x width)
V2T_TARGET_HEIGHT = 480
V2T_TARGET_WIDTH = 832

# Supported video file extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def is_video_file(path: str) -> bool:
    """Check if the path points to a video file based on extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def load_video_file(
    video_path: str,
    target_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    crop_mode: str = "center",
) -> torch.Tensor:
    """
    Load video file and convert to tensor format for V2T inference.

    Args:
        video_path: Path to input video file (.mp4, .avi, etc.)
        target_fps: Target FPS for frame sampling (None = use original FPS)
        max_frames: Maximum number of frames to extract (None = all frames)
        crop_mode: Crop position - "center", "top_left", "top_right",
                   "bottom_left", "bottom_right"

    Returns:
        Video frames tensor [B, C, T, H, W] in range [-1, 1]
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.error("OpenCV is required for video file input. Install with: pip install opencv-python")
        raise

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / original_fps if original_fps > 0 else 0

    logger.info(f"Input video properties:")
    logger.info(f"  Resolution: {original_width}x{original_height}")
    logger.info(f"  FPS: {original_fps:.2f}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Duration: {duration:.2f}s")

    # Validate dimensions
    if original_height < V2T_TARGET_HEIGHT or original_width < V2T_TARGET_WIDTH:
        cap.release()
        raise ValueError(
            f"Video ({original_width}x{original_height}) is smaller than target "
            f"({V2T_TARGET_WIDTH}x{V2T_TARGET_HEIGHT}). Upscaling is not supported."
        )

    # Calculate crop coordinates
    if crop_mode == "center":
        y_start = (original_height - V2T_TARGET_HEIGHT) // 2
        x_start = (original_width - V2T_TARGET_WIDTH) // 2
    elif crop_mode == "top_left":
        y_start, x_start = 0, 0
    elif crop_mode == "top_right":
        y_start = 0
        x_start = original_width - V2T_TARGET_WIDTH
    elif crop_mode == "bottom_left":
        y_start = original_height - V2T_TARGET_HEIGHT
        x_start = 0
    elif crop_mode == "bottom_right":
        y_start = original_height - V2T_TARGET_HEIGHT
        x_start = original_width - V2T_TARGET_WIDTH
    else:
        cap.release()
        raise ValueError(f"Unknown crop_mode: {crop_mode}")

    y_end = y_start + V2T_TARGET_HEIGHT
    x_end = x_start + V2T_TARGET_WIDTH

    logger.info(f"Crop region: ({x_start}, {y_start}) to ({x_end}, {y_end}) [{crop_mode}]")

    # Calculate frame sampling interval
    if target_fps is not None and target_fps < original_fps:
        frame_interval = original_fps / target_fps
        effective_fps = target_fps
    else:
        frame_interval = 1.0
        effective_fps = original_fps

    logger.info(f"Effective FPS: {effective_fps:.2f}")

    frames = []
    frame_idx = 0
    next_sample_idx = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frames according to target FPS
        if frame_idx >= next_sample_idx:
            # Crop frame
            cropped = frame[y_start:y_end, x_start:x_end]
            # Convert BGR to RGB
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            frames.append(cropped_rgb)
            next_sample_idx += frame_interval

            # Check max frames limit
            if max_frames is not None and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames extracted from video")

    logger.info(f"Extracted {len(frames)} frames -> tensor shape [1, 3, {len(frames)}, {V2T_TARGET_HEIGHT}, {V2T_TARGET_WIDTH}]")

    # Convert to tensor
    # Stack frames: list of [H, W, C] -> [T, H, W, C]
    import numpy as np
    frames_array = np.stack(frames, axis=0)

    # Rearrange to [T, C, H, W]
    frames_array = np.transpose(frames_array, (0, 3, 1, 2))

    # Convert to float32 and normalize to [-1, 1]
    frames_tensor = torch.from_numpy(frames_array).float()
    frames_tensor = (frames_tensor / 127.5) - 1.0  # [0, 255] -> [-1, 1]

    # Add batch dimension: [T, C, H, W] -> [B, C, T, H, W]
    frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)

    return frames_tensor


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """
    Load checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        device: Device to load checkpoint to

    Returns:
        Checkpoint dictionary with keys: model, optimizer, step, etc.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Validate checkpoint structure
    if "model" not in checkpoint:
        raise ValueError("Checkpoint missing 'model' key. Expected format from train_unified_wan.py")

    step = checkpoint.get("step", "unknown")
    logger.info(f"Loaded checkpoint from step {step}")

    return checkpoint


def build_model_config(
    wan_model_path: str,
    length: int = 1024,
    hidden_size: int = 1536,
    n_blocks: int = 30,
    n_heads: int = 12,
    cond_dim: int = 1536,
    dropout: float = 0.0,
) -> OmegaConf:
    """
    Build model configuration.

    Args:
        wan_model_path: Path to pretrained Wan model
        length: Maximum text sequence length
        hidden_size: Hidden dimension size
        n_blocks: Number of transformer blocks
        n_heads: Number of attention heads
        cond_dim: Conditioning dimension
        dropout: Dropout rate

    Returns:
        OmegaConf configuration object
    """
    config = OmegaConf.create({
        "wan_model_path": wan_model_path,
        "length": length,
        "hidden_size": hidden_size,
        "n_blocks": n_blocks,
        "n_heads": n_heads,
        "cond_dim": cond_dim,
        "dropout": dropout,
    })
    return config


def load_model_and_components(
    checkpoint_path: str,
    wan_model_path: str,
    device: torch.device,
    model_config: Optional[OmegaConf] = None,
) -> Tuple[torch.nn.Module, T5Tokenizer, T5EncoderModel, Optional[AutoencoderKLWan], int]:
    """
    Load the unified model and all required components.

    Args:
        checkpoint_path: Path to trained checkpoint
        wan_model_path: Path to pretrained Wan model directory
        device: Device to load model to
        model_config: Optional model configuration

    Returns:
        Tuple of (model, tokenizer, text_encoder, vae, vocab_size)
    """
    # Load tokenizer
    logger.info(f"Loading tokenizer from: {wan_model_path}")
    tokenizer = T5Tokenizer.from_pretrained(
        wan_model_path,
        subfolder="tokenizer",
        model_max_length=model_config.length if model_config else 1024,
    )
    vocab_size = tokenizer.vocab_size
    logger.info(f"Tokenizer loaded, vocab_size={vocab_size}")

    # Load T5 text encoder (for conditioning)
    logger.info("Loading T5 text encoder...")
    try:
        text_encoder = T5EncoderModel.from_pretrained(
            wan_model_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        ).to(device)
        text_encoder.eval()
        logger.info("T5 text encoder loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load T5 text encoder: {e}")
        logger.warning("Text encoding will need to be provided externally")
        text_encoder = None

    # Load VAE (optional, for video decoding)
    logger.info("Loading VAE...")
    try:
        vae = AutoencoderKLWan.from_pretrained(
            wan_model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(device)
        vae.eval()
        logger.info("VAE loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load VAE: {e}")
        logger.warning("Video decoding will not be available")
        vae = None

    # Build model config if not provided
    if model_config is None:
        model_config = build_model_config(wan_model_path)

    # Initialize model architecture
    logger.info("Initializing WanUnifiedTransformer...")

    # Determine if source distribution is masked
    source_distribution = "uniform"  # Default, matching training config
    masked = source_distribution == "mask"

    model = WanUnifiedTransformer(
        config=model_config,
        vocab_size=vocab_size,
        masked=masked,
        ckpt_path=wan_model_path,
    )

    # Load checkpoint weights
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_state_dict = checkpoint["model"]

    # Handle potential key mismatches from FSDP training
    # FSDP may add prefixes like "_fsdp_wrapped_module."
    cleaned_state_dict = {}
    for key, value in model_state_dict.items():
        # Remove common FSDP prefixes
        clean_key = key.replace("_fsdp_wrapped_module.", "")
        clean_key = clean_key.replace("_orig_mod.", "")
        cleaned_state_dict[clean_key] = value

    # Load state dict
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys in checkpoint: {len(missing)} keys")
        if len(missing) < 10:
            logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {len(unexpected)} keys")
        if len(unexpected) < 10:
            logger.warning(f"Unexpected keys: {unexpected}")

    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

    return model, tokenizer, text_encoder, vae, vocab_size


# ==============================================================================
# T2V Inference (Text-to-Video)
# ==============================================================================

@torch.no_grad()
def inference_t2v(
    model: torch.nn.Module,
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    vae: AutoencoderKLWan,
    prompt: str,
    device: torch.device,
    num_inference_steps: int = 50,
    video_shape: Tuple[int, ...] = (1, 16, 21, 60, 90),  # [B, C, T, H, W] latent shape
    cfg_scale: float = 1.0,
    scheduler_shift: float = 3.0,
    null_cond_t: Optional[torch.Tensor] = None,
    # T2VT mode parameters
    generate_text: bool = False,
    vocab_size: Optional[int] = None,
    text_max_length: int = 1024,
    source_distribution: str = "uniform",
) -> Tuple[torch.Tensor, torch.Tensor, Optional[str]]:
    """
    Generate video frames from text prompt, optionally with joint text generation (T2VT).

    Args:
        model: WanUnifiedTransformer model
        tokenizer: T5 tokenizer
        text_encoder: T5 text encoder
        vae: VAE for decoding latent to pixel space
        prompt: Text prompt for video generation
        device: Computation device
        num_inference_steps: Number of denoising steps
        video_shape: Output video latent shape [B, C, T, H, W]
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
        scheduler_shift: Flow matching scheduler shift parameter
        null_cond_t: Null condition T5 embedding for CFG [1, seq_len, hidden_dim].
                     Required when cfg_scale > 1.0.
        generate_text: If True, jointly generate text alongside video (T2VT mode).
        vocab_size: Vocabulary size (required when generate_text=True).
        text_max_length: Maximum text sequence length for T2VT.
        source_distribution: Source distribution for text ("uniform" or "mask").

    Returns:
        Tuple of (video_frames, video_latent, generated_text):
            - video_frames: Pixel-space video tensor [B, C, T, H, W] in range [-1, 1]
            - video_latent: VAE latent tensor [B, C, T, H, W] (for debugging)
            - generated_text: Generated text string (only when generate_text=True, else None)
    """
    if vae is None:
        raise ValueError("VAE is required for T2V inference to decode video frames")

    # CFG validation
    use_cfg = cfg_scale > 1.0
    if use_cfg and null_cond_t is None:
        raise ValueError("null_cond_t is required when cfg_scale > 1.0")

    # T2VT validation
    if generate_text and vocab_size is None:
        raise ValueError("vocab_size is required when generate_text=True (T2VT mode)")

    mode_str = "T2VT" if generate_text else "T2V"
    logger.info(f"{mode_str} Inference: '{prompt}'")
    logger.info(f"Latent shape: {video_shape}, Steps: {num_inference_steps}")
    if use_cfg:
        logger.info(f"CFG enabled: scale={cfg_scale}")
    if generate_text:
        logger.info(f"Text generation enabled: max_length={text_max_length}, source={source_distribution}")

    batch_size = video_shape[0]

    # Encode text prompt
    logger.info("Encoding text prompt...")
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    t5_tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    # Get T5 embeddings for conditioning
    if text_encoder is not None:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            encoder_output = text_encoder(
                input_ids=t5_tokens["input_ids"].to(device),
                attention_mask=t5_tokens["attention_mask"].to(device),
            )
            cond_t = encoder_output.last_hidden_state  # [1, seq_len, hidden_dim]
    else:
        raise ValueError("T5 text encoder required for T2V inference")

    # Initialize video noise
    x_v = torch.randn(video_shape, device=device, dtype=torch.bfloat16)

    # Initialize flow scheduler for video denoising
    flow_scheduler = FlowMatchScheduler(
        num_train_timesteps=1000,
        num_inference_steps=num_inference_steps,
        shift=scheduler_shift,
        sigma_min=0.0,
        extra_one_step=True,
        is_training=False,
    )
    flow_scheduler.set_timesteps(num_inference_steps, device=str(device))

    # Text token initialization
    if generate_text:
        # T2VT mode: Initialize text from source distribution for joint generation
        # The text will be denoised alongside video
        if source_distribution == "mask":
            src_dist = MaskedSourceDistribution(mask_token=vocab_size)
        else:
            src_dist = UniformSourceDistribution(vocab_size=vocab_size)
        x_t = src_dist.sample(
            tensor_size=(batch_size, text_max_length),
            device=device,
        )
        # No attention mask needed for generated text (all tokens are valid)
        x_t_attention_mask = torch.ones(batch_size, text_max_length, device=device, dtype=torch.bool)
        logger.info(f"Initialized text tokens from {source_distribution} distribution: shape={x_t.shape}")
    else:
        # T2V mode: Use the tokenized prompt as fixed conditioning
        x_t = input_ids.expand(batch_size, -1)
        x_t_attention_mask = attention_mask.expand(batch_size, -1)

    logger.info(f"Starting denoising loop ({num_inference_steps} steps)...")

    # Denoising loop
    for i, timestep in enumerate(flow_scheduler.timesteps):
        if i % 10 == 0:
            logger.info(f"  Step {i+1}/{num_inference_steps}, timestep={timestep.item():.1f}")

        time_v = timestep.expand(batch_size).to(device, dtype=x_v.dtype)

        # Compute text timestep
        if generate_text:
            # T2VT mode: text timestep synchronized with video denoising progress
            # Map video timestep to text flow matching time
            # Video timestep goes from ~1000 to 0, text time_t goes from ~1000 to 0
            # In training: time_t = (1.0 - t) * num_train_timesteps where t is in [0,1]
            # So when video timestep is high (early), text timestep should also be high (noisy)
            time_t = timestep.expand(batch_size).to(device, dtype=x_v.dtype)
        else:
            # T2V mode: Text timestep = 0 (fully "denoised" / fixed prompt)
            time_t = torch.zeros(batch_size, device=device, dtype=x_v.dtype)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if use_cfg:
                # CFG: run model with both conditional and unconditional inputs
                # Conditional forward pass
                velocity_cond, logits_t = model(
                    x_v=x_v,
                    x_t=x_t,
                    time_v=time_v,
                    time_t=time_t,
                    cond_t=cond_t.expand(batch_size, -1, -1),
                    x_t_attention_mask=x_t_attention_mask,
                )
                # Unconditional forward pass (with null condition)
                velocity_uncond, _ = model(
                    x_v=x_v,
                    x_t=x_t,
                    time_v=time_v,
                    time_t=time_t,
                    cond_t=null_cond_t.expand(batch_size, -1, -1),
                    x_t_attention_mask=x_t_attention_mask,
                )
                # Apply CFG formula: v = v_uncond + cfg_scale * (v_cond - v_uncond)
                velocity_pred_v = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
            else:
                # Standard forward pass without CFG
                velocity_pred_v, logits_t = model(
                    x_v=x_v,
                    x_t=x_t,
                    time_v=time_v,
                    time_t=time_t,
                    cond_t=cond_t.expand(batch_size, -1, -1),
                    x_t_attention_mask=x_t_attention_mask,
                )

        # Update video latent
        x_v = flow_scheduler.step(velocity_pred_v, timestep, x_v)

        # Update text tokens (T2VT mode only)
        if generate_text:
            # Discrete flow matching update for text
            # Compute interpolation weight based on progress (0 = start, 1 = end)
            progress = i / (num_inference_steps - 1) if num_inference_steps > 1 else 1.0
            # Get probabilities from logits
            probs = torch.softmax(logits_t.float(), dim=-1)
            # Sample from the predicted distribution
            sampled_tokens = torch.multinomial(
                probs.view(-1, probs.size(-1)), num_samples=1
            ).view(batch_size, -1)
            # Discrete flow matching: interpolate between source and target
            # With probability proportional to progress, keep the sampled token
            # Early steps: more likely to keep source (noisy), late steps: more likely to use predicted
            keep_sampled = torch.rand(x_t.shape, device=device) < progress
            x_t = torch.where(keep_sampled, sampled_tokens, x_t)

    video_latent = x_v.clone()

    # Decode latent to pixel-space video frames
    logger.info("Decoding video latent to pixel frames...")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        video_frames = vae.decode(x_v).sample

    logger.info(f"Video frames shape: {video_frames.shape}")

    # Decode generated text (T2VT mode only)
    generated_text = None
    if generate_text:
        # Final refinement: take argmax of last forward pass logits for cleaner output
        # Run one more forward pass with time_t=0 to get final predictions
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            time_v_final = torch.zeros(batch_size, device=device, dtype=video_latent.dtype)
            time_t_final = torch.zeros(batch_size, device=device, dtype=video_latent.dtype)
            _, final_logits = model(
                x_v=video_latent,
                x_t=x_t,
                time_v=time_v_final,
                time_t=time_t_final,
                cond_t=cond_t.expand(batch_size, -1, -1),
                x_t_attention_mask=x_t_attention_mask,
            )
        # Take argmax to get final token predictions
        final_tokens = torch.argmax(final_logits, dim=-1)
        # Decode tokens to text
        generated_texts = tokenizer.batch_decode(final_tokens, skip_special_tokens=True)
        generated_text = generated_texts[0] if len(generated_texts) == 1 else generated_texts
        logger.info(f"Generated text: {generated_text[:200]}..." if len(str(generated_text)) > 200 else f"Generated text: {generated_text}")

    logger.info(f"{mode_str} inference complete!")

    return video_frames, video_latent, generated_text


# ==============================================================================
# V2T Inference (Video-to-Text)
# ==============================================================================

@torch.no_grad()
def inference_v2t(
    model: torch.nn.Module,
    tokenizer: T5Tokenizer,
    text_encoder: Optional[T5EncoderModel],
    vae: AutoencoderKLWan,
    video_frames: torch.Tensor,
    device: torch.device,
    vocab_size: int,
    num_inference_steps: int = 50,
    max_length: int = 1024,
    source_distribution: str = "uniform",
    scheduler_exponent: float = 1.0,
    cond_text: Optional[str] = None,
) -> str:
    """
    Generate text caption from video frames.

    Args:
        model: WanUnifiedTransformer model
        tokenizer: T5 tokenizer
        text_encoder: T5 text encoder (optional, for conditioning)
        vae: VAE for encoding video frames to latent space
        video_frames: Video frames tensor [B, C, T, H, W] in pixel space
                      Values can be in range [-1, 1] or [0, 1] (auto-normalized)
        device: Computation device
        vocab_size: Vocabulary size
        num_inference_steps: Number of sampling steps
        max_length: Maximum output text length
        source_distribution: "uniform" or "mask"
        scheduler_exponent: Polynomial scheduler exponent
        cond_text: Optional conditioning text (for guided generation)

    Returns:
        Generated text string
    """
    if vae is None:
        raise ValueError("VAE is required for V2T inference to encode video frames")

    logger.info("V2T Inference starting...")
    logger.info(f"Video frames shape: {video_frames.shape}")
    logger.info(f"Steps: {num_inference_steps}, Max length: {max_length}")

    # Move video frames to device and normalize if needed
    video_frames = video_frames.to(device, dtype=torch.bfloat16)

    # Auto-normalize: if values are in [0, 1], convert to [-1, 1]
    if video_frames.min() >= 0 and video_frames.max() <= 1:
        logger.info("Video frames appear to be in [0, 1] range, normalizing to [-1, 1]")
        video_frames = video_frames * 2 - 1

    # Encode video frames to latent space using VAE
    logger.info("Encoding video frames to latent space...")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        video_latent = vae.encode(video_frames).latent_dist.sample()
        # Apply VAE scaling factor if available
        if hasattr(vae.config, 'scaling_factor'):
            video_latent = video_latent * vae.config.scaling_factor

    logger.info(f"Video latent shape: {video_latent.shape}")

    batch_size = video_latent.shape[0]

    # Prepare conditioning
    if cond_text is not None and text_encoder is not None:
        logger.info(f"Using conditioning text: '{cond_text}'")
        tokens = tokenizer(
            cond_text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            encoder_output = text_encoder(
                input_ids=tokens["input_ids"].to(device),
                attention_mask=tokens["attention_mask"].to(device),
            )
            cond_t = encoder_output.last_hidden_state
    else:
        # Create dummy conditioning (zeros)
        logger.info("No conditioning text provided, using zero conditioning")
        # T5 encoder output dimension (NOT model hidden size)
        # The cond_text_embedder projects from T5 dim (4096) to model hidden size (1536)
        if text_encoder is not None and hasattr(text_encoder, "config"):
            t5_hidden_dim = text_encoder.config.hidden_size
        else:
            t5_hidden_dim = 4096  # Default for T5/UMT5-XXL
        cond_t = torch.zeros(batch_size, 512, t5_hidden_dim, device=device, dtype=torch.bfloat16)

    # Initialize source distribution
    if source_distribution == "mask":
        src_dist = MaskedSourceDistribution(mask_token=vocab_size)
        add_token = 1
    else:
        src_dist = UniformSourceDistribution(vocab_size=vocab_size)
        add_token = 0

    # Initialize discrete flow matching path and solver
    scheduler = PolynomialConvexScheduler(n=scheduler_exponent)
    path = MixtureDiscreteProbPath(scheduler=scheduler)

    # Wrap model for text generation
    wrapped_model = TextModelWrapper(
        model=model,
        video_latent=video_latent,
        cond_t=cond_t,
        num_train_timesteps=1000,
        device=device,
    )

    solver = MixtureDiscreteEulerSolver(
        model=wrapped_model,
        path=path,
        vocabulary_size=vocab_size + add_token,
    )

    # Initialize text tokens from source distribution
    x_init = src_dist.sample(
        tensor_size=(batch_size, max_length),
        device=device,
    )

    logger.info(f"Starting text sampling ({num_inference_steps} steps)...")

    # Sample using discrete flow matching
    sample = solver.sample(
        x_init=x_init,
        step_size=1 / num_inference_steps,
        verbose=True,
        dtype_categorical=torch.float64,
        time_grid=torch.tensor([0.0, 1.0]),
    )

    # Decode tokens to text
    generated_text = tokenizer.batch_decode(sample, skip_special_tokens=True)

    logger.info("V2T inference complete!")
    return generated_text[0] if len(generated_text) == 1 else generated_text


# ==============================================================================
# Dry Run Validation
# ==============================================================================

@torch.no_grad()
def dry_run(
    model: torch.nn.Module,
    tokenizer: T5Tokenizer,
    device: torch.device,
    vocab_size: int,
) -> bool:
    """
    Validate that the model can perform forward passes with dummy inputs.

    Args:
        model: WanUnifiedTransformer model
        tokenizer: T5 tokenizer
        device: Computation device
        vocab_size: Vocabulary size

    Returns:
        True if validation passes, False otherwise
    """
    logger.info("=" * 60)
    logger.info("Starting dry run validation...")
    logger.info("=" * 60)

    try:
        # Create dummy inputs
        batch_size = 1
        seq_len = 64  # Small sequence for speed
        video_shape = (batch_size, 16, 5, 30, 45)  # Small video

        logger.info(f"Creating dummy inputs...")
        logger.info(f"  Video shape: {video_shape}")
        logger.info(f"  Text seq_len: {seq_len}")

        x_v = torch.randn(video_shape, device=device, dtype=torch.bfloat16)
        x_t = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        time_v = torch.tensor([500.0], device=device, dtype=torch.bfloat16)
        time_t = torch.tensor([500.0], device=device, dtype=torch.bfloat16)
        # cond_t uses T5 encoder output dimension (4096), NOT model hidden size (1536)
        cond_t = torch.randn(batch_size, 512, 4096, device=device, dtype=torch.bfloat16)
        attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

        logger.info("Running forward pass...")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out_v, out_t = model(
                x_v=x_v,
                x_t=x_t,
                time_v=time_v,
                time_t=time_t,
                cond_t=cond_t,
                x_t_attention_mask=attention_mask,
            )

        logger.info(f"Forward pass successful!")
        logger.info(f"  Video output shape: {out_v.shape}")
        logger.info(f"  Text output shape: {out_t.shape}")
        logger.info(f"  Video output dtype: {out_v.dtype}")
        logger.info(f"  Text output dtype: {out_t.dtype}")

        # Verify output shapes
        expected_video_shape = video_shape
        expected_text_shape = (batch_size, seq_len, vocab_size)

        if out_v.shape != expected_video_shape:
            logger.warning(f"Video output shape mismatch: expected {expected_video_shape}, got {out_v.shape}")
        if out_t.shape != expected_text_shape:
            logger.warning(f"Text output shape mismatch: expected {expected_text_shape}, got {out_t.shape}")

        # Check for NaN/Inf
        if torch.isnan(out_v).any() or torch.isinf(out_v).any():
            logger.warning("Video output contains NaN or Inf values!")
        if torch.isnan(out_t).any() or torch.isinf(out_t).any():
            logger.warning("Text output contains NaN or Inf values!")

        logger.info("=" * 60)
        logger.info("✓ Dry run validation PASSED")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"Dry run validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# Main Entry Point
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference script for WanUnifiedTransformer (T2V and V2T)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pth file)",
    )
    parser.add_argument(
        "--wan_model_path",
        type=str,
        required=True,
        help="Path to pretrained Wan model directory (diffusers format)",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["t2v", "t2vt", "v2t"],
        default="t2v",
        help="Inference mode: t2v (text-to-video), t2vt (text-to-video-text joint), or v2t (video-to-text)",
    )

    # T2V specific arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful sunset over the ocean with waves crashing on the shore.",
        help="Text prompt for T2V generation",
    )
    parser.add_argument(
        "--video_frames",
        type=int,
        default=21,
        help="Number of video frames to generate (T2V)",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=480,
        help="Video height in pixels (will be converted to latent size)",
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=720,
        help="Video width in pixels (will be converted to latent size)",
    )

    # V2T specific arguments
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to video input for V2T. Accepts: .pt tensor file [B,C,T,H,W] "
             "or video file (.mp4, .avi, .mov, etc.) which will be auto-converted to 480x832",
    )
    parser.add_argument(
        "--cond_text",
        type=str,
        default=None,
        help="Optional conditioning text for V2T (guides generation)",
    )
    # Keep --video_latent as alias for backward compatibility
    parser.add_argument(
        "--video_latent",
        type=str,
        default=None,
        help="[DEPRECATED] Use --video_path instead. Path to video tensor file.",
    )
    # Video file input options (only used when --video_path is a video file)
    parser.add_argument(
        "--input_fps",
        type=float,
        default=None,
        help="Target FPS for sampling frames from video file (default: use original FPS)",
    )
    parser.add_argument(
        "--max_input_frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract from video file (default: all frames)",
    )
    parser.add_argument(
        "--crop_mode",
        type=str,
        choices=["center", "top_left", "top_right", "bottom_left", "bottom_right"],
        default="center",
        help="Crop position when converting video file to 480x832 (default: center)",
    )

    # Common arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum text sequence length",
    )

    # Model configuration
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1536,
        help="Model hidden size",
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=30,
        help="Number of transformer blocks",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )

    # Flow matching parameters
    parser.add_argument(
        "--source_distribution",
        type=str,
        choices=["uniform", "mask"],
        default="uniform",
        help="Source distribution for text flow matching",
    )
    parser.add_argument(
        "--scheduler_shift",
        type=float,
        default=3.0,
        help="Flow matching scheduler shift parameter",
    )

    # Classifier-free guidance parameters
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger guidance)",
    )
    parser.add_argument(
        "--null_cond_path",
        type=str,
        default="empty_context_t5_emb.pt",
        help="Path to pre-computed null condition T5 embedding (.pt file) for CFG. "
             "If not provided but cfg_scale > 1.0, will encode empty string as null condition.",
    )

    # Special modes
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run validation only (no actual generation)",
    )
    parser.add_argument(
        "--save_latent",
        action="store_true",
        help="Also save VAE latent alongside video frames (for debugging)",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save video frames as individual PNG images (T2V only)",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        choices=["png", "jpg"],
        default="png",
        help="Image format for saving frames (default: png)",
    )

    # Video export arguments
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Export video frames as MP4/GIF file (T2V only)",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=24,
        help="Frames per second for exported video (default: 24)",
    )
    parser.add_argument(
        "--video_format",
        type=str,
        choices=["mp4", "gif"],
        default="mp4",
        help="Video format for export (default: mp4)",
    )
    parser.add_argument(
        "--video_quality",
        type=float,
        default=8.0,
        help="Video quality for MP4 export (0-10, higher is better, default: 8.0)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model config
    model_config = build_model_config(
        wan_model_path=args.wan_model_path,
        length=args.max_length,
        hidden_size=args.hidden_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
    )

    # Load model and components
    logger.info("Loading model and components...")
    model, tokenizer, text_encoder, vae, vocab_size = load_model_and_components(
        checkpoint_path=args.checkpoint,
        wan_model_path=args.wan_model_path,
        device=device,
        model_config=model_config,
    )

    # Dry run mode
    if args.dry_run:
        success = dry_run(model, tokenizer, device, vocab_size)
        sys.exit(0 if success else 1)

    # T2V / T2VT Mode (both handled by the same code path)
    if args.mode in ("t2v", "t2vt"):
        if not args.prompt:
            logger.error(f"{args.mode.upper()} mode requires --prompt argument")
            sys.exit(1)

        if vae is None:
            logger.error(f"{args.mode.upper()} mode requires VAE for decoding video frames")
            sys.exit(1)

        # Calculate latent dimensions (assuming 8x spatial downsampling from VAE)
        # latent_height = args.video_height // 8
        # latent_width = args.video_width // 8
        # NOTE: Temporally fixed to 480*832
        # wan2.2 480p -> 480*832
        latent_height = 480 // 8
        latent_width = 832 // 8
        video_shape = (1, 16, args.video_frames, latent_height, latent_width)

        # Prepare null condition for CFG if needed
        null_cond_t = None
        if args.cfg_scale > 1.0:
            if args.null_cond_path is not None:
                # Load pre-computed null condition embedding
                logger.info(f"Loading null condition from: {args.null_cond_path}")
                null_cond_t = torch.load(args.null_cond_path, map_location=device, weights_only=True)
                null_cond_t = null_cond_t.to(device, dtype=torch.bfloat16)
            else:
                # Generate null condition by encoding empty string
                if text_encoder is None:
                    logger.error("T5 text encoder required to generate null condition for CFG")
                    sys.exit(1)
                logger.info("Generating null condition by encoding empty string...")
                empty_tokens = tokenizer(
                    "",
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    encoder_output = text_encoder(
                        input_ids=empty_tokens["input_ids"].to(device),
                        attention_mask=empty_tokens["attention_mask"].to(device),
                    )
                    null_cond_t = encoder_output.last_hidden_state

        # Determine if we're doing joint text generation (T2VT)
        generate_text = args.mode == "t2vt"

        logger.info(f"{args.mode.upper()}: Generating {'video + text' if generate_text else 'video'} for prompt: '{args.prompt}'")
        video_frames, video_latent, generated_text = inference_t2v(
            model=model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            prompt=args.prompt,
            device=device,
            num_inference_steps=args.num_inference_steps,
            video_shape=video_shape,
            scheduler_shift=args.scheduler_shift,
            cfg_scale=args.cfg_scale,
            null_cond_t=null_cond_t,
            # T2VT-specific parameters
            generate_text=generate_text,
            vocab_size=vocab_size if generate_text else None,
            text_max_length=args.max_length,
            source_distribution=args.source_distribution,
        )

        # Save video frames (pixel space)
        frames_path = Path(args.output_dir) / "video_frames.pt"
        torch.save(video_frames.cpu(), frames_path)
        logger.info(f"Video frames saved to: {frames_path}")
        logger.info(f"  Shape: {video_frames.shape}, Range: [{video_frames.min():.3f}, {video_frames.max():.3f}]")

        # Optionally save latent for debugging
        if args.save_latent:
            latent_path = Path(args.output_dir) / "video_latent.pt"
            torch.save(video_latent.cpu(), latent_path)
            logger.info(f"Video latent saved to: {latent_path}")

        # Optionally save frames as individual images
        if args.save_images:
            save_video_frames_as_images(
                video_frames=video_frames,
                output_dir=args.output_dir,
                image_format=args.image_format,
                prefix="frame",
            )

        # Optionally export as video file (MP4 or GIF)
        if args.save_video:
            video_filename = f"output.{args.video_format}"
            video_path = Path(args.output_dir) / video_filename
            logger.info(f"Exporting video to: {video_path}")

            if args.video_format == "mp4":
                tensor_to_video(
                    video_tensor=video_frames,
                    output_path=video_path,
                    fps=args.video_fps,
                    quality=args.video_quality,
                )
            else:  # gif
                tensor_to_gif(
                    video_tensor=video_frames,
                    output_path=video_path,
                    fps=args.video_fps,
                )
            logger.info(f"Video exported to: {video_path}")

        # Save generated text (T2VT mode only)
        if generate_text and generated_text is not None:
            text_output_path = Path(args.output_dir) / "generated_text.txt"
            with open(text_output_path, "w", encoding="utf-8") as f:
                f.write(generated_text if isinstance(generated_text, str) else "\n".join(generated_text))
            logger.info(f"Generated text saved to: {text_output_path}")
            logger.info(f"Generated text:\n{'-' * 40}\n{generated_text}\n{'-' * 40}")

    # V2T Mode
    elif args.mode == "v2t":
        # Support both --video_path and deprecated --video_latent
        video_input_path = args.video_path or args.video_latent
        if not video_input_path:
            logger.error("V2T mode requires --video_path argument")
            sys.exit(1)

        if not os.path.exists(video_input_path):
            logger.error(f"Video file not found: {video_input_path}")
            sys.exit(1)

        if vae is None:
            logger.error("V2T mode requires VAE for encoding video frames")
            sys.exit(1)

        # Load video frames - either from .pt file or video file
        if is_video_file(video_input_path):
            logger.info(f"V2T: Loading video file: {video_input_path}")
            video_frames = load_video_file(
                video_path=video_input_path,
                target_fps=args.input_fps,
                max_frames=args.max_input_frames,
                crop_mode=args.crop_mode,
            )
        else:
            # Assume it's a .pt tensor file
            logger.info(f"V2T: Loading tensor file: {video_input_path}")
            video_frames = torch.load(video_input_path, map_location=device, weights_only=True)

        # Ensure correct shape [B, C, T, H, W]
        if video_frames.dim() == 4:
            video_frames = video_frames.unsqueeze(0)

        generated_text = inference_v2t(
            model=model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            video_frames=video_frames,
            device=device,
            vocab_size=vocab_size,
            num_inference_steps=args.num_inference_steps,
            max_length=args.max_length,
            source_distribution=args.source_distribution,
            cond_text=args.cond_text,
        )

        # Save and print result
        output_path = Path(args.output_dir) / "generated_text.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(generated_text)

        logger.info(f"Generated text saved to: {output_path}")
        logger.info(f"Generated text:\n{'-' * 40}\n{generated_text}\n{'-' * 40}")

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
