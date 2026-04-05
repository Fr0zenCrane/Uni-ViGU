# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/facebookresearch/DiT
# which is released under NonCommercial-4.0 license
# Part of this implementation is adapted from https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
# which is released under MIT license
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

import math
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from torch import nn, Tensor

from . import rotary

from diffusers.models.transformers import WanTransformerUnifiedBlock
from diffusers.models import WanTransformer3DModel


def bias_dropout_add_scale(
    x: Tensor, scale: Tensor, residual: Optional[Tensor], prob: float, training: bool
) -> Tensor:
    return residual + scale * F.dropout(x, p=prob, training=training)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])

        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(time: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time.device)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, time: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be devisable by n_heads"

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout

        self.head_dim = self.dim // self.n_heads

        self.norm1 = LayerNorm(dim=dim)

        self.qw = nn.Linear(dim, dim, bias=False)
        self.kw = nn.Linear(dim, dim, bias=False)
        self.vw = nn.Linear(dim, dim, bias=False)

        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: Tensor, rotary_cos_sin: Tensor, c: Tensor) -> Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        x_skip = x
        x = modulate(x=self.norm1(x), shift=shift_msa, scale=scale_msa)

        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        q, k, v = (
            item.view(batch_size, seq_len, self.n_heads, self.head_dim)
            for item in (q, k, v)
        )

        with torch.amp.autocast("cuda", enabled=False):
            cos, sin = rotary_cos_sin
            original_dtype = q.dtype

            q = rotary.apply_rotary_emb_torch(
                x=q.float(), cos=cos.float(), sin=sin.float()
            ).to(original_dtype)
            k = rotary.apply_rotary_emb_torch(
                x=k.float(), cos=cos.float(), sin=sin.float()
            ).to(original_dtype)

        q, k, v = (item.transpose(1, 2) for item in (q, k, v))

        x = F.scaled_dot_product_attention(query=q, key=k, value=v)
        x = rearrange(x, "b h s d -> b s (h d)", b=batch_size)
        x = bias_dropout_add_scale(
            x=self.attn_out(x),
            scale=gate_msa,
            residual=x_skip,
            prob=self.dropout,
            training=self.training,
        )
        x = bias_dropout_add_scale(
            x=self.mlp(modulate(x=self.norm2(x), shift=shift_mlp, scale=scale_mlp)),
            scale=gate_mlp,
            residual=x,
            prob=self.dropout,
            training=self.training,
        )

        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(x=self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)

        return x


class WanDDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x: Tensor, shift, scale) -> Tensor:
        x = modulate(x=self.norm_final(x), shift=shift, scale=scale).to(x.dtype)
        x = self.linear(x)

        return x


class WanDitFinalLayer(nn.Module):
    def __init__(self, wan_model):
        super().__init__()
        self.norm_out = wan_model.norm_out
        self.proj_out = wan_model.proj_out

    def forward(self, x: Tensor, shift, scale) -> Tensor:
        x = (self.norm_out(x.float()) * (1 + scale) + shift).type_as(x)
        x = self.proj_out(x)
        
        return x

class WanDDitTimeEmbedding(nn.Module):
    def __init__(
        self,
        wan_condition_embedder
    ):
        super().__init__()

        self.timesteps_proj = wan_condition_embedder.timesteps_proj
        self.time_embedder = wan_condition_embedder.time_embedder
        self.act_fn = wan_condition_embedder.act_fn
        self.time_proj = wan_condition_embedder.time_proj

    def forward(
        self,
        timestep: torch.Tensor,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)

        temb = self.time_embedder(timestep).to(torch.bfloat16)
        timestep_proj = self.time_proj(self.act_fn(temb))

        return temb, timestep_proj

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, masked: bool, config: DictConfig):
        super().__init__()

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size

        add_token = 1 if masked else 0

        self.vocab_embed = nn.Embedding(self.vocab_size + add_token, config.hidden_size)

        self.time_embedding = TimestepEmbedder(hidden_size=config.cond_dim)
        self.rotary_emb = rotary.Rotary(dim=config.hidden_size // config.n_heads)

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=config.hidden_size,
                    n_heads=config.n_heads,
                    cond_dim=config.cond_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )

        self.output_layer = DDitFinalLayer(
            hidden_size=config.hidden_size,
            out_channels=vocab_size + add_token,
            cond_dim=config.cond_dim,
        )

    def forward(self, x_t: Tensor, time: Tensor) -> Tensor:
        """
        Input:
            x_t:    Tensor [bs, seq_len]
            time:   Tensor [bs]
        Output:
            x:      Tensor [bs, seq_len, vocab_size]
        """
        x = self.vocab_embed(x_t) # [bs, seq_len] -> [bs, seq_len, hidden_size]
        c = F.silu(self.time_embedding(time=time)) # [bs, cond_size]
        rotary_cos_sin = self.rotary_emb(x=x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):        
            for i in range(len(self.blocks)):
                x = self.blocks[i](x=x, rotary_cos_sin=rotary_cos_sin, c=c) # x.shape is consistent as [bs, seq_len, hidden_size]

            x = self.output_layer(x=x, c=c) # [bs, seq_len, hidden_size] -> [bs, seq_len, vocab_size]

        return x


class WanDiscrete2DTransformer(nn.Module):
    def __init__(self, vocab_size: int, masked: bool, config: DictConfig, ckpt_path):
        super().__init__()

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size

        add_token = 1 if masked else 0

        self.vocab_embed = nn.Embedding(self.vocab_size + add_token, config.hidden_size)
        
        # init blocks and embeddings from wan
        wan_model = WanTransformer3DModel.from_pretrained(
            ckpt_path,
            subfolder="transformer",
            device_map="cuda",
        )
        self.blocks = wan_model.blocks
        
        self.time_embedding = WanDDitTimeEmbedding(wan_model.condition_embedder)
        self.scale_shift_table = wan_model.scale_shift_table
        self.rotary_emb = rotary.Rotary(dim=config.hidden_size // config.n_heads)
        
        self.output_layer = WanDDitFinalLayer(
            hidden_size=config.hidden_size,
            out_channels=vocab_size + add_token,
            cond_dim=config.cond_dim,
        )
        
        del wan_model 
        torch.cuda.empty_cache()

    def get_input_embedding(self, x_t:Tensor) -> Tensor:
        """
        Input:
            x_t:    Tensor [bs, seq_len]
        Output:
            x:      Tensor [bs, seq_len, vocab_size]
        """        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x = self.vocab_embed(x_t) # [bs, seq_len] -> [bs, seq_len, hidden_size]
        
        return x        

    def forward(self, x_t: Tensor, time: Tensor) -> Tensor:
        """
        Input:
            x_t:    Tensor [bs, seq_len]
            time:   Tensor [bs]
        Output:
            x:      Tensor [bs, seq_len, vocab_size]
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x = self.vocab_embed(x_t) # [bs, seq_len] -> [bs, seq_len, hidden_size]
            temb, timestep_proj = self.time_embedding(time) # [bs, cond_size]
            timestep_proj = timestep_proj.unflatten(1, (6, -1)) # [bs, 6, cond_size]

            rotary_cos_sin = self.rotary_emb(x=x)        
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, x, timestep_proj, rotary_emb_1d=rotary_cos_sin) # x.shape is consistent as [bs, seq_len, hidden_size]
            
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
            x = self.output_layer(x, shift, scale) # [bs, seq_len, hidden_size] -> [bs, seq_len, vocab_size]

        return x


class WanUnifiedTransformer(nn.Module):
    """
    Unified Wan DiT transformer capable of modeling video-text joint distribution.
    """
    def __init__(self, vocab_size: int, masked: bool, config: DictConfig, ckpt_path):
        super().__init__()

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size

        add_token = 1 if masked else 0

        self.vocab_embed = nn.Embedding(self.vocab_size + add_token, config.hidden_size)
        
        # init blocks and embeddings from wan
        wan_model = WanTransformer3DModel.from_pretrained(
            ckpt_path,
            subfolder="transformer",
            device_map="cuda",
        )
        self.blocks = nn.ModuleList([
            WanTransformerUnifiedBlock(block) for block in wan_model.blocks
        ])        
        # text conditioning and time conditioning
        self.time_embedding_v = WanDDitTimeEmbedding(wan_model.condition_embedder)
        self.time_embedding_t = copy.deepcopy(self.time_embedding_v)
        self.cond_text_embedder = wan_model.condition_embedder.text_embedder
        self.scale_shift_table_v = copy.deepcopy(wan_model.scale_shift_table)
        self.scale_shift_table_t = copy.deepcopy(wan_model.scale_shift_table)
        # rope position embedding
        self.rotary_emb_text = rotary.Rotary(dim=config.hidden_size // config.n_heads)
        self.rotary_emb_video = wan_model.rope
        # transfer 3d vae latents to sequence
        self.patch_embedding = wan_model.patch_embedding
        
        self.text_head = WanDDitFinalLayer(
            hidden_size=config.hidden_size,
            out_channels=vocab_size + add_token,
            cond_dim=config.cond_dim,
        )
        self.video_head = WanDitFinalLayer(
            wan_model = wan_model
        )
        
        del wan_model 
        torch.cuda.empty_cache()     

    def forward(self, 
                x_v: Tensor, 
                x_t: Tensor, 
                time_v: Tensor,
                time_t: Tensor, 
                cond_t: Tensor,
                x_t_attention_mask: Optional[Tensor] = None,
            ) -> Tensor:
        """
        Input:
            x_v:      Tensor [bs, channel, time, height, width] 
            x_t:      Tensor [bs, seq_len]
            time_v:   Tensor [bs]
            time_t:   Tensor [bs]
            cond_t:   Tensor [bs, 512, text_embedder_dim]
            x_t_attention_mask: Tensor [bs, seq_len]
        Output:
            x_v:      Tensor [bs, channel, time, height, width] 
            x_t:      Tensor [bs, seq_len] text token sequence
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # 1.Prepare Input
            # text embedding & 1d rope
            x_t = self.vocab_embed(x_t) # [bs, seq_len] -> [bs, seq_len, hidden_size]
            rotary_cos_sin_1d = self.rotary_emb_text(x_t)
            
            # 3d video rope before patch embedding
            rotary_cos_sin_3d = self.rotary_emb_video(x_v)
            # obtain dim for video output reshape 
            batch_size, _, num_frames, height, width = x_v.shape
            p_t, p_h, p_w = 1, 2, 2 # fixed patch size (1, 2, 2)
            post_patch_num_frames = num_frames // p_t
            post_patch_height = height // p_h
            post_patch_width = width // p_w
            # video patch embedding and build attention mask
            x_v = self.patch_embedding(x_v).flatten(2).transpose(1, 2) # [bs, c, t, h, w] -> [bs, hid, t', h', w'] -> [bs, patch_num, hid]
            if x_t_attention_mask is not None:
                x_t_attention_mask = x_t_attention_mask.bool()
                x_v_attention_mask = torch.ones(x_v.shape[:2], device=x_v.device, dtype=x_t_attention_mask.dtype)
                attention_mask = torch.cat([x_v_attention_mask, x_t_attention_mask], dim=1)
            else:
                attention_mask = None
            # t5 text conditioning
            cond_t = self.cond_text_embedder(cond_t) # [bs, 512, embed_size] -> [bs, 512, hidden_size]

            # time conditioning for text, video
            temb_t, token_proj_t = self.time_embedding_t(time_t)
            temb_v, token_proj_v = self.time_embedding_v(time_v)
            timestep_proj_t = token_proj_t.unflatten(1, (6, -1)) # [bs, 6 * cond_size] -> [bs, 6, cond_size]
            timestep_proj_v = token_proj_v.unflatten(1, (6, -1)) # [bs, 6 * cond_size] -> [bs, 6, cond_size]

            #2. DiT processing
            for i in range(len(self.blocks)):
                x_v, x_t = self.blocks[i](x_v, x_t, cond_t, timestep_proj_v, timestep_proj_t, rotary_cos_sin_3d, rotary_cos_sin_1d, attention_mask) # x.shape is consistent as [bs, seq_len, hidden_size]

            # 3. Output processing
            # video output: norm, projection & unpatchify
            shift_v, scale_v = (self.scale_shift_table_v.to(temb_v.device) + temb_v.unsqueeze(1)).chunk(2, dim=1)
            x_v = self.video_head(x_v, shift_v, scale_v) # [bs, vid_seq_len, hidden_size] -> [bs, vid_seq_len, 64(vae_channel*pt*ph*pw)]
            x_v = x_v.reshape(
                batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
            )
            x_v = x_v.permute(0, 7, 1, 4, 2, 5, 3, 6)
            x_v = x_v.flatten(6, 7).flatten(4, 5).flatten(2, 3) # reshape to b c t h w
            
            # text output: norm & projection
            shift_t, scale_t = (self.scale_shift_table_t.to(temb_t.device) + temb_t.unsqueeze(1)).chunk(2, dim=1)
            x_t = self.text_head(x_t, shift_t, scale_t) # [bs, seq_len, hidden_size] -> [bs, seq_len, vocab_size]

        return x_v, x_t
