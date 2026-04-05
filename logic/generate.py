# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional

import torch
from flow_matching.path import ProbPath
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from torch import nn, Tensor
from transformers.tokenization_utils import PreTrainedTokenizer

from .flow import SourceDistribution


class WrappedModel(ModelWrapper):
    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        # Note: logit's precision is important.
        return torch.softmax(self.model(x_t=x, time=t).float(), -1)


# Unconditional generation
def generate_samples(
    model: nn.Module,
    step: int,
    vocab_size: int,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    device: torch.device,
    path: ProbPath,
    source_distribution: SourceDistribution,
    sample_batch_size: int,
    sequence_length: int,
    sampling_steps: int,
    time_epsilon: float = 0.0,
    sample_dir: Optional[Path] = None,
    dtype_categorical: torch.dtype = torch.float64,
) -> Tensor:
    wrapped_probability_denoiser = WrappedModel(model=model)

    add_token = 1 if source_distribution.masked else 0
    solver = MixtureDiscreteEulerSolver(
        model=wrapped_probability_denoiser,
        path=path,
        vocabulary_size=vocab_size + add_token,
    )

    x_init = source_distribution.sample(
        tensor_size=(sample_batch_size, sequence_length), device=device
    )

    sample = solver.sample(
        x_init=x_init,
        step_size=1 / sampling_steps,
        verbose=True,
        dtype_categorical=dtype_categorical,
        time_grid=torch.tensor([0.0, 1.0 - time_epsilon]),
    )
 
    sentences = tokenizer.batch_decode(sample)

    if sample_dir is not None:
        file_name = sample_dir / f"iter_{step}" / f"sample_{rank}.txt"
        file_name.parents[0].mkdir(exist_ok=True, parents=True)

        with open(file_name, "w") as file:
            for sentence in sentences:
                file.write(f"{sentence}\n{'=' * 20} New sample {'=' * 20}\n")

    return sample


def generate_samples_with_context(
    model: nn.Module,
    step: int,
    vocab_size: int,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    device: torch.device,
    path: ProbPath,
    source_distribution: SourceDistribution,
    sequence_length: int,
    sampling_steps: int,
    time_epsilon: float = 0.0,
    sample_dir: Optional[Path] = None,
    dtype_categorical: torch.dtype = torch.float64,
    input_ids: Tensor = None,
    input_text: str = None,
) -> Tensor:

    if input_ids is None:
        if input_text is None:
            raise ValueError("Must specify 'input_ids' or 'input_text'")
        input_ids = tokenizer(input_text, return_tensors="pt")['input_ids']
    
    input_ids = input_ids.to(device)
    prefix_length = input_ids.shape[1]

    if prefix_length > sequence_length:
        input_ids = input_ids[:, :sequence_length]
        prefix_length = sequence_length

    wrapped_probability_denoiser = WrappedModel(model=model)
    add_token = 1 if source_distribution.masked else 0
    solver = MixtureDiscreteEulerSolver(
        model=wrapped_probability_denoiser,
        path=path,
        vocabulary_size=vocab_size + add_token,
    )

    sample_batch_size = 1
    x_init = source_distribution.sample(
        tensor_size=(sample_batch_size, sequence_length), device=device
    )
    
    x_init[:, :prefix_length] = input_ids

    if rank == 0:
        print(f"Sampling with simplified context injection (Prefix len: {prefix_length})...")

    sample = solver.sample(
        x_init=x_init,
        step_size=1 / sampling_steps,
        verbose=True,
        dtype_categorical=dtype_categorical,
        time_grid=torch.tensor([0.0, 1.0 - time_epsilon]),
    )

    sentences = tokenizer.batch_decode(sample)

    if sample_dir is not None:
        file_name = sample_dir / f"iter_{step}" / f"context_sample_{rank}.txt"
        file_name.parents[0].mkdir(exist_ok=True, parents=True)

        with open(file_name, "w") as file:
            for sentence in sentences:
                file.write(f"[Context]: {input_text}\n")
                file.write(f"[Result]: {sentence}\n")
                file.write(f"{'=' * 20} New sample {'=' * 20}\n")

    return sample