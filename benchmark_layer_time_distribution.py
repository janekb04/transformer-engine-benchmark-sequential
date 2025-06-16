# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
from transformer_engine.common.recipe import Format, DelayedScaling
from sequential_te_layer import SequentialTETransformerLayer
from utils import speedometer, pin_process_to_cpu

HIDDEN_SIZE = 768
SEQUENCE_LENGTH = 512
BATCH_SIZE = 4
FFN_HIDDEN_SIZE = 2048
NUM_ATTENTION_HEADS = 12
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")

sequential_te_transformer_layer = SequentialTETransformerLayer(
    HIDDEN_SIZE,
    FFN_HIDDEN_SIZE,
    NUM_ATTENTION_HEADS    
)
sequential_te_transformer_layer.to(dtype=DTYPE).cuda()

# Synthetic data
x = torch.rand(SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE).cuda().to(dtype=DTYPE)
dy = torch.rand(SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE).cuda().to(dtype=DTYPE)

# Setup fp8 autocast kwargs
fp8_format = Format.HYBRID
fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
fp8_autocast_kwargs = { "enabled": True, "fp8_recipe": fp8_recipe }

pin_process_to_cpu()

speedometer(
    sequential_te_transformer_layer,
    x,
    dy,
    { "attention_mask": None },
    fp8_autocast_kwargs,
    10000000,
    0,
    100,
    True
)