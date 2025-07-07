# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math
from typing import Optional
import torch
import transformer_engine.pytorch as te
import nvtx
import gc

def speedometer(
    layer: torch.nn.Module,
    input: torch.Tensor,
    output_grad: torch.Tensor,
    forward_kwargs: dict,
    fp8_autocast_kwargs: dict | None,
    timing_iters: int,
    warmup_iters: int,
    repeats_per_iter: int,
    print_time_per_layer: bool = False,
):
    """Measure average run time for a PyTorch module

    Performs forward and backward passes.
    """
    if fp8_autocast_kwargs is None:
        fp8_autocast_kwargs = {"enabled": False}

    reenable_gc = gc.isenabled()
    gc.disable()

    def _benchmark(iters: int, print_time_per_layer: bool):
        forward_times: list[float] = []
        backward_times: list[float] = []
        for _ in range(iters):
            start_forward = torch.cuda.Event(enable_timing=True)
            start_backward = torch.cuda.Event(enable_timing=True)
            end_backward = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_forward.record(torch.cuda.default_stream())

            with te.fp8_autocast(**fp8_autocast_kwargs):
                output = layer(input, **forward_kwargs)
                for _ in range(1, repeats_per_iter):
                    output = layer(output, **forward_kwargs)

            start_backward.record(torch.cuda.default_stream())
            output.backward(output_grad)

            end_backward.record(torch.cuda.default_stream())
            torch.cuda.synchronize()

            forward_time = start_forward.elapsed_time(start_backward)  # in ms
            time_per_layer = forward_time / repeats_per_iter
            forward_times.append(time_per_layer)

            backward_time = start_backward.elapsed_time(end_backward)
            time_per_layer = backward_time / repeats_per_iter
            backward_times.append(time_per_layer)

            if print_time_per_layer:
                print(f"{forward_times[-1]}/{backward_times[-1]}", flush=True)

        return forward_times, backward_times

    # Warmup runs
    with nvtx.annotate("warmup"):
        _benchmark(warmup_iters, False)

    # Timing runs
    with nvtx.annotate("timing"):
        forward_times, backward_times = _benchmark(timing_iters, print_time_per_layer)

    forward_times_tensor = torch.tensor(forward_times)
    forward_mean = forward_times_tensor.mean().item()

    backward_times_tensor = torch.tensor(backward_times)
    backward_mean = backward_times_tensor.mean().item()

    if reenable_gc:
        gc.collect()
        gc.enable()

    return forward_mean, backward_mean

class DotProductAttention(torch.nn.Module):
    """Attention operation in Transformer layer

    Built with plain PyTorch modules.

    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.projection_size = kv_channels * num_attention_heads
        self.hidden_size_per_attention_head = kv_channels
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.dropout = torch.nn.Dropout(attention_dropout)

    def masked_softmax(self, inp: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            inp.masked_fill_(mask, -10000.0)
        return torch.nn.Softmax(dim=-1)(inp)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = query.size(1)
        np = query.size(2)
        sq = query.size(0)
        sk = key.size(0)
        hn = value.size(3)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query = query.view(sq, b * np, -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(sk, b * np, -1)

        bmm1 = (
            torch.bmm(query.transpose(0, 1), key.transpose(0, 1).transpose(1, 2)) / self.norm_factor
        )

        # change view to [b, np, sq, sk]
        attention_scores = bmm1.view(b, np, sq, sk)

        attention_probs = self.masked_softmax(attention_scores, attention_mask)

        attention_probs = self.dropout(attention_probs)

        # change view [sk, b * np, hn]
        value = value.view(sk, b * np, -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(b * np, sq, -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(b, np, sq, hn)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        context = context.view(sq, b, self.projection_size)

        return context
