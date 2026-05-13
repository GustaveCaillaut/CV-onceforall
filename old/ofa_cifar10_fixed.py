"""
Mini-OFA for CIFAR-10
=====================

This file implements a compact, research-friendly version of the Once-for-All
(OFA) pipeline for CIFAR-10:

1. Train a weight-sharing supernet with elastic resolution, depth, kernel size,
   and expansion ratio.
2. Use progressive shrinking schedules similar in spirit to OFA:
      largest -> elastic kernel -> elastic depth -> elastic expansion.
3. Sample sub-networks from the trained supernet and build an accuracy dataset.
4. Train an MLP accuracy predictor on architecture encodings.
5. Run evolutionary search under a MAC constraint.
6. Save checkpoints, best models, sampled architecture datasets, predictors,
   and search results.

This is intentionally simpler than the original ImageNet OFA implementation.
It is designed to be easy to modify, especially the sampling strategies.

Typical usage
-------------

Train the supernet progressively:

    python ofa_cifar10.py train_supernet \
        --data-dir ./data \
        --save-dir ./runs/ofa_cifar10 \
        --epochs-largest 50 \
        --epochs-kernel 25 \
        --epochs-depth 25 \
        --epochs-width 25

Build an accuracy dataset with, for example, 500 sampled subnets:

    python ofa_cifar10.py sample_acc_dataset \
        --data-dir ./data \
        --save-dir ./runs/ofa_cifar10 \
        --checkpoint ./runs/ofa_cifar10/checkpoints/supernet_best.pt \
        --num-arch-samples 500 \
        --output ./runs/ofa_cifar10/acc_dataset_500.jsonl

Train the predictor:

    python ofa_cifar10.py train_acc_predictor \
        --dataset ./runs/ofa_cifar10/acc_dataset_500.jsonl \
        --save-dir ./runs/ofa_cifar10

Search under a MAC budget:

    python ofa_cifar10.py search \
        --predictor ./runs/ofa_cifar10/accuracy_predictor.pt \
        --save-dir ./runs/ofa_cifar10 \
        --macs-limit 80e6

Evaluate a selected architecture:

    python ofa_cifar10.py eval_subnet \
        --data-dir ./data \
        --checkpoint ./runs/ofa_cifar10/checkpoints/supernet_best.pt \
        --spec-json ./runs/ofa_cifar10/search/best_spec.json \
        --bn-calib-batches 100

Notes
-----

- The original OFA paper uses ImageNet, MobileNetV3-like blocks, hardware latency
  lookup tables, and a large-scale 16K architecture accuracy set. This script is
  a faithful didactic version for CIFAR-10, not a byte-for-byte reproduction.
- Here, the default search constraint is MACs, because it is portable. You can
  replace `estimate_macs_given_spec` or add a latency lookup table later.
- The variable `--num-arch-samples` controls the analogue of the "16K" sampling
  set. Try 100, 500, 2000, etc. and compare predictor RMSE / final searched model.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------------------------------------------------------------
# Reproducibility and filesystem helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set Python and PyTorch random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can slow things down; keep cuDNN benchmark off for
    # more stable runs across different active subnet shapes.
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_string() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


# -----------------------------------------------------------------------------
# Search space specification
# -----------------------------------------------------------------------------


@dataclass
class SearchSpace:
    """All discrete choices used by the CIFAR-10 OFA supernet.

    The original OFA ImageNet search space has 5 stages x 4 blocks = 20 elastic
    blocks. We keep that structure because it makes the code close to the paper.

    - r: input resolution choices.
    - d: stage depth choices. If d_s=2 in stage s, we keep the first 2 blocks of
      that stage and skip blocks 3 and 4. This creates the nested inclusion rule.
    - ks: kernel-size choices for each elastic block.
    - e: expansion-ratio choices. This is the width-like degree of freedom in
      this compact implementation: larger e means more middle channels in MBConv.
    """

    resolutions: Tuple[int, ...] = (24, 28, 32)
    depth_choices: Tuple[int, ...] = (2, 3, 4)
    kernel_choices: Tuple[int, ...] = (3, 5, 7)
    expand_choices: Tuple[int, ...] = (3, 4, 6)
    n_stages: int = 5
    blocks_per_stage: int = 4

    @property
    def n_blocks(self) -> int:
        return self.n_stages * self.blocks_per_stage

    @property
    def max_resolution(self) -> int:
        return max(self.resolutions)

    @property
    def max_depth(self) -> int:
        return max(self.depth_choices)

    @property
    def max_kernel(self) -> int:
        return max(self.kernel_choices)

    @property
    def max_expand(self) -> int:
        return max(self.expand_choices)

    def largest_spec(self) -> Dict:
        """Return the largest subnetwork specification."""
        return {
            "r": self.max_resolution,
            "d": [self.max_depth] * self.n_stages,
            "ks": [self.max_kernel] * self.n_blocks,
            "e": [self.max_expand] * self.n_blocks,
        }

    def random_spec(
        self,
        allowed_resolutions: Optional[Sequence[int]] = None,
        allowed_depths: Optional[Sequence[int]] = None,
        allowed_kernels: Optional[Sequence[int]] = None,
        allowed_expands: Optional[Sequence[int]] = None,
    ) -> Dict:
        """Sample a random architecture tuple.

        This is the main function to edit if you want to experiment with sampling.
        For example, you can implement stratified sampling, biased sampling toward
        small models, active learning based on predictor uncertainty, etc.
        """
        rs = tuple(allowed_resolutions or self.resolutions)
        ds = tuple(allowed_depths or self.depth_choices)
        ks = tuple(allowed_kernels or self.kernel_choices)
        es = tuple(allowed_expands or self.expand_choices)
        return {
            "r": random.choice(rs),
            "d": [random.choice(ds) for _ in range(self.n_stages)],
            "ks": [random.choice(ks) for _ in range(self.n_blocks)],
            "e": [random.choice(es) for _ in range(self.n_blocks)],
        }

    def sample_for_training_stage(self, stage: str) -> Dict:
        """Sample a subnet according to a progressive shrinking training stage.

        Stages are deliberately simple and close to OFA Appendix B:
        - largest: only largest model.
        - kernel: elastic kernel, max depth and max expansion.
        - depth1: depths in {3,4}, elastic kernel, max expansion.
        - depth2: depths in {2,3,4}, elastic kernel, max expansion.
        - width1: expansions in {4,6}, elastic kernel and depth.
        - width2/full: expansions in {3,4,6}, fully elastic.
        """
        if stage == "largest":
            return self.largest_spec()
        if stage == "kernel":
            return self.random_spec(
                allowed_depths=(self.max_depth,),
                allowed_expands=(self.max_expand,),
            )
        if stage == "depth1":
            return self.random_spec(
                allowed_depths=(3, 4),
                allowed_expands=(self.max_expand,),
            )
        if stage == "depth2":
            return self.random_spec(
                allowed_depths=(2, 3, 4),
                allowed_expands=(self.max_expand,),
            )
        if stage == "width1":
            return self.random_spec(allowed_expands=(4, 6))
        if stage in {"width2", "full"}:
            return self.random_spec()
        raise ValueError(f"Unknown progressive stage: {stage}")


# -----------------------------------------------------------------------------
# Dynamic layers
# -----------------------------------------------------------------------------


class DynamicBatchNorm2d(nn.Module):
    """BatchNorm2d whose number of active channels is inferred from the input.

    We store statistics for the maximum channel count and slice the prefix for a
    smaller subnet. This is a common trick in slimmable / OFA-style networks.
    """

    def __init__(self, max_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.max_features = max_features
        self.bn = nn.BatchNorm2d(max_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(1)
        return F.batch_norm(
            x,
            self.bn.running_mean[:c],
            self.bn.running_var[:c],
            self.bn.weight[:c],
            self.bn.bias[:c],
            self.bn.training,
            self.bn.momentum,
            self.bn.eps,
        )

    def reset_running_stats(self) -> None:
        self.bn.reset_running_stats()


class ConvBNAct(nn.Module):
    """Static Conv-BN-Activation block used for fixed layers."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, act: str = "relu"):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = build_activation(act)
        self.out_channels = out_ch
        self.in_channels = in_ch
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DynamicPointConv2d(nn.Module):
    """1x1 convolution with dynamic input and output channels via prefix slicing."""

    def __init__(self, max_in_ch: int, max_out_ch: int, bias: bool = False):
        super().__init__()
        self.max_in_ch = max_in_ch
        self.max_out_ch = max_out_ch
        self.weight = nn.Parameter(torch.empty(max_out_ch, max_in_ch, 1, 1))
        self.bias = nn.Parameter(torch.zeros(max_out_ch)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_out")

    def forward(self, x: torch.Tensor, out_ch: int) -> torch.Tensor:
        in_ch = x.size(1)
        weight = self.weight[:out_ch, :in_ch, :, :]
        bias = self.bias[:out_ch] if self.bias is not None else None
        return F.conv2d(x, weight, bias=bias, stride=1, padding=0)


class DynamicDepthwiseConv2d(nn.Module):
    """Depthwise convolution with dynamic channels and kernel size.

    Kernel nesting follows OFA's rule: a 3x3 kernel is the center crop of a 5x5,
    which is the center crop of a 7x7. This gives deterministic inclusion.
    """

    def __init__(self, max_channels: int, max_kernel_size: int = 7):
        super().__init__()
        self.max_channels = max_channels
        self.max_kernel_size = max_kernel_size
        self.weight = nn.Parameter(torch.empty(max_channels, 1, max_kernel_size, max_kernel_size))
        nn.init.kaiming_normal_(self.weight, mode="fan_out")

    def forward(self, x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
        c = x.size(1)
        start = (self.max_kernel_size - kernel_size) // 2
        end = start + kernel_size
        weight = self.weight[:c, :, start:end, start:end]
        padding = kernel_size // 2
        return F.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=c)


class DynamicMBConv(nn.Module):
    """Dynamic Mobile Inverted Bottleneck block.

    In this compact implementation, output channels are fixed by the stage, while
    the expansion ratio controls the middle-channel count. That middle-channel
    dimension is the elastic-width analogue.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        max_expand: int = 6,
        max_kernel_size: int = 7,
        act: str = "relu",
    ):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.stride = stride
        self.max_expand = max_expand
        self.max_kernel_size = max_kernel_size
        self.max_mid_ch = int(round(in_ch * max_expand))

        self.expand_conv = DynamicPointConv2d(in_ch, self.max_mid_ch, bias=False)
        self.expand_bn = DynamicBatchNorm2d(self.max_mid_ch)
        self.depth_conv = DynamicDepthwiseConv2d(self.max_mid_ch, max_kernel_size)
        self.depth_bn = DynamicBatchNorm2d(self.max_mid_ch)
        self.project_conv = DynamicPointConv2d(self.max_mid_ch, out_ch, bias=False)
        self.project_bn = DynamicBatchNorm2d(out_ch)
        self.act = build_activation(act)

    def forward(self, x: torch.Tensor, kernel_size: int, expand_ratio: int) -> torch.Tensor:
        mid_ch = int(round(self.in_channels * expand_ratio))
        residual = x

        out = self.expand_conv(x, out_ch=mid_ch)
        out = self.act(self.expand_bn(out))
        out = self.depth_conv(out, kernel_size=kernel_size, stride=self.stride)
        out = self.act(self.depth_bn(out))
        out = self.project_conv(out, out_ch=self.out_channels)
        out = self.project_bn(out)

        if self.stride == 1 and self.in_channels == self.out_channels:
            out = out + residual
        return out

    @torch.no_grad()
    def sort_middle_channels_by_l1(self) -> None:
        """Optional OFA-like channel sorting for the expansion dimension.

        Original OFA sorts channels by importance before shrinking width, so that
        smaller subnets use the most important prefix channels. Here we sort the
        middle channels according to an L1 score from the expansion conv output
        weights. This is a simplified but useful version of the idea.

        Important: channel reordering must be applied consistently to the expand,
        depthwise, and project layers. This method does that for the middle axis.
        """
        importance = self.expand_conv.weight.abs().sum(dim=(1, 2, 3))
        order = torch.argsort(importance, descending=True)

        # Reorder expansion output channels.
        self.expand_conv.weight.data = self.expand_conv.weight.data[order, :, :, :]
        self.expand_bn.bn.weight.data = self.expand_bn.bn.weight.data[order]
        self.expand_bn.bn.bias.data = self.expand_bn.bn.bias.data[order]
        self.expand_bn.bn.running_mean.data = self.expand_bn.bn.running_mean.data[order]
        self.expand_bn.bn.running_var.data = self.expand_bn.bn.running_var.data[order]

        # Reorder depthwise channels.
        self.depth_conv.weight.data = self.depth_conv.weight.data[order, :, :, :]
        self.depth_bn.bn.weight.data = self.depth_bn.bn.weight.data[order]
        self.depth_bn.bn.bias.data = self.depth_bn.bn.bias.data[order]
        self.depth_bn.bn.running_mean.data = self.depth_bn.bn.running_mean.data[order]
        self.depth_bn.bn.running_var.data = self.depth_bn.bn.running_var.data[order]

        # Reorder project input channels.
        self.project_conv.weight.data = self.project_conv.weight.data[:, order, :, :]



def build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "h_swish":
        return nn.Hardswish(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


# -----------------------------------------------------------------------------
# OFA supernet for CIFAR-10
# -----------------------------------------------------------------------------


class OFACifarNet(nn.Module):
    """A small OFA-style supernet for CIFAR-10.

    Architecture layout:
      stem conv -> fixed MBConv -> 5 stages x 4 elastic MBConv blocks
      -> final 1x1 conv -> global average pooling -> classifier

    A subnet is fully determined by a spec dict:
      {"r": int, "d": List[int x5], "ks": List[int x20], "e": List[int x20]}

    The deterministic extraction rules are:
      - depth d_s: keep the first d_s blocks in stage s;
      - kernel k: use center crop of max 7x7 kernel;
      - expansion e: use first round(in_channels*e) middle channels;
      - resolution r: resize the input to r x r.
    """

    def __init__(self, num_classes: int = 10, search_space: Optional[SearchSpace] = None):
        super().__init__()
        self.search_space = search_space or SearchSpace()
        self.num_classes = num_classes

        # CIFAR-10 images are small, so the stem uses stride 1.
        self.stem_out = 24
        self.first_conv = ConvBNAct(3, self.stem_out, kernel_size=3, stride=1, act="relu")

        # One fixed first block, like MobileNet-style networks often have.
        self.first_block = DynamicMBConv(
            in_ch=self.stem_out,
            out_ch=self.stem_out,
            stride=1,
            max_expand=1,
            max_kernel_size=3,
            act="relu",
        )

        # Stage configuration for CIFAR-10. Output channels are fixed; width
        # elasticity comes from the expansion ratio inside each MBConv.
        self.stage_out_channels = [32, 48, 64, 96, 128]
        self.stage_strides = [1, 2, 2, 2, 1]
        self.stage_acts = ["relu", "relu", "relu", "h_swish", "h_swish"]

        blocks = []
        in_ch = self.stem_out
        for s in range(self.search_space.n_stages):
            out_ch = self.stage_out_channels[s]
            for b in range(self.search_space.blocks_per_stage):
                stride = self.stage_strides[s] if b == 0 else 1
                block = DynamicMBConv(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    stride=stride,
                    max_expand=self.search_space.max_expand,
                    max_kernel_size=self.search_space.max_kernel,
                    act=self.stage_acts[s],
                )
                blocks.append(block)
                in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        self.final_expand_in = self.stage_out_channels[-1]
        self.final_expand_out = 256
        self.final_expand_layer = ConvBNAct(self.final_expand_in, self.final_expand_out, 1, 1, act="h_swish")
        self.classifier = nn.Linear(self.final_expand_out, num_classes)

        self.active_spec = self.search_space.largest_spec()

    def set_active_subnet(self, spec: Dict) -> None:
        """Set the currently active subnet spec."""
        validate_spec(spec, self.search_space)
        # Deep-copy so later external modifications do not silently affect the model.
        self.active_spec = copy.deepcopy(spec)

    def sample_active_subnet(self, stage: str = "full") -> Dict:
        """Sample and activate a subnet for a progressive training stage."""
        spec = self.search_space.sample_for_training_stage(stage)
        self.set_active_subnet(spec)
        return spec

    def set_largest_subnet(self) -> None:
        self.set_active_subnet(self.search_space.largest_spec())

    def forward(self, x: torch.Tensor, spec: Optional[Dict] = None) -> torch.Tensor:
        if spec is not None:
            self.set_active_subnet(spec)
        spec = self.active_spec

        # Elastic resolution: resize input deterministically for the active subnet.
        r = int(spec["r"])
        if x.size(-1) != r or x.size(-2) != r:
            x = F.interpolate(x, size=(r, r), mode="bilinear", align_corners=False)

        out = self.first_conv(x)
        out = self.first_block(out, kernel_size=3, expand_ratio=1)

        idx = 0
        for s in range(self.search_space.n_stages):
            active_depth = int(spec["d"][s])
            for b in range(self.search_space.blocks_per_stage):
                block = self.blocks[idx]
                if b < active_depth:
                    out = block(
                        out,
                        kernel_size=int(spec["ks"][idx]),
                        expand_ratio=int(spec["e"][idx]),
                    )
                # If b >= active_depth, the block is skipped. Notice that this is
                # only allowed for trailing blocks, preserving nested inclusion.
                idx += 1

        out = self.final_expand_layer(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        logits = self.classifier(out)
        return logits

    @torch.no_grad()
    def sort_all_middle_channels_by_l1(self) -> None:
        """Apply channel sorting to all dynamic MBConv blocks."""
        self.first_block.sort_middle_channels_by_l1()
        for block in self.blocks:
            block.sort_middle_channels_by_l1()


# -----------------------------------------------------------------------------
# Spec validation, encoding, and MAC estimation
# -----------------------------------------------------------------------------


def validate_spec(spec: Dict, ss: SearchSpace) -> None:
    """Fail early if a spec is malformed."""
    if int(spec["r"]) not in ss.resolutions:
        raise ValueError(f"Invalid resolution: {spec['r']}")
    if len(spec["d"]) != ss.n_stages:
        raise ValueError("spec['d'] must have length n_stages")
    if len(spec["ks"]) != ss.n_blocks:
        raise ValueError("spec['ks'] must have length n_blocks")
    if len(spec["e"]) != ss.n_blocks:
        raise ValueError("spec['e'] must have length n_blocks")
    for d in spec["d"]:
        if int(d) not in ss.depth_choices:
            raise ValueError(f"Invalid depth: {d}")
    for k in spec["ks"]:
        if int(k) not in ss.kernel_choices:
            raise ValueError(f"Invalid kernel: {k}")
    for e in spec["e"]:
        if int(e) not in ss.expand_choices:
            raise ValueError(f"Invalid expand ratio: {e}")


def spec_to_jsonable(spec: Dict) -> Dict:
    return {
        "r": int(spec["r"]),
        "d": [int(x) for x in spec["d"]],
        "ks": [int(x) for x in spec["ks"]],
        "e": [int(x) for x in spec["e"]],
    }


def encode_spec(spec: Dict, ss: SearchSpace) -> torch.Tensor:
    """One-hot encode a subnet architecture for the accuracy predictor.

    This mirrors the OFA appendix idea: one-hot vectors for per-layer kernel and
    expansion; skipped layers are encoded with all zeros. We also encode stage
    depths and input resolution.
    """
    validate_spec(spec, ss)
    pieces: List[float] = []

    # Resolution one-hot.
    for r in ss.resolutions:
        pieces.append(1.0 if int(spec["r"]) == r else 0.0)

    # Stage depth one-hot.
    for d in spec["d"]:
        for choice in ss.depth_choices:
            pieces.append(1.0 if int(d) == choice else 0.0)

    # Per-block kernel and expansion one-hot. If the block is skipped by depth,
    # we add zeros for both, exactly to tell the predictor "this layer is absent".
    for i in range(ss.n_blocks):
        stage = i // ss.blocks_per_stage
        within = i % ss.blocks_per_stage
        active = within < int(spec["d"][stage])
        for choice in ss.kernel_choices:
            pieces.append(1.0 if active and int(spec["ks"][i]) == choice else 0.0)
        for choice in ss.expand_choices:
            pieces.append(1.0 if active and int(spec["e"][i]) == choice else 0.0)

    return torch.tensor(pieces, dtype=torch.float32)


def encoding_dim(ss: SearchSpace) -> int:
    return (
        len(ss.resolutions)
        + ss.n_stages * len(ss.depth_choices)
        + ss.n_blocks * (len(ss.kernel_choices) + len(ss.expand_choices))
    )


def conv2d_macs(h: int, w: int, cin: int, cout: int, k: int, groups: int = 1) -> int:
    """Multiply-accumulate count for a convolution output of spatial size h x w."""
    return h * w * cout * (cin // groups) * k * k


def mbconv_macs(h: int, w: int, cin: int, cout: int, stride: int, k: int, e: int) -> Tuple[int, int, int]:
    """Return (macs, out_h, out_w) for an MBConv-like block."""
    mid = int(round(cin * e))
    out_h = int((h - 1) / stride + 1)
    out_w = int((w - 1) / stride + 1)
    macs = 0
    # expansion 1x1 at input resolution
    macs += conv2d_macs(h, w, cin, mid, 1, groups=1)
    # depthwise kxk at output resolution
    macs += conv2d_macs(out_h, out_w, mid, mid, k, groups=mid)
    # projection 1x1 at output resolution
    macs += conv2d_macs(out_h, out_w, mid, cout, 1, groups=1)
    return macs, out_h, out_w


def estimate_macs_given_spec(spec: Dict, ss: Optional[SearchSpace] = None) -> int:
    """Analytic MAC estimate for OFACifarNet under a spec.

    This replaces the hardware latency LUT for a simple CIFAR-10 project. Later,
    you can implement a latency lookup table with the same interface.
    """
    ss = ss or SearchSpace()
    validate_spec(spec, ss)

    r = int(spec["r"])
    h = w = r
    macs = 0

    # Stem conv: 3 -> 24, stride 1.
    macs += conv2d_macs(h, w, 3, 24, 3)

    # Fixed first MBConv: 24 -> 24, k=3, e=1.
    first_macs, h, w = mbconv_macs(h, w, 24, 24, stride=1, k=3, e=1)
    macs += first_macs

    stage_out_channels = [32, 48, 64, 96, 128]
    stage_strides = [1, 2, 2, 2, 1]
    cin = 24
    idx = 0
    for s in range(ss.n_stages):
        cout = stage_out_channels[s]
        for b in range(ss.blocks_per_stage):
            stride = stage_strides[s] if b == 0 else 1
            if b < int(spec["d"][s]):
                block_macs, h, w = mbconv_macs(
                    h,
                    w,
                    cin,
                    cout,
                    stride=stride,
                    k=int(spec["ks"][idx]),
                    e=int(spec["e"][idx]),
                )
                macs += block_macs
                cin = cout
            idx += 1

    # Final 1x1 expand and classifier. Pooling is negligible and ignored.
    macs += conv2d_macs(h, w, 128, 256, 1)
    macs += 256 * 10
    return int(macs)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


def build_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: int = 5000,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, predictor-val, and test loaders.

    CIFAR-10 has no official validation set. We reserve `val_split` images from
    the training set for subnet accuracy estimation / predictor supervision.
    """
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    full_train_aug = datasets.CIFAR10(data_dir, train=True, transform=train_tf, download=True)
    full_train_eval = datasets.CIFAR10(data_dir, train=True, transform=eval_tf, download=True)
    test_set = datasets.CIFAR10(data_dir, train=False, transform=eval_tf, download=True)

    n = len(full_train_aug)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    val_indices = indices[:val_split]
    train_indices = indices[val_split:]

    train_set = Subset(full_train_aug, train_indices)
    val_set = Subset(full_train_eval, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# Training / evaluation utilities
# -----------------------------------------------------------------------------


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item() * 100.0


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Standard KL distillation loss."""
    t = temperature
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits / t, dim=1),
        reduction="batchmean",
    ) * (t * t)


@torch.no_grad()
def get_teacher_logits(model: OFACifarNet, images: torch.Tensor) -> torch.Tensor:
    """Forward the largest subnet in eval mode without updating BN stats.

    During progressive shrinking, `model` should be a frozen teacher copy.
    Using the student itself as teacher creates a moving target and can destroy
    the largest subnet as soon as kernel/depth/width sampling begins.
    """
    was_training = model.training
    old_spec = copy.deepcopy(model.active_spec)
    model.eval()
    model.set_largest_subnet()
    logits = model(images)
    model.set_active_subnet(old_spec)
    if was_training:
        model.train()
    return logits.detach()


def make_frozen_teacher(model: OFACifarNet) -> OFACifarNet:
    """Return a frozen copy of the current largest-subnet model."""
    teacher = copy.deepcopy(model)
    teacher.eval()
    teacher.set_largest_subnet()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def train_one_epoch(
    model: OFACifarNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    stage: str,
    subnets_per_batch: int,
    kd_ratio: float,
    kd_temperature: float,
    log_interval: int = 50,
    teacher_model: Optional[OFACifarNet] = None,
) -> Dict[str, float]:
    """Train one epoch with sampled subnets.

    If subnets_per_batch > 1, we average gradients over multiple sampled subnets,
    matching the OFA idea of aggregating gradients for depth/width stages.
    """
    model.train()
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    kd_meter = AverageMeter()
    acc_meter = AverageMeter()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Teacher is only used after largest-network training. It is a frozen
        # copy of the largest subnet, created at the transition out of the
        # `largest` stage.
        use_kd = kd_ratio > 0 and stage != "largest" and teacher_model is not None
        teacher_logits = get_teacher_logits(teacher_model, images) if use_kd else None

        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_acc = 0.0

        for _ in range(subnets_per_batch):
            spec = model.sample_active_subnet(stage)
            logits = model(images, spec=spec)
            ce = F.cross_entropy(logits, targets)
            if use_kd:
                kd = kd_loss(logits, teacher_logits, temperature=kd_temperature)
                loss = (1.0 - kd_ratio) * ce + kd_ratio * kd
            else:
                kd = torch.zeros_like(ce)
                loss = ce

            # Average the gradient contributions over sampled subnets.
            (loss / subnets_per_batch).backward()
            total_loss += loss.item()
            total_ce += ce.item()
            total_kd += kd.item()
            total_acc += accuracy_top1(logits.detach(), targets)

        optimizer.step()

        bs = images.size(0)
        loss_meter.update(total_loss / subnets_per_batch, bs)
        ce_meter.update(total_ce / subnets_per_batch, bs)
        kd_meter.update(total_kd / subnets_per_batch, bs)
        acc_meter.update(total_acc / subnets_per_batch, bs)

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"epoch={epoch:03d} stage={stage:8s} step={step:04d}/{len(loader)} "
                f"loss={loss_meter.avg:.4f} ce={ce_meter.avg:.4f} "
                f"kd={kd_meter.avg:.4f} acc={acc_meter.avg:.2f}"
            )

    return {"loss": loss_meter.avg, "ce": ce_meter.avg, "kd": kd_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def evaluate(
    model: OFACifarNet,
    loader: DataLoader,
    device: torch.device,
    spec: Optional[Dict] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate one active subnet on a loader."""
    model.eval()
    if spec is not None:
        model.set_active_subnet(spec)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy_top1(logits, targets), bs)
    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def reset_bn_stats(module: nn.Module) -> None:
    """Reset all BatchNorm running statistics."""
    for m in module.modules():
        if isinstance(m, DynamicBatchNorm2d):
            m.reset_running_stats()
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()


@torch.no_grad()
def calibrate_bn(
    model: OFACifarNet,
    loader: DataLoader,
    device: torch.device,
    spec: Dict,
    num_batches: int = 50,
) -> None:
    """Recompute BN statistics for a specific subnet.

    This is important for fair subnet evaluation in weight-sharing networks.
    For speed, use a small number of batches during large architecture sampling.
    """
    reset_bn_stats(model)
    model.train()
    model.set_active_subnet(spec)
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        images = images.to(device, non_blocking=True)
        _ = model(images)
    model.eval()


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    model: OFACifarNet,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_acc: float,
    extra: Optional[Dict] = None,
) -> None:
    ensure_dir(Path(path).parent)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "best_acc": best_acc,
        "search_space": asdict(model.search_space),
        "active_spec": spec_to_jsonable(model.active_spec),
        "extra": extra or {},
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_supernet_checkpoint(path: str | Path, device: torch.device) -> OFACifarNet:
    ckpt = torch.load(path, map_location=device)
    ss_dict = ckpt.get("search_space", {})
    ss = SearchSpace(**ss_dict) if ss_dict else SearchSpace()
    model = OFACifarNet(search_space=ss).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    active_spec = ckpt.get("active_spec", ss.largest_spec())
    model.set_active_subnet(active_spec)
    return model


# -----------------------------------------------------------------------------
# Accuracy predictor
# -----------------------------------------------------------------------------


class AccuracyPredictor(nn.Module):
    """Simple MLP predictor: architecture one-hot -> accuracy percentage."""

    def __init__(self, input_dim: int, hidden_dim: int = 400, n_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ArchAccDataset(torch.utils.data.Dataset):
    """JSONL dataset of {spec, acc, macs}."""

    def __init__(self, jsonl_path: str | Path, ss: Optional[SearchSpace] = None):
        self.ss = ss or SearchSpace()
        self.items = []
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.items.append(item)
        if not self.items:
            raise ValueError(f"Empty architecture dataset: {jsonl_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        x = encode_spec(item["spec"], self.ss)
        y = torch.tensor(float(item["acc"]), dtype=torch.float32)
        return x, y


def train_accuracy_predictor(
    jsonl_path: str | Path,
    save_dir: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> Path:
    """Train and save the MLP accuracy predictor."""
    set_seed(seed)
    ss = SearchSpace()
    dataset = ArchAccDataset(jsonl_path, ss)
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = max(1, int(0.8 * n))
    train_idx, val_idx = indices[:split], indices[split:]
    if not val_idx:
        val_idx = train_idx

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    model = AccuracyPredictor(input_dim=encoding_dim(ss)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_rmse = float("inf")
    out_dir = ensure_dir(save_dir)
    best_path = out_dir / "accuracy_predictor.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = AverageMeter()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss.update(loss.item(), x.size(0))

        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                pred = model(x).cpu()
                preds.append(pred)
                ys.append(y)
        preds_t = torch.cat(preds)
        ys_t = torch.cat(ys)
        rmse = torch.sqrt(F.mse_loss(preds_t, ys_t)).item()
        mae = torch.mean(torch.abs(preds_t - ys_t)).item()
        print(f"predictor epoch={epoch:03d} train_mse={train_loss.avg:.4f} val_rmse={rmse:.4f} val_mae={mae:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": encoding_dim(ss),
                    "search_space": asdict(ss),
                    "best_rmse": best_rmse,
                    "source_dataset": str(jsonl_path),
                },
                best_path,
            )

    print(f"Saved best accuracy predictor to {best_path} with RMSE={best_rmse:.4f}")
    return best_path


def load_accuracy_predictor(path: str | Path, device: torch.device) -> Tuple[AccuracyPredictor, SearchSpace]:
    ckpt = torch.load(path, map_location=device)
    ss = SearchSpace(**ckpt.get("search_space", {}))
    model = AccuracyPredictor(input_dim=ckpt["input_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ss


@torch.no_grad()
def predict_accuracy(model: AccuracyPredictor, spec: Dict, ss: SearchSpace, device: torch.device) -> float:
    x = encode_spec(spec, ss).unsqueeze(0).to(device)
    return float(model(x).item())


# -----------------------------------------------------------------------------
# Architecture sampling for predictor training
# -----------------------------------------------------------------------------


def sample_accuracy_dataset(
    model: OFACifarNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_arch_samples: int,
    output_path: str | Path,
    seed: int,
    bn_calib_batches: int,
    val_batches: Optional[int],
    sampling_strategy: str = "uniform",
) -> None:
    """Sample architectures and estimate their validation accuracy.

    This is the analogue of the OFA paper's 16K sampled subnets. Here you can
    vary `num_arch_samples` to answer the question: does using more sampled
    architectures improve the final predictor and search result?

    Current sampling strategies:
      - uniform: uniform over r, d, ks, e.
      - small_biased: 70% small-ish models, 30% uniform.
      - large_biased: 70% large-ish models, 30% uniform.
    """
    set_seed(seed)
    ss = model.search_space
    ensure_dir(Path(output_path).parent)

    def draw_spec() -> Dict:
        if sampling_strategy == "uniform":
            return ss.random_spec()
        if sampling_strategy == "small_biased":
            if random.random() < 0.7:
                return ss.random_spec(
                    allowed_resolutions=ss.resolutions[:2],
                    allowed_depths=(2, 3),
                    allowed_kernels=(3, 5),
                    allowed_expands=(3, 4),
                )
            return ss.random_spec()
        if sampling_strategy == "large_biased":
            if random.random() < 0.7:
                return ss.random_spec(
                    allowed_resolutions=ss.resolutions[1:],
                    allowed_depths=(3, 4),
                    allowed_kernels=(5, 7),
                    allowed_expands=(4, 6),
                )
            return ss.random_spec()
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    seen = set()
    with open(output_path, "w") as f:
        for i in range(num_arch_samples):
            # Avoid duplicate specs, which can happen if num_arch_samples is large.
            for _ in range(100):
                spec = spec_to_jsonable(draw_spec())
                key = json.dumps(spec, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    break
            macs = estimate_macs_given_spec(spec, ss)

            # BN recalibration is expensive but gives much more meaningful subnet
            # accuracy estimates.
            if bn_calib_batches > 0:
                calibrate_bn(model, train_loader, device, spec, num_batches=bn_calib_batches)
            metrics = evaluate(model, val_loader, device, spec=spec, max_batches=val_batches)
            row = {"spec": spec, "acc": metrics["acc"], "loss": metrics["loss"], "macs": macs}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(
                f"sample {i + 1:05d}/{num_arch_samples:05d} "
                f"acc={metrics['acc']:.2f} macs={macs/1e6:.2f}M spec={spec}"
            )

    print(f"Saved architecture accuracy dataset to {output_path}")


# -----------------------------------------------------------------------------
# Evolutionary search
# -----------------------------------------------------------------------------


def mutate_spec(spec: Dict, ss: SearchSpace, mutation_prob: float = 0.1) -> Dict:
    child = spec_to_jsonable(spec)
    if random.random() < mutation_prob:
        child["r"] = random.choice(ss.resolutions)
    for s in range(ss.n_stages):
        if random.random() < mutation_prob:
            child["d"][s] = random.choice(ss.depth_choices)
    for i in range(ss.n_blocks):
        if random.random() < mutation_prob:
            child["ks"][i] = random.choice(ss.kernel_choices)
        if random.random() < mutation_prob:
            child["e"][i] = random.choice(ss.expand_choices)
    return child


def crossover_spec(parent_a: Dict, parent_b: Dict, ss: SearchSpace) -> Dict:
    child = {
        "r": random.choice([parent_a["r"], parent_b["r"]]),
        "d": [],
        "ks": [],
        "e": [],
    }
    for s in range(ss.n_stages):
        child["d"].append(random.choice([parent_a["d"][s], parent_b["d"][s]]))
    for i in range(ss.n_blocks):
        child["ks"].append(random.choice([parent_a["ks"][i], parent_b["ks"][i]]))
        child["e"].append(random.choice([parent_a["e"][i], parent_b["e"][i]]))
    return child


def evolutionary_search(
    predictor: AccuracyPredictor,
    ss: SearchSpace,
    device: torch.device,
    macs_limit: float,
    population_size: int,
    parent_size: int,
    generations: int,
    mutation_prob: float,
    seed: int,
    save_dir: str | Path,
) -> Dict:
    """Search for high predicted accuracy subject to a MAC constraint."""
    set_seed(seed)
    save_dir = ensure_dir(save_dir)

    def score(spec: Dict) -> Tuple[float, int]:
        macs = estimate_macs_given_spec(spec, ss)
        if macs > macs_limit:
            return -1e9 - (macs - macs_limit) / 1e6, macs
        acc = predict_accuracy(predictor, spec, ss, device)
        return acc, macs

    # Initial feasible population.
    population = []
    attempts = 0
    while len(population) < population_size and attempts < population_size * 100:
        attempts += 1
        spec = spec_to_jsonable(ss.random_spec())
        acc, macs = score(spec)
        if macs <= macs_limit:
            population.append({"spec": spec, "pred_acc": acc, "macs": macs})
    if len(population) == 0:
        raise RuntimeError("No feasible architecture found. Increase --macs-limit.")

    best = max(population, key=lambda x: x["pred_acc"])
    history = []

    for gen in range(generations):
        population = sorted(population, key=lambda x: x["pred_acc"], reverse=True)
        parents = population[:parent_size]
        if parents[0]["pred_acc"] > best["pred_acc"]:
            best = copy.deepcopy(parents[0])

        print(
            f"gen={gen:03d} best_pred_acc={best['pred_acc']:.3f} "
            f"best_macs={best['macs']/1e6:.2f}M spec={best['spec']}"
        )
        history.append({"generation": gen, "best": copy.deepcopy(best)})

        children = []
        seen = {json.dumps(p["spec"], sort_keys=True) for p in parents}
        while len(children) + len(parents) < population_size:
            if random.random() < 0.5 and len(parents) >= 2:
                pa, pb = random.sample(parents, 2)
                child_spec = crossover_spec(pa["spec"], pb["spec"], ss)
            else:
                pa = random.choice(parents)
                child_spec = mutate_spec(pa["spec"], ss, mutation_prob=mutation_prob)
            child_spec = spec_to_jsonable(child_spec)
            key = json.dumps(child_spec, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            pred_acc, macs = score(child_spec)
            if macs <= macs_limit:
                children.append({"spec": child_spec, "pred_acc": pred_acc, "macs": macs})

        population = parents + children

    best_path = save_dir / "best_spec.json"
    hist_path = save_dir / "search_history.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved best searched architecture to {best_path}")
    return best


# -----------------------------------------------------------------------------
# CLI commands
# -----------------------------------------------------------------------------


def cmd_train_supernet(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if getattr(args, "shrink_lr", None) is not None and args.shrink_lr < 0:
        args.shrink_lr = None
    save_dir = ensure_dir(args.save_dir)
    ckpt_dir = ensure_dir(save_dir / "checkpoints")

    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )
    model = OFACifarNet().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # A simple cosine scheduler over the total number of epochs.
    schedule = []
    schedule += [("largest", args.epochs_largest, 1, 0.0)]
    schedule += [("kernel", args.epochs_kernel, 1, args.kd_ratio)]
    schedule += [("depth1", args.epochs_depth1, 2, args.kd_ratio)]
    schedule += [("depth2", args.epochs_depth2, 2, args.kd_ratio)]
    schedule += [("width1", args.epochs_width1, 4, args.kd_ratio)]
    schedule += [("width2", args.epochs_width2, 4, args.kd_ratio)]
    total_epochs = sum(e for _, e, _, _ in schedule)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))

    best_acc = -1.0
    epoch_global = 0
    teacher_model: Optional[OFACifarNet] = None
    shrink_scheduler_initialized = False

    # Save the config used for this run.
    args_dict = {k:v for k,v in vars(args).items() if k != "func"}
    with open(save_dir / "run_args.json", "w") as f:
        json.dump(args_dict, f, indent=2, default=str)

    for stage, n_epochs, subnets_per_batch, kd_ratio in schedule:
        if n_epochs <= 0:
            continue
        print(f"\n=== Starting stage={stage}, epochs={n_epochs}, subnets_per_batch={subnets_per_batch} ===")

        # OFA progressive shrinking is fine-tuning. Keeping the high LR used
        # for the largest subnet can wipe out the shared weights when smaller
        # kernels/depths/widths are introduced.
        if stage != "largest" and args.shrink_lr is not None and not shrink_scheduler_initialized:
            for group in optimizer.param_groups:
                group["lr"] = args.shrink_lr
            shrink_epochs = sum(e for st, e, _, _ in schedule if st != "largest")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, shrink_epochs))
            shrink_scheduler_initialized = True

        # Freeze the trained largest subnet once and use it as the distillation
        # teacher for all later stages.
        if stage != "largest" and teacher_model is None:
            teacher_model = make_frozen_teacher(model).to(device)
            print("Frozen largest-subnet teacher created for KD.")

        # Apply simplified channel sorting before width-shrinking stages.
        if args.channel_sort and stage in {"width1", "width2"}:
            print("Applying simplified L1 channel sorting before width-shrinking stage.")
            model.sort_all_middle_channels_by_l1()

        for _ in range(n_epochs):
            epoch_global += 1
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch_global,
                stage=stage,
                subnets_per_batch=subnets_per_batch,
                kd_ratio=kd_ratio,
                kd_temperature=args.kd_temperature,
                log_interval=args.log_interval,
                teacher_model=teacher_model,
            )
            scheduler.step()

            # Track validation accuracy of the largest subnet. This is not the only
            # metric that matters, but it is a stable checkpoint signal.
            model.set_largest_subnet()
            val_metrics = evaluate(model, val_loader, device, spec=model.search_space.largest_spec())
            print(
                f"epoch={epoch_global:03d} done stage={stage} "
                f"train_acc={train_metrics['acc']:.2f} val_largest_acc={val_metrics['acc']:.2f} "
                f"lr={scheduler.get_last_lr()[0]:.5f}"
            )

            is_best = val_metrics["acc"] > best_acc
            if is_best:
                best_acc = val_metrics["acc"]
                save_checkpoint(
                    ckpt_dir / "supernet_best.pt",
                    model,
                    optimizer,
                    epoch_global,
                    best_acc,
                    extra={"stage": stage, "val_metrics": val_metrics},
                )
            if epoch_global % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"supernet_epoch_{epoch_global:03d}.pt",
                    model,
                    optimizer,
                    epoch_global,
                    best_acc,
                    extra={"stage": stage, "val_metrics": val_metrics},
                )

    save_checkpoint(ckpt_dir / "supernet_last.pt", model, optimizer, epoch_global, best_acc)
    print(f"Training complete. Best largest-subnet validation accuracy: {best_acc:.2f}")

    # Final test uses the best checkpoint, not the last weights. Later shrinking
    # stages optimize many subnets and may perturb the largest-subnet BN stats.
    best_path = ckpt_dir / "supernet_best.pt"
    if best_path.exists():
        best_model = load_supernet_checkpoint(best_path, device)
        best_model.set_largest_subnet()
        test_metrics = evaluate(best_model, test_loader, device, spec=best_model.search_space.largest_spec())
        print(f"Best largest subnet test accuracy: {test_metrics['acc']:.2f}")
    else:
        model.set_largest_subnet()
        test_metrics = evaluate(model, test_loader, device, spec=model.search_space.largest_spec())
        print(f"Largest subnet test accuracy: {test_metrics['acc']:.2f}")


def cmd_sample_acc_dataset(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_loader, val_loader, _ = build_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )
    model = load_supernet_checkpoint(args.checkpoint, device)
    output = args.output or str(Path(args.save_dir) / f"acc_dataset_{args.num_arch_samples}.jsonl")
    sample_accuracy_dataset(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_arch_samples=args.num_arch_samples,
        output_path=output,
        seed=args.seed,
        bn_calib_batches=args.bn_calib_batches,
        val_batches=args.val_batches,
        sampling_strategy=args.sampling_strategy,
    )


def cmd_train_acc_predictor(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_accuracy_predictor(
        jsonl_path=args.dataset,
        save_dir=args.save_dir,
        epochs=args.predictor_epochs,
        batch_size=args.predictor_batch_size,
        lr=args.predictor_lr,
        weight_decay=args.predictor_weight_decay,
        seed=args.seed,
        device=device,
    )


def cmd_search(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    predictor, ss = load_accuracy_predictor(args.predictor, device)
    evolutionary_search(
        predictor=predictor,
        ss=ss,
        device=device,
        macs_limit=float(args.macs_limit),
        population_size=args.population_size,
        parent_size=args.parent_size,
        generations=args.generations,
        mutation_prob=args.mutation_prob,
        seed=args.seed,
        save_dir=Path(args.save_dir) / "search",
    )


def cmd_eval_subnet(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_loader, _, test_loader = build_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )
    model = load_supernet_checkpoint(args.checkpoint, device)
    with open(args.spec_json, "r") as f:
        obj = json.load(f)
    spec = obj["spec"] if "spec" in obj else obj
    validate_spec(spec, model.search_space)
    if args.bn_calib_batches > 0:
        calibrate_bn(model, train_loader, device, spec, num_batches=args.bn_calib_batches)
    metrics = evaluate(model, test_loader, device, spec=spec)
    macs = estimate_macs_given_spec(spec, model.search_space)
    print(f"Subnet spec: {spec}")
    print(f"MACs: {macs/1e6:.2f}M")
    print(f"Test accuracy: {metrics['acc']:.2f}, loss: {metrics['loss']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini-OFA for CIFAR-10")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", type=str, default="cuda")
        p.add_argument("--save-dir", type=str, default="./runs/ofa_cifar10")

    def add_data(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", type=str, default="./data")
        p.add_argument("--batch-size", type=int, default=128)
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument("--val-split", type=int, default=5000)

    p = sub.add_parser("train_supernet")
    add_common(p)
    add_data(p)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument(
        "--shrink-lr",
        type=float,
        default=0.005,
        help="Fine-tuning LR used after largest stage. Use a negative value to keep the global cosine LR.",
    )
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--epochs-largest", type=int, default=50)
    p.add_argument("--epochs-kernel", type=int, default=25)
    p.add_argument("--epochs-depth1", type=int, default=10)
    p.add_argument("--epochs-depth2", type=int, default=25)
    p.add_argument("--epochs-width1", type=int, default=10)
    p.add_argument("--epochs-width2", type=int, default=25)
    p.add_argument("--kd-ratio", type=float, default=0.5)
    p.add_argument("--kd-temperature", type=float, default=2.0)
    p.add_argument("--channel-sort", action="store_true")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=50)
    p.set_defaults(func=cmd_train_supernet)

    p = sub.add_parser("sample_acc_dataset")
    add_common(p)
    add_data(p)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-arch-samples", type=int, default=500)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--bn-calib-batches", type=int, default=20)
    p.add_argument("--val-batches", type=int, default=None)
    p.add_argument("--sampling-strategy", type=str, default="uniform", choices=["uniform", "small_biased", "large_biased"])
    p.set_defaults(func=cmd_sample_acc_dataset)

    p = sub.add_parser("train_acc_predictor")
    add_common(p)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--predictor-epochs", type=int, default=200)
    p.add_argument("--predictor-batch-size", type=int, default=64)
    p.add_argument("--predictor-lr", type=float, default=1e-3)
    p.add_argument("--predictor-weight-decay", type=float, default=1e-4)
    p.set_defaults(func=cmd_train_acc_predictor)

    p = sub.add_parser("search")
    add_common(p)
    p.add_argument("--predictor", type=str, required=True)
    p.add_argument("--macs-limit", type=float, default=80e6)
    p.add_argument("--population-size", type=int, default=100)
    p.add_argument("--parent-size", type=int, default=25)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--mutation-prob", type=float, default=0.1)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("eval_subnet")
    add_common(p)
    add_data(p)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--spec-json", type=str, required=True)
    p.add_argument("--bn-calib-batches", type=int, default=100)
    p.set_defaults(func=cmd_eval_subnet)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
