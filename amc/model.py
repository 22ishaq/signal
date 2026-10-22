"""1-D residual CNN that classifies raw I/Q frames.

The network normalises each frame to unit RMS power (so it is insensitive to the absolute
received level after fading), downsamples once in the stem, then applies residual 1-D convolution
blocks and global average + max pooling. Global pooling makes it accept any frame length.

The stride-2 stem matters in practice: a full-resolution stem with the same depth costs ~3x the
MACs and was ~34x slower per training step on Apple's MPS backend.
"""

from __future__ import annotations

import torch
from torch import nn


class ResBlock1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.skip = (
            nn.Identity()
            if stride == 1 and c_in == c_out
            else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + self.skip(x))


class AMCNet(nn.Module):
    def __init__(self, n_classes: int = 4, width: int = 32, kernel: int = 5, dropout: float = 0.2):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv1d(2, w, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(w), nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            ResBlock1d(w, w, kernel=kernel),
            ResBlock1d(w, 2 * w, stride=2, kernel=kernel),
            ResBlock1d(2 * w, 2 * w, kernel=kernel),
            ResBlock1d(2 * w, 4 * w, stride=2, kernel=kernel),
            ResBlock1d(4 * w, 4 * w, stride=2, kernel=kernel),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(8 * w, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x.pow(2).sum(dim=1, keepdim=True), dim=2, keepdim=True)).clamp_min(1e-8)
        h = self.blocks(self.stem(x / rms))
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, frame_len: int) -> int:
    """Multiply-accumulates per frame for the conv and linear layers (forward hooks)."""
    macs = 0

    def conv_hook(m: nn.Conv1d, _inp, out):
        nonlocal macs
        macs += out.shape[1] * out.shape[2] * (m.in_channels // m.groups) * m.kernel_size[0]

    def linear_hook(m: nn.Linear, _inp, _out):
        nonlocal macs
        macs += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 2, frame_len, device=next(model.parameters()).device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs
