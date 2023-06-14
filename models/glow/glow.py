import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.glow.act_norm import ActNorm
from models.glow.coupling import Coupling, CondCoupling
from models.glow.inv_conv import InvConv

import math

class Glow(nn.Module):
    def __init__(self, num_channels, num_levels, num_steps, mode='sketch', inp_channel = 1, cond_channel = 1, cc = True):
        super(Glow, self).__init__()

        # Use bounds to rescale images before converting to logits, not learned
        self.register_buffer('bounds', torch.tensor([0.95], dtype=torch.float32))
        self.dequant_flows = _Dequantization(in_channels=4 * inp_channel,
                                             mid_channels=num_channels,
                                             num_flows=4, cc = cc)
        self.flows = _Glow(in_channels=4 * inp_channel, 
                           cond_channels=4 * cond_channel,
                           mid_channels=num_channels,
                           num_levels=num_levels,
                           num_steps=num_steps,
                           cc = cc)
        self.mode = mode

    def forward(self, x, x_cond, reverse=False):
        sldj = torch.zeros(x.size(0), device=x.device)
        x = squeeze(x)
        x_cond = squeeze(x_cond)
        if not reverse:
            # # Variational Dequantization
            # x, sldj = self.dequantize(x, sldj)
            # x, sldj = self.to_logits(x, sldj)

            # Uniform Dequantization
            x, sldj = self._pre_process(x)
        x, sldj = self.flows(x, x_cond, sldj, reverse)
        x = squeeze(x, reverse=True)

        return x, sldj

    def dequantize(self, x, sldj):
        if self.dequant_flows is not None:
            x, sldj = self.dequant_flows(x, sldj)
        else:
            x = (x * 255. + torch.rand_like(x)) / 256.

        return x, sldj

    def to_logits(self, x, sldj):
        """Convert the input image `x` to logits.

        Args:
            x (torch.Tensor): Input image.
            sldj (torch.Tensor): Sum log-determinant of Jacobian.

        Returns:
            y (torch.Tensor): Dequantized logits of `x`.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        y = (2 * x - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = sldj + ldj.flatten(1).sum(-1)

        return y, sldj

    def _pre_process(self, x):
        """Dequantize the input image `x` and convert to logits.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1

        Args:
            x (torch.Tensor): Input image.

        Returns:
            y (torch.Tensor): Dequantized logits of `x`.
        """
        # y = x
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return y, sldj

def safe_log(x):
    return torch.log(x.clamp(min=1e-22))

class _Dequantization(nn.Module):
    def __init__(self, in_channels, mid_channels, num_flows=4, cc = True):
        super(_Dequantization, self).__init__()
        self.flows = nn.ModuleList([_FlowStep(in_channels=in_channels,
                                              cond_channels=in_channels,
                                              mid_channels=mid_channels,
                                              cc = cc)
                                    for _ in range(num_flows)])

    def forward(self, x, sldj):
        u = torch.randn_like(x)
        eps_nll = 0.5 * (u ** 2 + math.log(2 * math.pi))

        for flow in self.flows:
            u, sldj = flow(u, x, sldj)

        u = torch.sigmoid(u)
        x = (x * 255. + u) / 256.

        sigmoid_ldj = safe_log(u) + safe_log(1. - u)
        sldj = sldj + (eps_nll + sigmoid_ldj).flatten(1).sum(-1)

        return x, sldj

class _Glow(nn.Module):
    """Recursive constructor for a Glow model. Each call creates a single level.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in hidden layers of each step.
        num_levels (int): Number of levels to construct. Counter for recursion.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(self, in_channels, cond_channels, mid_channels, num_levels, num_steps, cc = True):
        super(_Glow, self).__init__()
        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels,
                                              cond_channels=cond_channels,
                                              mid_channels=mid_channels,
                                              cc = cc)
                                    for _ in range(num_steps)])

        if num_levels > 1:
            self.next = _Glow(in_channels=4 * in_channels,
                              cond_channels=4 * cond_channels,
                              mid_channels=mid_channels,
                              num_levels=num_levels - 1,
                              num_steps=num_steps,
                              cc = cc)
        else:
            self.next = None

    def forward(self, x, x_cond, sldj, reverse=False):
        if not reverse:
            for step in self.steps:
                x, sldj = step(x, x_cond, sldj, reverse)

        if self.next is not None:
            x = squeeze(x)
            x_cond = squeeze(x_cond)
            # x, x_split = x.chunk(2, dim=1)
            x, sldj = self.next(x, x_cond, sldj, reverse)
            # x = torch.cat((x, x_split), dim=1)
            x = squeeze(x, reverse=True)
            x_cond = squeeze(x_cond, reverse=True)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, x_cond, sldj, reverse)

        return x, sldj


class _FlowStep(nn.Module):
    def __init__(self, in_channels, cond_channels, mid_channels, cc = True):
        super(_FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels, return_ldj=True)
        self.conv = InvConv(in_channels)
        self.cc = cc
        if self.cc:
            self.cond_coup = CondCoupling(in_channels, cond_channels)
        self.coup = Coupling(in_channels // 2, cond_channels, mid_channels)

    def forward(self, x, x_cond, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, x_cond, sldj, reverse)
            if self.cc:
                x, sldj = self.cond_coup(x, x_cond, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.norm(x, sldj, reverse)
        else:
            x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            if self.cc:
                x, sldj = self.cond_coup(x, x_cond, sldj, reverse)
            x, sldj = self.coup(x, x_cond, sldj, reverse)

        return x, sldj


def squeeze(x, reverse=False):
    """Trade spatial extent for channels. In forward direction, convert each
    1x4x4 volume of input into a 4x1x1 volume of output.

    Args:
        x (torch.Tensor): Input to squeeze or unsqueeze.
        reverse (bool): Reverse the operation, i.e., unsqueeze.

    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    if reverse:
        # Unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # Squeeze
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x
    
if __name__ == "__main__":
    model = Glow(num_channels=128, num_levels=3, num_steps=6)
    img = torch.randn(1,3,32,32) *3.
    img = img.sigmoid()
    z, ldj = model(img)
    rc, _ = model(z, reverse=True)
    print(img-rc.sigmoid())
