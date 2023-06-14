import numpy as np
import torch.nn as nn
import torch.nn.utils as utils
from scipy.stats import gennorm
import torch
import scipy.special as sp

def bits_per_dim(x, nll):
    """Get the bits per dimension implied by using model with `loss`
    for compressing `x`, assuming each entry can take on `k` discrete values.

    Args:
        x (torch.Tensor): Input to the model. Just used for dimensions.
        nll (torch.Tensor): Scalar negative log-likelihood loss tensor.

    Returns:
        bpd (torch.Tensor): Bits per dimension implied if compressing `x`.
    """
    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)

    return bpd


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters under `optimizer`.

    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)


# class NLLLoss(nn.Module):
#     """Negative log-likelihood loss assuming isotropic gaussian with unit norm.

#     Args:
#         k (int or float): Number of discrete values in each input dimension.
#             E.g., `k` is 256 for natural images.

#     See Also:
#         Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
#     """
#     def __init__(self, k=256, shape = 2, device = torch.device('cuda:0')):
#         super(NLLLoss, self).__init__()
#         self.k = k
#         self.shape = shape
#         self.device = device
        
#     def forward(self, z, sldj):
#         shape = self.shape
#         term1 = torch.log(torch.tensor(shape / (2 * sp.gamma(1 / shape))))
#         term2 = torch.abs(z) ** shape
#         prior_ll = term1 - term2
#         # prior_ll = torch.tensor(gennorm.logpdf(z.detach().cpu().numpy(), shape, loc = loc, scale = scale)).to(self.device)
#         # prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
#         prior_ll = prior_ll.flatten(1).sum(-1) \
#             - np.log(self.k) * np.prod(z.size()[1:])
#         ll = prior_ll + sldj
#         nll = -ll.mean()
#         # nll = torch.tensor(torch.nan)
#         # if torch.isnan(nll):
#         #     nll = torch.tensor(0.0, requires_grad=True).to(self.device)

#         return nll

class NLLLoss(nn.Module):
    """Negative log-likelihood loss assuming isotropic gaussian with unit norm.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, k=256, shape = 2):
        super(NLLLoss, self).__init__()
        self.k = k
        self.shape = shape

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll