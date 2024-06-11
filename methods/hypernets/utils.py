from typing import Dict

import torch
from torch import nn


def get_param_dict(net: nn.Module) -> Dict[str, nn.Parameter]:
    """A dict of named parameters of an nn.Module"""
    return {n: p for (n, p) in net.named_parameters()}


def set_from_param_dict(module: nn.Module, param_dict: Dict[str, torch.Tensor]):
    """
    Sets the values of `module` parameters with the values from `param_dict`.

    Works just like:
        nn.Module.load_state_dict()

    with the exception that those parameters are not tunable by default, because
    we set their values to bare tensors instead of nn.Parameter.

    This means that a network with such params cannot be trained directly with an optimizer.
    However, gradients may still flow through those tensors, so it's useful for the use-case of hypernetworks.

    """
    for sdk, v in param_dict.items():
        keys = sdk.split(".")
        param_name = keys[-1]
        m = module
        for k in keys[:-1]:
            try:
                k = int(k)
                m = m[k]
            except:
                m = getattr(m, k)

        param = getattr(m, param_name)
        assert param.shape == v.shape, (sdk, param.shape, v.shape)
        delattr(m, param_name)
        setattr(m, param_name, v)


def kl_diag_gauss_with_standard_gauss(mean, logvar):
    mean_flat = torch.cat([t.view(-1) for t in mean])
    logvar_flat = torch.cat([t.view(-1) for t in logvar])
    var_flat = logvar_flat.exp()

    return -0.5 * torch.sum(1 + logvar_flat - mean_flat.pow(2) - var_flat)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu
