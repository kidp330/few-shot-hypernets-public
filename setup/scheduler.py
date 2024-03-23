import torch.optim.lr_scheduler as torch_schedulers

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from io_params import ParamHolder


def get_scheduler(params: ParamHolder, optimizer: Optimizer) -> LRScheduler:
    if params.lr_scheduler == "cosine":
        return torch_schedulers.CosineAnnealingWarmRestarts(optimizer, T_0=params.stop_epoch)
    elif params.lr_scheduler in ["none", "multisteplr"]:
        milestones = params.milestones or list(range(0, params.stop_epoch,
                                                     params.stop_epoch // 4))[1:]
        return torch_schedulers.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=1,
        )
    raise TypeError(params.lr_scheduler)
