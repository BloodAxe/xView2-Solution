import torch
from pytorch_toolbelt.optimization.lr_schedules import PolyLR
from torch.optim.lr_scheduler import (
    OneCycleLR,
    ExponentialLR,
    CyclicLR,
    MultiStepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)


def get_scheduler(scheduler_name: str, optimizer, lr, num_epochs, batches_in_epoch=None):
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None

    if scheduler_name.lower() == "poly":
        return PolyLR(optimizer, num_epochs, gamma=0.9)

    if scheduler_name.lower() == "cos":
        return CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5)

    if scheduler_name.lower() == "cosr":
        return CosineAnnealingWarmRestarts(optimizer, T_0=max(2, num_epochs // 4), eta_min=1e-5)

    if scheduler_name.lower() in {"1cycle", "one_cycle"}:
        return OneCycleLR(
            optimizer, lr_range=(lr, 1e-6, 1e-5), num_steps=batches_in_epoch, warmup_fraction=0.05, decay_fraction=0.1
        )

    if scheduler_name.lower() == "exp":
        return ExponentialLR(optimizer, gamma=0.95)

    if scheduler_name.lower() == "clr":
        return CyclicLR(
            optimizer,
            base_lr=1e-6,
            max_lr=lr,
            step_size_up=batches_in_epoch // 4,
            # mode='exp_range',
            gamma=0.99,
        )

    if scheduler_name.lower() == "multistep":
        return MultiStepLR(
            optimizer, milestones=[int(num_epochs * 0.5), int(num_epochs * 0.7), int(num_epochs * 0.9)], gamma=0.3
        )

    if scheduler_name.lower() == "simple":
        return MultiStepLR(optimizer, milestones=[int(num_epochs * 0.4), int(num_epochs * 0.7)], gamma=0.4)

    raise KeyError(scheduler_name)
