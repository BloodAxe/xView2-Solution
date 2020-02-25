import torch

__all__ = ["report_checkpoint", "clean_checkpoint"]

from catalyst.dl import CriterionCallback

from .dataset import UNLABELED_SAMPLE
from .losses import get_loss


def report_checkpoint(checkpoint):
    print("Epoch          :", checkpoint["epoch"])

    # {'mask_pre/bce': 0.011939526008819881, 'mask_post/ce': 0.039905175798535336, 'loss': 0.05184470175895215, 'jaccard': 0.6682964469961886, '_base/lr': 0.001,
    # '_base/momentum': 0.9, '_timers/data_time': 0.2810825881448131, '_timers/model_time': 0.025946252149632927, '_timers/batch_time': 0.3070834094035581, '_timers/_fps': 121.48878000184467,
    # 'localization_f1': 0.7123450379603988, 'damage_f1': 0.021565931686082063, 'weighted_f1': 0.22879966356837708, 'jaccard_no-damage': 0.4595737876547124, 'jaccard_minor-damage': 0.7845541293707017, 'jaccard_major-damage': 0.7821522489229876, 'jaccard_destroyed': 0.6469056220363518}
    skip_fields = [
        "_base/lr",
        "_base/momentum",
        "_timers/data_time",
        "_timers/model_time",
        "_timers/batch_time",
        "_timers/_fps",
    ]
    print(
        "Metrics (Train):", [(k, v) for k, v, in checkpoint["epoch_metrics"]["train"].items() if k not in skip_fields]
    )
    print(
        "Metrics (Valid):", [(k, v) for k, v, in checkpoint["epoch_metrics"]["valid"].items() if k not in skip_fields]
    )


def clean_checkpoint(src_fname, dst_fname):
    checkpoint = torch.load(src_fname)

    keys = ["criterion_state_dict", "optimizer_state_dict", "scheduler_state_dict"]

    for key in keys:
        if key in checkpoint:
            del checkpoint[key]

    torch.save(checkpoint, dst_fname)


def get_criterion_callback(loss_name, input_key, output_key, prefix=None, loss_weight=1.0, ignore_index=UNLABELED_SAMPLE):
    criterions_dict = {f"{prefix}/{loss_name}": get_loss(loss_name, ignore_index=ignore_index)}
    if prefix is None:
        prefix = input_key

    criterion_callback = CriterionCallback(
        prefix=f"{prefix}/{loss_name}",
        input_key=input_key,
        output_key=output_key,
        criterion_key=f"{prefix}/{loss_name}",
        multiplier=float(loss_weight),
    )

    return criterions_dict, criterion_callback, criterion_callback.prefix

def get_criterion(loss_name, prefix=None, ignore_index=UNLABELED_SAMPLE):
    loss = get_loss(loss_name, ignore_index=ignore_index)
    prefix = f"{prefix}/{loss_name}"
    return loss, prefix
