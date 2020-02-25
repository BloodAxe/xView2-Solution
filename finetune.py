from __future__ import absolute_import

import argparse
import collections
import json
import os
from datetime import datetime

from catalyst.dl import SupervisedRunner, OptimizerCallback, SchedulerCallback
from catalyst.dl.callbacks import CriterionAggregatorCallback, AccuracyCallback
from catalyst.utils import load_checkpoint, unpack_checkpoint
from pytorch_toolbelt.utils import fs, torch_utils
from pytorch_toolbelt.utils.catalyst import ShowPolarBatchesCallback, ConfusionMatrixCallback
from pytorch_toolbelt.utils.random import set_manual_seed
from pytorch_toolbelt.utils.torch_utils import count_parameters, transfer_weights, get_optimizable_parameters
from torch import nn
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader

from xview.dataset import (
    INPUT_IMAGE_KEY,
    OUTPUT_MASK_KEY,
    INPUT_MASK_KEY,
    get_datasets,
    OUTPUT_MASK_4_KEY,
    UNLABELED_SAMPLE,
    get_pseudolabeling_dataset,
    DISASTER_TYPE_KEY,
    UNKNOWN_DISASTER_TYPE_CLASS,
    DISASTER_TYPES,
    OUTPUT_EMBEDDING_KEY,
    DAMAGE_TYPE_KEY,
    OUTPUT_MASK_8_KEY,
    OUTPUT_MASK_16_KEY,
    OUTPUT_MASK_32_KEY,
)
from xview.metric import CompetitionMetricCallback
from xview.models import get_model
from xview.optim import get_optimizer
from xview.scheduler import get_scheduler
from xview.train_utils import clean_checkpoint, report_checkpoint, get_criterion_callback
from xview.visualization import draw_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-acc", "--accumulation-steps", type=int, default=1, help="Number of batches to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "-dd", "--data-dir", type=str, required=True, help="Data directory for INRIA sattelite dataset"
    )
    parser.add_argument("-m", "--model", type=str, default="resnet34_fpncat128", help="")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch Size during training, e.g. -b 64")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Epoch to run")
    # parser.add_argument('-es', '--early-stopping', type=int, default=None, help='Maximum number of epochs without improvement')
    # parser.add_argument('-fe', '--freeze-encoder', type=int, default=0, help='Freeze encoder parameters for N epochs')
    # parser.add_argument('-ft', '--fine-tune', action='store_true')
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--disaster-type-loss",
        type=str,
        default=None,  # [["ce", 1.0]],
        action="append",
        nargs="+",
        help="Criterion for classifying disaster type",
    )
    parser.add_argument(
        "--damage-type-loss",
        type=str,
        default=None,  # [["bce", 1.0]],
        action="append",
        nargs="+",
        help="Criterion for classifying presence of building with particular damage type",
    )

    parser.add_argument("-l", "--criterion", type=str, default=None, action="append", nargs="+", help="Criterion")
    parser.add_argument(
        "--mask4", type=str, default=None, action="append", nargs="+", help="Criterion for mask with stride 4"
    )
    parser.add_argument(
        "--mask8", type=str, default=None, action="append", nargs="+", help="Criterion for mask with stride 8"
    )
    parser.add_argument(
        "--mask16", type=str, default=None, action="append", nargs="+", help="Criterion for mask with stride 16"
    )
    parser.add_argument(
        "--mask32", type=str, default=None, action="append", nargs="+", help="Criterion for mask with stride 32"
    )
    parser.add_argument("--embedding", type=str, default=None)

    parser.add_argument("-o", "--optimizer", default="RAdam", help="Name of the optimizer")
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None, help="Checkpoint filename to use as initial model weights"
    )
    parser.add_argument("-w", "--workers", default=8, type=int, help="Num workers")
    parser.add_argument("-a", "--augmentations", default="safe", type=str, help="Level of image augmentations")
    parser.add_argument("--transfer", default=None, type=str, help="")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--size", default=512, type=int)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("-s", "--scheduler", default="multistep", type=str, help="")
    parser.add_argument("-x", "--experiment", default=None, type=str, help="")
    parser.add_argument("-d", "--dropout", default=0.0, type=float, help="Dropout before head layer")
    parser.add_argument("-pl", "--pseudolabeling", type=str, required=True)
    parser.add_argument("-wd", "--weight-decay", default=0, type=float, help="L2 weight decay")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dsv", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--only-buildings", action="store_true")
    parser.add_argument("--freeze-bn", action="store_true")
    parser.add_argument("--crops", action="store_true", help="Train on random crops")
    parser.add_argument("--post-transform", action="store_true")

    args = parser.parse_args()
    set_manual_seed(args.seed)

    data_dir = args.data_dir
    num_workers = args.workers
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    model_name = args.model
    optimizer_name = args.optimizer
    image_size = args.size, args.size
    fast = args.fast
    augmentations = args.augmentations
    fp16 = args.fp16
    scheduler_name = args.scheduler
    experiment = args.experiment
    dropout = args.dropout
    segmentation_losses = args.criterion
    verbose = args.verbose
    show = args.show
    accumulation_steps = args.accumulation_steps
    weight_decay = args.weight_decay
    fold = args.fold
    balance = args.balance
    only_buildings = args.only_buildings
    freeze_bn = args.freeze_bn
    train_on_crops = args.crops
    enable_post_image_transform = args.post_transform
    disaster_type_loss = args.disaster_type_loss
    train_batch_size = args.batch_size
    embedding_criterion = args.embedding
    damage_type_loss = args.damage_type_loss
    pseudolabels_dir = args.pseudolabeling

    # Compute batch size for validaion
    if train_on_crops:
        valid_batch_size = max(1, (train_batch_size * (image_size[0] * image_size[1])) // (1024 ** 2))
    else:
        valid_batch_size = train_batch_size

    run_train = num_epochs > 0

    model: nn.Module = get_model(model_name, dropout=dropout).cuda()

    if args.transfer:
        transfer_checkpoint = fs.auto_file(args.transfer)
        print("Transfering weights from model checkpoint", transfer_checkpoint)
        checkpoint = load_checkpoint(transfer_checkpoint)
        pretrained_dict = checkpoint["model_state_dict"]

        transfer_weights(model, pretrained_dict)

    if args.checkpoint:
        checkpoint = load_checkpoint(fs.auto_file(args.checkpoint))
        unpack_checkpoint(checkpoint, model=model)

        print("Loaded model weights from:", args.checkpoint)
        report_checkpoint(checkpoint)

    if freeze_bn:
        torch_utils.freeze_bn(model)
        print("Freezing bn params")

    runner = SupervisedRunner(input_key=INPUT_IMAGE_KEY, output_key=None)
    main_metric = "weighted_f1"
    cmd_args = vars(args)

    current_time = datetime.now().strftime("%b%d_%H_%M")
    checkpoint_prefix = f"{current_time}_{args.model}_{args.size}_fold{fold}"

    if fp16:
        checkpoint_prefix += "_fp16"

    if fast:
        checkpoint_prefix += "_fast"

    if pseudolabels_dir:
        checkpoint_prefix += "_pseudo"

    if train_on_crops:
        checkpoint_prefix += "_crops"

    if experiment is not None:
        checkpoint_prefix = experiment

    log_dir = os.path.join("runs", checkpoint_prefix)
    os.makedirs(log_dir, exist_ok=False)

    config_fname = os.path.join(log_dir, f"{checkpoint_prefix}.json")
    with open(config_fname, "w") as f:
        train_session_args = vars(args)
        f.write(json.dumps(train_session_args, indent=2))

    default_callbacks = [
        CompetitionMetricCallback(input_key=INPUT_MASK_KEY, output_key=OUTPUT_MASK_KEY, prefix="weighted_f1"),
        ConfusionMatrixCallback(
            input_key=INPUT_MASK_KEY,
            output_key=OUTPUT_MASK_KEY,
            class_names=["land", "no_damage", "minor_damage", "major_damage", "destroyed"],
            ignore_index=UNLABELED_SAMPLE,
        ),
    ]

    if show:
        default_callbacks += [
            ShowPolarBatchesCallback(draw_predictions, metric=main_metric + "_batch", minimize=False)
        ]

    train_ds, valid_ds, train_sampler = get_datasets(
        data_dir=data_dir,
        image_size=image_size,
        augmentation=augmentations,
        fast=fast,
        fold=fold,
        balance=balance,
        only_buildings=only_buildings,
        train_on_crops=train_on_crops,
        crops_multiplication_factor=1,
        enable_post_image_transform=enable_post_image_transform,
    )

    if run_train:
        loaders = collections.OrderedDict()
        callbacks = default_callbacks.copy()
        criterions_dict = {}
        losses = []

        unlabeled_train = get_pseudolabeling_dataset(
            data_dir,
            include_masks=True,
            image_size=image_size,
            augmentation="medium_nmd",
            train_on_crops=train_on_crops,
            enable_post_image_transform=enable_post_image_transform,
            pseudolabels_dir=pseudolabels_dir,
        )

        train_ds = train_ds + unlabeled_train

        print("Using online pseudolabeling with ", len(unlabeled_train), "samples")

        loaders["train"] = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

        loaders["valid"] = DataLoader(valid_ds, batch_size=valid_batch_size, num_workers=num_workers, pin_memory=True)

        # Create losses
        for criterion in segmentation_losses:
            if isinstance(criterion, (list, tuple)) and len(criterion) == 2:
                loss_name, loss_weight = criterion
            else:
                loss_name, loss_weight = criterion[0], 1.0

            cd, criterion, criterion_name = get_criterion_callback(
                loss_name,
                prefix="segmentation",
                input_key=INPUT_MASK_KEY,
                output_key=OUTPUT_MASK_KEY,
                loss_weight=float(loss_weight),
            )
            criterions_dict.update(cd)
            callbacks.append(criterion)
            losses.append(criterion_name)
            print(INPUT_MASK_KEY, "Using loss", loss_name, loss_weight)

        if args.mask4 is not None:
            for criterion in args.mask4:
                if isinstance(criterion, (list, tuple)):
                    loss_name, loss_weight = criterion
                else:
                    loss_name, loss_weight = criterion, 1.0

                cd, criterion, criterion_name = get_criterion_callback(
                    loss_name,
                    prefix="mask4",
                    input_key=INPUT_MASK_KEY,
                    output_key=OUTPUT_MASK_4_KEY,
                    loss_weight=float(loss_weight),
                )
                criterions_dict.update(cd)
                callbacks.append(criterion)
                losses.append(criterion_name)
                print(OUTPUT_MASK_4_KEY, "Using loss", loss_name, loss_weight)

        if args.mask8 is not None:
            for criterion in args.mask8:
                if isinstance(criterion, (list, tuple)):
                    loss_name, loss_weight = criterion
                else:
                    loss_name, loss_weight = criterion, 1.0

                cd, criterion, criterion_name = get_criterion_callback(
                    loss_name,
                    prefix="mask8",
                    input_key=INPUT_MASK_KEY,
                    output_key=OUTPUT_MASK_8_KEY,
                    loss_weight=float(loss_weight),
                )
                criterions_dict.update(cd)
                callbacks.append(criterion)
                losses.append(criterion_name)
                print(OUTPUT_MASK_8_KEY, "Using loss", loss_name, loss_weight)

        if args.mask16 is not None:
            for criterion in args.mask16:
                if isinstance(criterion, (list, tuple)):
                    loss_name, loss_weight = criterion
                else:
                    loss_name, loss_weight = criterion, 1.0

                cd, criterion, criterion_name = get_criterion_callback(
                    loss_name,
                    prefix="mask16",
                    input_key=INPUT_MASK_KEY,
                    output_key=OUTPUT_MASK_16_KEY,
                    loss_weight=float(loss_weight),
                )
                criterions_dict.update(cd)
                callbacks.append(criterion)
                losses.append(criterion_name)
                print(OUTPUT_MASK_16_KEY, "Using loss", loss_name, loss_weight)

        if args.mask32 is not None:
            for criterion in args.mask32:
                if isinstance(criterion, (list, tuple)):
                    loss_name, loss_weight = criterion
                else:
                    loss_name, loss_weight = criterion, 1.0

                cd, criterion, criterion_name = get_criterion_callback(
                    loss_name,
                    prefix="mask32",
                    input_key=INPUT_MASK_KEY,
                    output_key=OUTPUT_MASK_32_KEY,
                    loss_weight=float(loss_weight),
                )
                criterions_dict.update(cd)
                callbacks.append(criterion)
                losses.append(criterion_name)
                print(OUTPUT_MASK_32_KEY, "Using loss", loss_name, loss_weight)

        if disaster_type_loss is not None:
            callbacks += [
                ConfusionMatrixCallback(
                    input_key=DISASTER_TYPE_KEY,
                    output_key=DISASTER_TYPE_KEY,
                    class_names=DISASTER_TYPES,
                    ignore_index=UNKNOWN_DISASTER_TYPE_CLASS,
                    prefix=f"{DISASTER_TYPE_KEY}/confusion_matrix",
                ),
                AccuracyCallback(
                    input_key=DISASTER_TYPE_KEY,
                    output_key=DISASTER_TYPE_KEY,
                    prefix=f"{DISASTER_TYPE_KEY}/accuracy",
                    activation="Softmax",
                ),
            ]

            for criterion in disaster_type_loss:
                if isinstance(criterion, (list, tuple)):
                    loss_name, loss_weight = criterion
                else:
                    loss_name, loss_weight = criterion, 1.0

                cd, criterion, criterion_name = get_criterion_callback(
                    loss_name,
                    prefix=DISASTER_TYPE_KEY,
                    input_key=DISASTER_TYPE_KEY,
                    output_key=DISASTER_TYPE_KEY,
                    loss_weight=float(loss_weight),
                    ignore_index=UNKNOWN_DISASTER_TYPE_CLASS,
                )
                criterions_dict.update(cd)
                callbacks.append(criterion)
                losses.append(criterion_name)
                print(DISASTER_TYPE_KEY, "Using loss", loss_name, loss_weight)

        if damage_type_loss is not None:
            callbacks += [
                # MultilabelConfusionMatrixCallback(
                #     input_key=DAMAGE_TYPE_KEY,
                #     output_key=DAMAGE_TYPE_KEY,
                #     class_names=DAMAGE_TYPES,
                #     prefix=f"{DAMAGE_TYPE_KEY}/confusion_matrix",
                # ),
                AccuracyCallback(
                    input_key=DAMAGE_TYPE_KEY,
                    output_key=DAMAGE_TYPE_KEY,
                    prefix=f"{DAMAGE_TYPE_KEY}/accuracy",
                    activation="Sigmoid",
                    threshold=0.5,
                )
            ]

            for criterion in damage_type_loss:
                if isinstance(criterion, (list, tuple)):
                    loss_name, loss_weight = criterion
                else:
                    loss_name, loss_weight = criterion, 1.0

                cd, criterion, criterion_name = get_criterion_callback(
                    loss_name,
                    prefix=DAMAGE_TYPE_KEY,
                    input_key=DAMAGE_TYPE_KEY,
                    output_key=DAMAGE_TYPE_KEY,
                    loss_weight=float(loss_weight),
                )
                criterions_dict.update(cd)
                callbacks.append(criterion)
                losses.append(criterion_name)
                print(DAMAGE_TYPE_KEY, "Using loss", loss_name, loss_weight)

        if embedding_criterion is not None:
            cd, criterion, criterion_name = get_criterion_callback(
                embedding_criterion,
                prefix="embedding",
                input_key=INPUT_MASK_KEY,
                output_key=OUTPUT_EMBEDDING_KEY,
                loss_weight=1.0,
            )
            criterions_dict.update(cd)
            callbacks.append(criterion)
            losses.append(criterion_name)
            print(OUTPUT_EMBEDDING_KEY, "Using loss", embedding_criterion)

        callbacks += [
            CriterionAggregatorCallback(prefix="loss", loss_keys=losses),
            OptimizerCallback(accumulation_steps=accumulation_steps, decouple_weight_decay=False),
        ]

        optimizer = get_optimizer(
            optimizer_name, get_optimizable_parameters(model), learning_rate, weight_decay=weight_decay
        )
        scheduler = get_scheduler(
            scheduler_name, optimizer, lr=learning_rate, num_epochs=num_epochs, batches_in_epoch=len(loaders["train"])
        )
        if isinstance(scheduler, CyclicLR):
            callbacks += [SchedulerCallback(mode="batch")]

        print("Train session    :", checkpoint_prefix)
        print("  FP16 mode      :", fp16)
        print("  Fast mode      :", args.fast)
        print("  Epochs         :", num_epochs)
        print("  Workers        :", num_workers)
        print("  Data dir       :", data_dir)
        print("  Log dir        :", log_dir)
        print("Data             ")
        print("  Augmentations  :", augmentations)
        print("  Train size     :", len(loaders["train"]), len(train_ds))
        print("  Valid size     :", len(loaders["valid"]), len(valid_ds))
        print("  Image size     :", image_size)
        print("  Train on crops :", train_on_crops)
        print("  Balance        :", balance)
        print("  Buildings only :", only_buildings)
        print("  Post transform :", enable_post_image_transform)
        print("  Pseudolabels   :", pseudolabels_dir)
        print("Model            :", model_name)
        print("  Parameters     :", count_parameters(model))
        print("  Dropout        :", dropout)
        print("Optimizer        :", optimizer_name)
        print("  Learning rate  :", learning_rate)
        print("  Weight decay   :", weight_decay)
        print("  Scheduler      :", scheduler_name)
        print("  Batch sizes    :", train_batch_size, valid_batch_size)
        print("  Criterion      :", segmentation_losses)
        print("  Damage type    :", damage_type_loss)
        print("  Disaster type  :", disaster_type_loss)
        print(" Embedding      :", embedding_criterion)

        # model training
        runner.train(
            fp16=fp16,
            model=model,
            criterion=criterions_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            loaders=loaders,
            logdir=os.path.join(log_dir, "opl"),
            num_epochs=num_epochs,
            verbose=verbose,
            main_metric=main_metric,
            minimize_metric=False,
            checkpoint_data={"cmd_args": cmd_args},
        )

        # Training is finished. Let's run predictions using best checkpoint weights
        best_checkpoint = os.path.join(log_dir, "main", "checkpoints", "best.pth")

        model_checkpoint = os.path.join(log_dir, "main", "checkpoints", f"{checkpoint_prefix}.pth")
        clean_checkpoint(best_checkpoint, model_checkpoint)

        del optimizer, loaders


if __name__ == "__main__":
    main()
