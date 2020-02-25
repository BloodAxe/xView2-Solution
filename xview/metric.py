from typing import List

import numpy as np
import pandas as pd
import torch
from catalyst.dl import Callback, RunnerState, CallbackOrder
from pytorch_toolbelt.utils.catalyst import get_tensorboard_logger
from pytorch_toolbelt.utils.torch_utils import to_numpy
from pytorch_toolbelt.utils.visualization import render_figure_to_tensor, plot_confusion_matrix
from torchnet.meter import ConfusionMeter

from .dataset import OUTPUT_MASK_PRE_KEY, OUTPUT_MASK_POST_KEY, INPUT_MASK_PRE_KEY, INPUT_MASK_POST_KEY
from .xview2_metrics import F1Recorder


class CompetitionMetricCallback(Callback):
    """
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        image_id_key: str = "image_id",
        prefix: str = "weighted_f1",
    ):
        super().__init__(CallbackOrder.Metric)
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.image_id_key = image_id_key
        self.all_rows = []

    def on_loader_start(self, state):
        self.all_rows = []

    @staticmethod
    def extract_buildings(x: np.ndarray):
        """ Returns a mask of the buildings in x """
        buildings = x.copy()
        buildings[x > 0] = 1
        return buildings

    @staticmethod
    def compute_tp_fn_fp(pred: np.ndarray, targ: np.ndarray, c: int) -> List[int]:
        """
        Computes the number of TPs, FNs, FPs, between a prediction (x) and a target (y) for the desired class (c)

        Args:
            pred (np.ndarray): prediction
            targ (np.ndarray): target
            c (int): positive class
        """
        TP = np.logical_and(pred == c, targ == c).sum()
        FN = np.logical_and(pred != c, targ == c).sum()
        FP = np.logical_and(pred == c, targ != c).sum()
        return [TP, FN, FP]

    @classmethod
    def get_row_pair(cls, lp, dp, lt, dt):
        """
        Builds a row of TPs, FNs, and FPs for both the localization dataframe and the damage dataframe.
        This pair of rows are built in the same function as damages are only assessed where buildings are predicted.

        Args:
            lp: localization predictions
            dp: damage predictions
            lt: localization targets
            dt: damage targets
        """
        lp_b, lt_b, dt_b = map(cls.extract_buildings, (lp, lt, dt))  # convert all damage scores 1-4 to 1

        dp = dp * lp_b  # only give credit to damages where buildings are predicted
        dp, dt = dp[dt_b == 1], dt[dt_b == 1]  # only score damage where there exist buildings in target damage

        lrow = cls.compute_tp_fn_fp(lp_b, lt_b, 1)
        drow = []
        for i in range(1, 5):
            drow += cls.compute_tp_fn_fp(dp, dt, i)
        return lrow, drow

    def on_batch_end(self, state: RunnerState):
        image_ids = state.input[self.image_id_key]
        outputs = to_numpy(torch.argmax(state.output[self.output_key].detach(), dim=1))
        targets = to_numpy(state.input[self.input_key].detach())

        rows = []
        for image_id, y_true, y_pred in zip(image_ids, targets, outputs):
            row = self.get_row_pair(y_pred, y_pred, y_true, y_true)
            rows.append(row)

        self.all_rows.extend(rows)

        score, localization_f1, damage_f1, damage_f1s = self.compute_metrics(rows)
        state.metrics.add_batch_value(self.prefix + "_batch" + "/localization_f1", localization_f1)
        state.metrics.add_batch_value(self.prefix + "_batch" + "/damage_f1", damage_f1)
        state.metrics.add_batch_value(self.prefix + "_batch", score)

    @staticmethod
    def compute_metrics(rows):
        lcolumns = ["lTP", "lFN", "lFP"]
        ldf = pd.DataFrame([lrow for lrow, drow in rows], columns=lcolumns)

        dcolumns = ["dTP1", "dFN1", "dFP1", "dTP2", "dFN2", "dFP2", "dTP3", "dFN3", "dFP3", "dTP4", "dFN4", "dFP4"]
        ddf = pd.DataFrame([drow for lrow, drow in rows], columns=dcolumns)

        TP = ldf["lTP"].sum()
        FP = ldf["lFP"].sum()
        FN = ldf["lFN"].sum()
        lf1r = F1Recorder(TP, FP, FN, "Buildings")

        dmg2str = {
            1: f"No damage     (1) ",
            2: f"Minor damage  (2) ",
            3: f"Major damage  (3) ",
            4: f"Destroyed     (4) ",
        }

        df1rs = []
        for i in range(1, 5):
            TP = ddf[f"dTP{i}"].sum()
            FP = ddf[f"dFP{i}"].sum()
            FN = ddf[f"dFN{i}"].sum()
            df1rs.append(F1Recorder(TP, FP, FN, dmg2str[i]))

        localization_f1 = lf1r.f1
        damage_f1s = [F1.f1 for F1 in df1rs]
        harmonic_mean = lambda xs: len(xs) / sum((x + 1e-6) ** -1 for x in xs)
        damage_f1 = harmonic_mean(damage_f1s)

        score = 0.3 * localization_f1 + 0.7 * damage_f1

        return score, localization_f1, damage_f1, damage_f1s

    def on_loader_end(self, state):
        score, localization_f1, damage_f1, damage_f1s = self.compute_metrics(self.all_rows)

        state.metrics.epoch_values[state.loader_name][self.prefix + "/localization_f1"] = localization_f1
        state.metrics.epoch_values[state.loader_name][self.prefix + "/damage_f1"] = damage_f1
        state.metrics.epoch_values[state.loader_name][self.prefix] = score

        class_names = ["no_damage", "minor_damage", "major_damage", "destroyed"]
        for i in range(4):
            state.metrics.epoch_values[state.loader_name][self.prefix + f"/{class_names[i]}"] = damage_f1s[i]


def default_multilabel_activation(x):
    return (x.sigmoid() > 0.5).long()


class MultilabelConfusionMatrixCallback(Callback):
    """
    Compute and log confusion matrix to Tensorboard.
    For use with Multiclass classification/segmentation.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "confusion_matrix",
        class_names: List[str] = None,
        num_classes: int = None,
        ignore_index=None,
        activation_fn=default_multilabel_activation,
    ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        :param ignore_index: same meaning as in nn.CrossEntropyLoss
        """
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.class_names = class_names
        self.num_classes = num_classes if class_names is None else len(class_names)
        self.output_key = output_key
        self.input_key = input_key
        self.ignore_index = ignore_index
        self.confusion_matrix = None
        self.activation_fn = activation_fn

    def on_loader_start(self, state):
        self.confusion_matrix = ConfusionMeter(self.num_classes)

    def on_batch_end(self, state: RunnerState):
        outputs: torch.Tensor = state.output[self.output_key].detach().cpu()
        outputs: torch.Tensor = self.activation_fn(outputs)

        targets: torch.Tensor = state.input[self.input_key].detach().cpu()

        # Flatten
        outputs = outputs.view(outputs.size(0), outputs.size(1), -1).permute(0, 2, 1).contiguous()
        targets = targets.view(targets.size(0), targets.size(1), -1).permute(0, 2, 1).contiguous()
        targets = targets.type_as(outputs)

        for class_index in range(self.num_classes):
            outputs_i = outputs[class_index].view(-1)
            targets_i = targets[class_index].view(-1)

            if self.ignore_index is not None:
                mask = targets_i != self.ignore_index
                outputs_i = outputs_i[mask]
                targets_i = targets_i[mask]

            self.confusion_matrix.add(predicted=outputs_i, target=targets_i)

    def on_loader_end(self, state):
        if self.class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]
        else:
            class_names = self.class_names

        num_classes = len(class_names)
        cm = self.confusion_matrix.value()

        fig = plot_confusion_matrix(
            cm,
            figsize=(6 + num_classes // 3, 6 + num_classes // 3),
            class_names=class_names,
            normalize=True,
            noshow=True,
        )
        fig = render_figure_to_tensor(fig)

        logger = get_tensorboard_logger(state)
        logger.add_image(f"{self.prefix}/epoch", fig, global_step=state.step)
