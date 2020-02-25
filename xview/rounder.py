from functools import partial
from multiprocessing.pool import Pool

import cv2
import numpy as np
import scipy as sp
import torch
from pytorch_toolbelt.utils.torch_utils import to_numpy

from xview.dataset import read_mask
from xview.metric import CompetitionMetricCallback
from xview.postprocessing import make_predictions_naive


@torch.no_grad()
def _compute_fn(args, coef_exp):
    xi, dmg_true = args
    dmg_pred = xi.astype(np.float32) * coef_exp
    loc_pred, dmg_pred = make_predictions_naive(dmg_pred)

    if loc_pred.shape[0] != 1024:
        loc_pred = cv2.resize(loc_pred, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
        dmg_pred = cv2.resize(dmg_pred, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)

    row = CompetitionMetricCallback.get_row_pair(loc_pred, dmg_pred, dmg_true, dmg_true)
    return row


class OptimizedRounder(object):
    def __init__(self, apply_softmax, workers=0):
        self.coef_ = 0
        self.workers = workers
        self.apply_softmax = apply_softmax

    def _target_metric_loss(self, coef, X, y):
        coef_exp = np.expand_dims(np.expand_dims(coef, -1), -1)

        all_rows = []

        proc_fn = partial(_compute_fn, coef_exp=coef_exp)

        with Pool(self.workers) as wp:
            for row in wp.imap_unordered(proc_fn, zip(X, y)):
                all_rows.append(row)

        score, localization_f1, damage_f1, damage_f1s = CompetitionMetricCallback.compute_metrics(all_rows)
        print(score, localization_f1, damage_f1, damage_f1s, "coeffs", coef)
        return 1.0 - score

    def _prepare_data(self, X, y):
        assert self.apply_softmax == "pre"
        X_data = [to_numpy(torch.from_numpy(np.load(xi)).float().softmax(dim=0)).astype(np.float16) for xi in X]
        Y_data = [read_mask(yi) for yi in y]
        print("Loaded data into memory")
        return X_data, Y_data

    def fit(self, X, y):
        X_data, Y_data = self._prepare_data(X, y)

        loss_partial = partial(self._target_metric_loss, X=X_data, y=Y_data)
        initial_coef = [0.5, 1.1, 1.1, 1.1, 1.1]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method="nelder-mead", options={"maxiter": 100, "xatol": 0.001}
        )

        del X_data, Y_data
        return self.coefficients()

    def predict(self, X, y, coef: np.ndarray):
        X_data, Y_data = self._prepare_data(X, y)

        coef_exp = np.expand_dims(np.expand_dims(coef, -1), -1)
        all_rows = []
        proc_fn = partial(_compute_fn, coef_exp=coef_exp)

        with Pool(self.workers) as wp:
            for row in wp.imap_unordered(proc_fn, zip(X_data, Y_data)):
                all_rows.append(row)

        score, localization_f1, damage_f1, damage_f1s = CompetitionMetricCallback.compute_metrics(all_rows)
        del X_data, Y_data
        return score, localization_f1, damage_f1, damage_f1s

    def coefficients(self):
        return self.coef_["x"]
