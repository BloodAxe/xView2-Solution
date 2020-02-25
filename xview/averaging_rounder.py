from functools import partial
from multiprocessing.pool import Pool

import cv2
import numpy as np
import scipy as sp
import torch
from pytorch_toolbelt.utils.torch_utils import to_numpy
from tqdm import tqdm

from xview.dataset import read_mask
from xview.metric import CompetitionMetricCallback
from xview.postprocessing import make_predictions_naive


def _compute_fn(args, coef_exp):
    xi, dmg_true = args

    loc_pred, dmg_pred = make_predictions_naive(xi.astype(np.float32) * coef_exp)
    row = CompetitionMetricCallback.get_row_pair(loc_pred, dmg_pred, dmg_true, dmg_true)
    return row


class AveragingOptimizedRounder(object):
    def __init__(self, apply_softmax, workers=0):
        self.coef_ = 0
        self.workers = workers
        self.apply_softmax = apply_softmax

    @torch.no_grad()
    def _prepare_data(self, X, y):
        X_data = []
        n = len(X[0])
        m = len(X)

        for i in tqdm(range(n), desc="Loading predictions"):
            x_preds = []
            for j in range(m):
                x = np.load(X[j][i])
                if self.apply_softmax == "pre":
                    x = torch.from_numpy(x).float().softmax(dim=0).numpy().astype(np.float16)
                x_preds.append(x)

            x = np.mean(np.stack(x_preds), axis=0)

            if self.apply_softmax == "post":
                x = torch.from_numpy(x).float().softmax(dim=0).numpy().astype(np.float16)

            X_data.append(x)

        Y_data = [read_mask(yi) for yi in tqdm(y, desc="Loading ground-truths")]
        assert len(X_data) == len(Y_data)

        print("Loaded data into memory")
        return X_data, Y_data

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

    def fit(self, X, y):
        X_data, Y_data = self._prepare_data(X, y)
        loss_partial = partial(self._target_metric_loss, X=X_data, y=Y_data)
        initial_coef = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method="nelder-mead", options={"maxiter": 100, "xatol": 0.001}
        )
        del X_data, Y_data
        return self.coefficients()

    def predict(self, X, y, coef: np.ndarray):
        coef_exp = np.expand_dims(np.expand_dims(coef, -1), -1)

        all_rows = []

        X_data, Y_data = self._prepare_data(X, y)

        proc_fn = partial(_compute_fn, coef_exp=coef_exp)

        with Pool(self.workers) as wp:
            for row in wp.imap_unordered(proc_fn, zip(X_data, Y_data)):
                all_rows.append(row)

        score, localization_f1, damage_f1, damage_f1s = CompetitionMetricCallback.compute_metrics(all_rows)
        del X_data, Y_data
        return score, localization_f1, damage_f1, damage_f1s

    def coefficients(self):
        return self.coef_["x"]
