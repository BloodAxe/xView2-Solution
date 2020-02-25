import numpy as np
import torch.nn.functional as F
from catalyst.dl import Callback, CallbackOrder, RunnerState
from pytorch_toolbelt.utils.catalyst import PseudolabelDatasetMixin
from pytorch_toolbelt.utils.torch_utils import to_numpy


class CEOnlinePseudolabelingCallback2d(Callback):
    """
    Online pseudo-labeling callback for multi-class problem.

    >>> unlabeled_train = get_test_dataset(
    >>>     data_dir, image_size=image_size, augmentation=augmentations
    >>> )
    >>> unlabeled_eval = get_test_dataset(
    >>>     data_dir, image_size=image_size
    >>> )
    >>>
    >>> callbacks += [
    >>>     CEOnlinePseudolabelingCallback2d(
    >>>         unlabeled_train.targets,
    >>>         pseudolabel_loader="label",
    >>>         prob_threshold=0.9)
    >>> ]
    >>> train_ds = train_ds + unlabeled_train
    >>>
    >>> loaders = collections.OrderedDict()
    >>> loaders["train"] = DataLoader(train_ds)
    >>> loaders["valid"] = DataLoader(valid_ds)
    >>> loaders["label"] = DataLoader(unlabeled_eval, shuffle=False) # ! shuffle=False is important !
    """

    def __init__(
        self,
        unlabeled_ds: PseudolabelDatasetMixin,
        pseudolabel_loader="label",
        prob_threshold=0.9,
        sample_index_key="index",
        output_key="logits",
        unlabeled_class=-100,
        label_smoothing=0.0,
        label_frequency=1,
    ):
        assert 1.0 > prob_threshold > 0.5

        super().__init__(CallbackOrder.Other)
        self.unlabeled_ds = unlabeled_ds
        self.pseudolabel_loader = pseudolabel_loader
        self.prob_threshold = prob_threshold
        self.sample_index_key = sample_index_key
        self.output_key = output_key
        self.unlabeled_class = unlabeled_class
        self.label_smoothing = label_smoothing
        self.label_frequency = label_frequency
        self.last_labeled_epoch = None
        self.should_relabel = None

    def on_epoch_start(self, state: RunnerState):
        self.should_relabel = (
            self.last_labeled_epoch is None or (state.epoch - self.last_labeled_epoch) % self.label_frequency == 0
        )

    def on_epoch_end(self, state: RunnerState):
        if self.should_relabel:
            self.last_labeled_epoch = state.epoch

    def get_probabilities(self, state: RunnerState):
        probs = state.output[self.output_key].detach().softmax(dim=1)
        indexes = state.input[self.sample_index_key]

        if probs.size(2) != 1024 or probs.size(3) != 1024:
            probs = F.interpolate(probs, size=(1024, 1024), mode="bilinear", align_corners=False)

        return to_numpy(probs), to_numpy(indexes)

    def on_batch_end(self, state: RunnerState):
        if state.loader_name != self.pseudolabel_loader:
            return

        if not self.should_relabel:
            return

        # Get predictions for batch
        probs, indexes = self.get_probabilities(state)

        for p, sample_index in zip(probs, indexes):
            max_prob = np.max(p, axis=0)
            class_index = np.argmax(p, axis=0)

            confident_classes = max_prob > self.prob_threshold
            class_index[~confident_classes] = self.unlabeled_class

            self.unlabeled_ds.set_target(sample_index, class_index)
