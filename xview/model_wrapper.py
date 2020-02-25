from typing import List, Union, Dict, Any

import torch
from catalyst.dl import CallbackOrder, logger, RunnerState, Callback
from catalyst.dl.callbacks.criterion import _add_loss_to_state, CriterionCallback
from torch import nn, Tensor

from xview.dataset import INPUT_IMAGE_KEY


class ModelTrainer(nn.Module):
    """
    Adapter class that computes loss on each GPU independently and returns only computed losses
    """

    def __init__(
            self,
            model: nn.Module,
            losses: List[nn.Module],
            loss_input_keys: List[str],
            loss_output_keys: List[str],
            loss_key="losses",
            model_input_key=INPUT_IMAGE_KEY,
    ):
        """

        :param model:
        :param loss_output_keys: List of keys to get outputs for each loss function
        :param losses: List of loss functions
        """
        super().__init__()
        self.model = model
        self.input_key = model_input_key
        self.output_keys = loss_output_keys
        self.losses = nn.ModuleList(losses)
        self.loss_key = loss_key
        self.loss_input_keys = loss_input_keys

    def forward(self, **input):
        model_output = self.model(input[self.input_key])

        losses = []
        for input_key, output_key, loss_fn in zip(self.loss_input_keys, self.output_keys, self.losses):
            target = input[input_key]
            output = model_output[output_key]
            loss = loss_fn(output, target)
            losses.append(loss)

        model_output[self.loss_key] = losses
        return model_output


class PassthroughCriterionCallback(CriterionCallback):
    """
    This callback allows you to aggregate the values of the loss
    (with different aggregation strategies)
    and put the value back into ``state.loss``.
    """

    def __init__(
            self,
            prefix: str,
            output_key="losses",
            loss_keys: Union[str, List[str], Dict[str, float]] = None,
            loss_aggregate_fn: str = "sum",
    ) -> None:
        """
        Args:
            prefix (str): new key for aggregated loss.
            loss_keys (Union[str, List[str], Dict[str, float]]): If not empty,
                it aggregates only the values from the loss by these keys.
                for ``weighted_sum`` aggregation it must be a Dict[str, float].
            loss_aggregate_fn (str): function for aggregation.
                Must be either ``sum``, ``mean`` or ``weighted_sum``.
        """
        super().__init__(prefix=prefix)
        if prefix is None or not isinstance(prefix, str):
            raise ValueError("prefix must be str")
        self.prefix = prefix

        if isinstance(loss_keys, str):
            loss_keys = [loss_keys]
        self.loss_keys = loss_keys
        self.output_key = output_key
        self.loss_aggregate_name = loss_aggregate_fn

    def on_stage_start(self, state: RunnerState):
        pass

    def on_batch_end(self, state: RunnerState) -> None:
        """
        Computes the loss and add it to the metrics
        """
        losses = state.output[self.output_key]
        losses = [torch.sum(x) for x in losses]  # Sum losses from all devices

        for loss_name, loss in zip(self.loss_keys, losses):
            state.metrics.add_batch_value(metrics_dict={loss_name: loss.item()})

        loss = torch.sum(torch.stack(losses))
        _add_loss_to_state(self.prefix, state, loss)

        state.metrics.add_batch_value(metrics_dict={self.prefix: loss.item()})
