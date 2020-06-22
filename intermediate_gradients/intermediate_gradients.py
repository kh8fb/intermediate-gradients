import torch
from torch import Tensor

from captum.attr._utils.attribution import GradientAttribution, LayerAttribution

from captum.attr import IntegratedGradients
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.attribution import GradientAttribution
from captum.attr._utils.batching import _batched_operator
from captum.attr._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_input_baseline,
    _is_tuple,
    _validate_input,
)
from captum.attr._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

class IntermediateGradients(IntegratedGradients):
    """
    Integrated Gradients is a model interpretability algorithm that assigns
    an importance score to each input feature by appproximating the integral
    of gradients between the model's output with respect to the inputs along
    the path between baselines and inputs.

    Intermediate Gradients is a modification of this algorithm that returns the
    gradients used to approximate the integral of gradients.
    These gradients can then be used for further attribution exploration.
    Returned from Intermediate Gradients is a tensor containing each of the
    gradients and a tensor of the step sizes used to calculate those gradients.
    """
    def attribute(
        self,
        inputs,
        baselines,
        additional_forward_args,
        n_steps,
        method,):
        """
        Get the intermediate gradients from the provided inputs and baselines.

        Parameters
        ----------
        inputs: torch.tensor(num_ids), dtye
            Input for which intermediate gradients are computed.
            This is the tensor that would be passed to the forward_func.
        baselines: torch.tensor(num_ids), d
            Baselines to define the starting point for gradient calculations.
            Should be the same length as inputs.
        additional_forward_args: 
            If the forward function takes any additional arguments,
            they can be provided here.  If there are multiple forward args,
            a tuple of the forward arguments can be provided.
        n_steps: int
            The number of steps used by the approximation method. Default: 50.
            The article suggests between 20 and 300 steps are enough to
            approximate the integral.
        method: str
            Method for determining step sizes for the gradients.
            One of `riemann_right`, `riemann_left`, `riemann_middle`,
            `riemann_trapezoid` or `gausslegendre`.

        Returns
        -------
        grads: torch.tensor(num_steps, num_ids, num_embeddings), dtype=float32
            Tensor of the gradients used in approximating the integrated gradients.
        step_sizes: torch.tensor(num_steps), dtype=float32
            Tensor of the step sizes used to calculate each of the gradients.
        """
        is_inputs_tuple = _is_tuple(inputs)

        inputs, baselines = _format_input_baseline(inputs, baselines)

        _validate_input(inputs, baselines, n_steps, method)

        # retrieve step size and scaling factor for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (1 x #steps x inputs[0].shape[1:], ...)
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (1 * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads = _batched_operator(
            self.gradient_func,
            scaled_features_tpl,
            input_additional_args,
            forward_fn=self.forward_func,
        )
        step_sizes = torch.tensor(step_sizes).view(n_steps, 1)
        return grads[0], step_sizes
