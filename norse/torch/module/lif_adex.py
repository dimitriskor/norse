from typing import Optional, Tuple

import numpy as np
import torch

from norse.torch.functional.lif_adex import (
    LIFAdExState,
    LIFAdExFeedForwardState,
    LIFAdExParameters,
    lif_adex_step,
    lif_adex_feed_forward_step,
)

from norse.torch.module.util import remove_autopses


class LIFAdExCell(torch.nn.Module):
    r"""Computes a single euler-integration step of a recurrent adaptive exponential
    LIF neuron-model adapted from
    http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model.
    More specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} \left(v_{\text{leak}} - v + i + \Delta_T exp\left({{v - v_{\text{th}}} \over {\Delta_T}}\right)\right) \\
            \dot{i} &= -1/\tau_{\text{syn}} i \\
            \dot{a} &= 1/\tau_{\text{ada}} \left( a_{current} (V - v_{\text{leak}}) - a \right)
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + w_{\text{input}} z_{\text{in}} \\
            i &= i + w_{\text{rec}} z_{\text{rec}}
        \end{align*}

    where :math:`z_{\text{rec}}` and :math:`z_{\text{in}}` are the
    recurrent and input spikes respectively.

    Parameters:
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden state.
        p (LIFExParameters): Parameters of the LIF neuron model.
        dt (float): Time step to use.
        autopses (bool): Allow self-connections in the recurrence? Defaults to False.

    Examples:

        >>> batch_size = 16
        >>> lif_ex = LIFAdExCell(10, 20)
        >>> input = torch.randn(batch_size, 10)
        >>> output, s0 = lif_ex(input)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFAdExParameters = LIFAdExParameters(),
        dt: float = 0.001,
        autopses: bool = False,
    ):
        super(LIFAdExCell, self).__init__()
        self.input_weights = torch.nn.Parameter(
            torch.randn(hidden_size, input_size) * np.sqrt(2 / hidden_size)
        )
        recurrent_weights = torch.randn(hidden_size, hidden_size) * np.sqrt(
            2 / hidden_size
        )
        self.recurrent_weights = torch.nn.Parameter(
            recurrent_weights if autopses else remove_autopses(recurrent_weights)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p = p
        self.dt = dt

    def extra_repr(self):
        s = f"{self.input_size}, {self.hidden_size}, p={self.p}, dt={self.dt}"
        return s

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[LIFAdExState] = None
    ) -> Tuple[torch.Tensor, LIFAdExState]:
        if state is None:
            state = LIFAdExState(
                z=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                v=self.p.v_leak.detach(),
                i=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                a=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            )
            state.v.requires_grad = True
        return lif_adex_step(
            input_tensor,
            state,
            self.input_weights,
            self.recurrent_weights,
            p=self.p,
            dt=self.dt,
        )


class LIFAdExLayer(torch.nn.Module):
    r"""A neuron layer that wraps a recurrent LIFAdExCell in time such
    that the layer keeps track of temporal sequences of spikes.
    After application, the layer returns a tuple containing
      (spikes from all timesteps, state from the last timestep).


    Example:
        >>> data = torch.zeros(10, 5, 2) # 10 timesteps, 5 batches, 2 neurons
        >>> l = LIFAdExLayer(2, 4)
        >>> l(data) # Returns tuple of (Tensor(10, 5, 4), LIFExState)
    """

    def __init__(self, *cell_args, **kw_args):
        super(LIFAdExLayer, self).__init__()
        self.cell = LIFAdExCell(*cell_args, **kw_args)

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[LIFAdExState] = None
    ) -> Tuple[torch.Tensor, LIFAdExState]:
        inputs = input_tensor.unbind(0)
        outputs = []  # torch.jit.annotate(List[torch.Tensor], [])
        for _, input_step in enumerate(inputs):
            out, state = self.cell(input_step, state)
            outputs += [out]
        # pytype: disable=bad-return-type
        return torch.stack(outputs), state
        # pytype: enable=bad-return-type


class LIFAdExFeedForwardCell(torch.nn.Module):
    r"""Computes a single euler-integration step of a feed-forward exponential
    LIF neuron-model adapted from
    http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model.
    It takes as input the input current as generated by an arbitrary torch
    module or function. More specifically it implements one integration step
    of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} \left(v_{\text{leak}} - v + i + \Delta_T exp\left({{v - v_{\text{th}}} \over {\Delta_T}}\right)\right) \\
            \dot{i} &= -1/\tau_{\text{syn}} i \\
            \dot{a} &= 1/\tau_{\text{ada}} \left( a_{current} (V - v_{\text{leak}}) - a \right)
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        i = i + i_{\text{in}}

    where :math:`i_{\text{in}}` is meant to be the result of applying
    an arbitrary pytorch module (such as a convolution) to input spikes.

    Parameters:
        p (LIFExParameters): Parameters of the LIFEx neuron model.
        dt (float): Time step to use.

    Examples:

        >>> batch_size = 16
        >>> lif_ex = LIFExFeedForwardCell()
        >>> data = torch.randn(batch_size, 20, 30)
        >>> output, s0 = lif_ex(data)
    """

    def __init__(self, p: LIFAdExParameters = LIFAdExParameters(), dt: float = 0.001):
        super(LIFAdExFeedForwardCell, self).__init__()
        self.p = p
        self.dt = dt

    def extra_repr(self):
        s = f"p={self.p}, dt={self.dt}"
        return s

    def forward(
        self, x: torch.Tensor, state: Optional[LIFAdExFeedForwardState] = None
    ) -> Tuple[torch.Tensor, LIFAdExFeedForwardState]:
        if state is None:
            state = LIFAdExFeedForwardState(
                v=self.p.v_leak.detach(),
                i=torch.zeros(
                    *x.shape,
                    device=x.device,
                    dtype=x.dtype,
                ),
                a=torch.zeros(
                    *x.shape,
                    device=x.device,
                    dtype=x.dtype,
                ),
            )
            state.v.requires_grad = True
        return lif_adex_feed_forward_step(x, state, p=self.p, dt=self.dt)
