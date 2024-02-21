"""
These receptive fields are derived from scale-space theory, specifically in the paper `Normative theory of visual receptive fields by Lindeberg, 2021 <https://www.sciencedirect.com/science/article/pii/S2405844021000025>`_.

For use in spiking / binary signals, see the paper on `Translation and Scale Invariance for Event-Based Object tracking by Pedersen et al., 2023 <https://dl.acm.org/doi/10.1145/3584954.3584996>`_
"""

from typing import Callable, List, NamedTuple, Optional, Tuple, Type, Union

import torch

#from norse.torch.module.leaky_integrator_box import LIBoxCell, LIBoxParameters
from norse.torch.module.snn import SNNCell

from norse.torch.module.leaky_integrator_box import LIBoxCell, LIBoxParameters
from norse.torch.module.lif import LIFCell
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.snn import SNNCell
from norse.torch.functional.receptive_field import (
    spatial_receptive_fields_with_derivatives,
    temporal_scale_distribution,
    _extract_derivatives,
)


class SpatialReceptiveField2d(torch.nn.Module):
    """Creates a spatial receptive field as 2-dimensional convolutions.
    The parameters decide the number of combinations to scan over, i. e. the number of receptive fields to generate.
    Specifically, we generate ``n_scales * n_angles * (n_ratios - 1) + n_scales`` output_channels with aggregation,
    and ``in_channels * (n_scales * n_angles * (n_ratios - 1) + n_scales)`` without aggregation.

    The ``(n_ratios - 1) + n_scales`` terms exist because at ``ratio = 1``, fields are perfectly symmetrical, and there
    is therefore no reason to scan over the angles and scales for ``ratio = 1``.
    However, ``n_scales`` receptive field still needs to be added (one for each scale-space).

    Parameters:
        n_scales (int): Number of scaling combinations (the size of the receptive field) drawn from a logarithmic distribution
        n_angles (int): Number of angular combinations (the orientation of the receptive field)
        n_ratios (int): Number of eccentricity combinations (how "flat" the receptive field is)
        size (int): The size of the square kernel in pixels
        derivatives (Union[int, List[Tuple[int, int]]]): The number of derivatives to use in the receptive field.
        aggregate (bool): If True, sums the input channels over all output channels. If False, every
        output channel is mapped to every input channel, which may blow up in complexity.
        **kwargs: Arguments passed on to the underlying torch.nn.Conv2d
    """

    def __init__(
        self,
        in_channels: int,
        n_scales: int,
        n_angles: int,
        n_ratios: int,
        size: int,
        n_time_scales: int = 1,
        derivatives: Union[int, List[Tuple[int, int]]] = 0,
        min_scale: float = 0.2,
        max_scale: float = 1.5,
        min_ratio: float = 0.2,
        max_ratio: float = 1,
        aggregate: bool = True,
        device: str = "cpu",
        gradient = [False, False, False],
        **kwargs
    ) -> None:
        super().__init__()

        self.aggregate = aggregate
        self.size = size
        self.derivatives = derivatives
        self.in_channels = in_channels
        self.kwargs = kwargs
        self.device = device
        self.angles = torch.linspace(0, torch.pi - torch.pi / n_angles, n_angles)
        self.ratios = torch.linspace(min_ratio, max_ratio, n_ratios)
        self.log_scales = torch.linspace(min_scale, max_scale, n_scales)
        scales = torch.exp(self.log_scales)
        
        self.update = False
        derivative_list, self.derivative_max = _extract_derivatives(self.derivatives)
        gf_attr = [[s, a, r, d[0], d[1]] for s in scales for a in self.angles for r in self.ratios for d in derivative_list]
        gf_attr = torch.tensor(gf_attr, requires_grad=False).repeat(n_time_scales,1)
        self.scales = gf_attr[:,0]
        self.angles = gf_attr[:,1]
        self.ratios = gf_attr[:,2]
        self.rest = gf_attr[:,3:]
        if gradient[0]:
            self.scales.requires_grad_(True) 
        if gradient[1]:
            self.angles.requires_grad_(True) 
        if gradient[2]:
            self.ratios.requires_grad_(True) 
        # self.gf_attr = torch.tensor(gf_attr, requires_grad=True, device=self.device)
        self.gf_attr = torch.cat((self.scales.unsqueeze(1), self.angles.unsqueeze(1), self.ratios.unsqueeze(1), self.rest), dim=1)
        self.fields = spatial_receptive_fields_with_derivatives(
            self.gf_attr,
            self.derivative_max,
            self.size,
        )
        self.fields = self.fields.to(self.device)
        if self.aggregate:
            self.out_channels = self.fields.shape[0]
            weights = self.fields.unsqueeze(1).repeat(1, in_channels, 1, 1)
        else:
            self.out_channels = self.fields.shape[0] * in_channels
            empty_weights = torch.zeros(in_channels, self.fields.shape[0], size, size)
            weights = []
            for i in range(in_channels):
                in_weights = empty_weights.clone()
                in_weights[i] = self.fields
                weights.append(in_weights)
            weights = torch.concat(weights, 1).permute(1, 0, 2, 3)
        weights = weights.to(self.device)
        self.conv = torch.nn.Conv2d(in_channels, self.out_channels, size, **kwargs, bias=False,)
        self.conv.weight.data = self.conv.weight.data.to(self.device)
        self.conv.weight.requires_grad_(False)
        self.conv.weight[:] = weights[:]

    def forward(self, x: torch.Tensor):
        if self.update:
            self.update = False
            self.fields = spatial_receptive_fields_with_derivatives(
                self.gf_attr,
                self.derivative_max,
                self.size,
            )
            self.fields = self.fields.to(self.device)
            if self.aggregate:
                self.out_channels = self.fields.shape[0]
                weights = self.fields.unsqueeze(1).repeat(1, self.in_channels, 1, 1)
            else:
                self.out_channels = self.fields.shape[0] * self.in_channels
                empty_weights = torch.zeros(self.in_channels, self.fields.shape[0], self.size, self.size)
                weights = []
                for i in range(self.in_channels):
                    in_weights = empty_weights.clone()
                    in_weights[i] = self.fields
                    weights.append(in_weights)
                weights = torch.concat(weights, 1).permute(1, 0, 2, 3)
            weights = weights.to(self.device)
            self.conv = torch.nn.Conv2d(self.in_channels, self.out_channels, self.size, **self.kwargs, bias=False,)
            self.conv.weight.data = self.conv.weight.data.to(self.device)
            self.conv.weight.requires_grad_(False)
            self.conv.weight[:] = weights[:]
        
        return self.conv(x)


class TemporalReceptiveField(torch.nn.Module):
    """Creates ``n_scales`` temporal receptive fields for arbitrary n-dimensional inputs.
    The scale spaces are selected in a range of [min_scale, max_scale] using an exponential distribution, scattered using ``torch.linspace``.

    Parameters:
        shape (torch.Size): The shape of the incoming tensor, where the first dimension denote channels
        n_scales (int): The number of temporal scale spaces to iterate over.
        activation (SNNCell): The activation neuron. Defaults to LIBoxCell
        activation_state_map (Callable): A function that takes a tensor and provides a neuron parameter tuple.
            Required if activation is changed, since the default behaviour provides LIBoxParameters.
        min_scale (float): The minimum scale space. Defaults to 1.
        max_scale (Optional[float]): The maximum scale. Defaults to None. If set, c is ignored.
        c (Optional[float]): The base from which to generate scale values. Should be a value
            between 1 to 2, exclusive. Defaults to sqrt(2). Ignored if max_scale is set.
        time_constants (Optional[torch.Tensor]): Hardcoded time constants. Will overwrite the automatically generated, logarithmically distributed scales, if set. Defaults to None.
        dt (float): Neuron simulation timestep. Defaults to 0.001.
    """
    def __init__(
        self,
        shape: torch.Size,
        n_scales: int = 4,
        min_scale: float = 1,
        max_scale: Optional[float] = None,
        c: float = 1.41421,
        time_constants: Optional[torch.Tensor] = None,
        dt: float = 0.001,
        device: str = "cpu"
    ):
        super().__init__()
        if time_constants is None:
            taus = (1 / dt) / temporal_scale_distribution(
                n_scales, min_scale=min_scale, max_scale=max_scale, c=c
            )
            self.time_constants = torch.stack(
                [
                    torch.full(
                        [shape[0]//n_scales],
                        tau,
                        dtype=torch.float32,
                    )
                    for tau in taus
                ]
            )
            self.time_constants = self.time_constants.flatten()
        else:
            self.time_constants = time_constants
        self.n_scales = n_scales
        # self.time_constants = self.time_constants.repeat_interleave(shape[1])
        self.time_constants = self.time_constants.to(device)
        self.ps = torch.nn.Parameter(self.time_constants, requires_grad=True)
        self.channels = [LIFCell(
            p=LIFParameters(tau_mem_inv=self.ps[i])).to(device) for i in range(self.ps.shape[0])]
        
    def forward(self, x: torch.Tensor, state: Optional[NamedTuple] = None):
        x_repeated = torch.einsum('ijkl->jikl', x)
        if state == None:
            state = [None for _ in range(len(self.channels))]
        y, st = torch.einsum('ijkl->jikl', torch.stack([self.channels[i](x.unsqueeze(0), state[i])[0] for i, x in enumerate(x_repeated)]).squeeze(dim=1)), [self.channels[i](x.unsqueeze(0), state[i])[1] for i, x in enumerate(x_repeated)]
        return y, st



