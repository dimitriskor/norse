"""
These receptive fields are derived from scale-space theory, specifically in the paper `Normative theory of visual receptive fields by Lindeberg, 2021 <https://www.sciencedirect.com/science/article/pii/S2405844021000025>`_.

For use in spiking / binary signals, see the paper on `Translation and Scale Invariance for Event-Based Object tracking by Pedersen et al., 2023 <https://dl.acm.org/doi/10.1145/3584954.3584996>`_
"""

from typing import List, Tuple, Union, Optional

import torch


def gaussian_kernel(x, s, c):
    """
    Efficiently creates a 2d gaussian kernel.

    Arguments:
      x (torch.Tensor): A 2-d matrix
      s (float): The variance of the gaussian
      c (torch.Tensor): A 2x2 covariance matrix describing the eccentricity of the gaussian
    """
    ci = torch.linalg.inv(c)
    cd = torch.linalg.det(c)
    fraction = 1 / (2 * torch.pi * s * torch.sqrt(cd))
    b = torch.einsum("bimj,jk->bik", -x.unsqueeze(2), ci)
    a = torch.einsum("bij,bij->bi", b, x)
    return fraction * torch.exp(a / (2 * s))


def spatial_receptive_field(
    angle, ratio, size: int, scale: float = 2.5, dx: int = 0, dy: int = 0, domain: float = 8
):
    """
    Creates a (size x size) receptive field kernel

    Arguments:
      angle (float): The rotation of the kernel in radians
      ratio (float): The eccentricity as a ratio
      size (int): The size of the square kernel in pixels
      scale (float): The scale of the field. Defaults to 2.5
      domain (float): The initial coordinates from which the field is sampled. Defaults to 8 (equal to -8 to 8).
    """
    sm = torch.ones(2)
    sm[0] = scale
    sm[1] = scale*ratio
    a = torch.linspace(-domain, domain, size)
    r = torch.ones((2,2))
    r[0][0] = angle.cos()
    r[0][1] = angle.sin()
    r[1][0] = -angle.sin()
    r[1][1] = angle.cos()    
    c = (r * sm) @ (sm * r).T
    xs, ys = torch.meshgrid(a, a, indexing="xy")
    coo = torch.stack([xs, ys], dim=2)
    k = gaussian_kernel(coo, scale, c)
    k = _derived_field(k, (dx, dy))
    return k / k.sum()


def _extract_derivatives(
    derivatives: Union[int, List[Tuple[int, int]]]
) -> Tuple[List[Tuple[int, int]], int]:
    if isinstance(derivatives, int):
        if derivatives == 0:
            return [(0, 0)], 0
        else:
            return [
                (x, y) for x in range(derivatives + 1) for y in range(derivatives + 1)
            ], derivatives
    elif isinstance(derivatives, list):
        return derivatives, max([max(x, y) for (x, y) in derivatives])
    else:
        raise ValueError(
            f"Derivatives expected either a number or a list of tuples, but got {derivatives}"
        )


def _derived_field(
    field: torch.Tensor, derivatives: Tuple[int, int]
) -> torch.Tensor:
    out = []
    (dx, dy) = derivatives
    if dx == 0:
        fx = field
    else:
        fx = field.diff(
            dim=0, prepend=torch.zeros(dx, field.shape[1]), n=dx)

    if dy == 0:
        fy = fx
    else:
        fy = fx.diff(
            dim=1, prepend=torch.zeros(field.shape[0], dy), n=dy)
    out.append(fy)
    return torch.concat(out)


def spatial_receptive_fields_with_derivatives(
    gf_attr,
    derivative_max: int,
    size: int,
    min_scale: float = 0.2,
    max_scale: float = 1.5,
    min_ratio: float = 0.2,
    max_ratio: float = 1,
) -> torch.Tensor:
    r"""
    Creates a number of receptive field with 1st directional derivatives.
    The parameters decide the number of combinations to scan over, i. e. the number of receptive fields to generate.
    Specifically, we generate ``derivatives * (n_angles * n_scales * (n_ratios - 1) + n_scales)`` fields.
    The ``(n_ratios - 1) + n_scales`` terms exist because at ``ratio = 1``, fields are perfectly symmetrical, and there
    is therefore no reason to scan over the angles and scales for ``ratio = 1``.
    However, ``n_scales`` receptive fields still need to be added (one for each scale-space).
    Finally, the ``derivatives *`` term comes from the addition of spatial derivatives.
    Arguments:
        n_scales (int): Number of scaling combinations (the size of the receptive field) drawn from a logarithmic distribution
        n_angles (int): Number of angular combinations (the orientation of the receptive field)
        n_ratios (int): Number of eccentricity combinations (how "flat" the receptive field is)
        size (int): The size of the square kernel in pixels
        derivatives (Union[int, List[Tuple[int, int]]]): The spatial derivatives to include. Defaults to 0 (no derivatives).
            Can either be a number, in which case 1 + 2 ** n derivatives will be made (except when 0, see below).
              Example: `derivatives=0` omits derivatives
              Example: `derivatives=1` provides 2 spatial derivatives + 1 without derivation
            Or a list of tuples specifying the derivatives in both spatial dimensions
              Example: `derivatives=[(0, 0), (1, 2)]` provides two outputs, one without derivation and one :math:`\partial_x \partial^2_y`
    """

    def _stack_empty(x):
        if len(x) == 0:
            return torch.tensor([])
        else:
            return torch.stack(x)

    # We add extra space in both the domain and size to account for the derivatives
    domain = 8 + derivative_max * size * 0.5

    rings = _stack_empty(
        [
            spatial_receptive_field(attr[1], attr[2], size=size + 2 * derivative_max, scale=attr[0], dx=attr[3], dy=attr[4], domain=domain)
            for attr in gf_attr
        ]
    )
    derived_fields = rings[
        :,
        derivative_max : size + derivative_max,
        derivative_max : size + derivative_max,  # Remove extra space
    ]
    #derived_fields.sum().backward()
    #print(angles.grad)

    return derived_fields




def temporal_scale_distribution(
    n_scales: int,
    min_scale: float = 1,
    max_scale: Optional[float] = None,
    c: Optional[float] = 1.41421,
):
    r"""
    Provides temporal scales according to [Lindeberg2016].
    The scales will be logarithmic by default, but can be changed by providing other values for c.

    .. math:
        \tau_k = c^{2(k - K)} \tau_{max}
        \mu_k = \sqrt(\tau_k - \tau_{k - 1})

    Arguments:
      n_scales (int): Number of scales to generate
      min_scale (float): The minimum scale
      max_scale (Optional[float]): The maximum scale. Defaults to None. If set, c is ignored.
      c (Optional[float]): The base from which to generate scale values. Should be a value
        between 1 to 2, exclusive. Defaults to sqrt(2). Ignored if max_scale is set.

    .. [Lindeberg2016] Lindeberg 2016, Time-Causal and Time-Recursive Spatio-Temporal
        Receptive Fields, https://link.springer.com/article/10.1007/s10851-015-0613-9.
    """
    xs = torch.linspace(1, n_scales, n_scales)
    if max_scale is not None:
        if n_scales > 1:  # Avoid division by zero when having a single scale
            c = (min_scale / max_scale) ** (1 / (2 * (n_scales - 1)))
        else:
            return torch.tensor([min_scale]).sqrt()
    else:
        max_scale = (c ** (2 * (n_scales - 1))) * min_scale
    taus = c ** (2 * (xs - n_scales)) * max_scale
    return taus.sqrt()
