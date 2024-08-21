import torch
from torch.distributions import constraints
import logging
from pyro.distributions.torch_transform import TransformModule

from normflows.flows.residual import iResBlock
from normflows.nets import LipschitzMLP

logger = logging.getLogger(__name__)


def new_inverse_fixed_point(self, y, atol=1e-5, rtol=1e-5):
    x, x_prev = y - self.nnet(y), y
    i = 0
    tol = atol + y.abs() * rtol
    while not torch.all((x - x_prev) ** 2 / tol < 1):
        x, x_prev = y - self.nnet(x), x
        i += 1
        if i > 200:
            logger.warning('Fixed point divergence!')
            break
    return x


iResBlock._inverse_fixed_point = new_inverse_fixed_point


class ResidualTransform(TransformModule):
    """
    Behrmann et al., 2018, "Residual Flows for Invertible Generative Modeling,
    """

    @property
    def sign(self):
        raise NotImplementedError   # sign is not defined for multidimensional transformations

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim: int, hidden_dim: int, exact_trace=True, lipschitz_const=0.9, cache_size=0, reduce_memory=True,
                 atol=1e-5, rtol=1e-5):
        super(ResidualTransform, self).__init__(cache_size=cache_size)
        self.input_dim = input_dim
        self.lnet = LipschitzMLP([self.input_dim, hidden_dim, self.input_dim],
                                 init_zeros=exact_trace, lipschitz_const=lipschitz_const)
        self.res_block = iResBlock(self.lnet, neumann_grad=reduce_memory, grad_in_forward=reduce_memory,
                                   exact_trace=exact_trace, brute_force=True)
        self.atol = atol
        self.rtol = rtol

    def _inverse(self, y):
        return self.res_block.forward(y)

    def log_abs_det_jacobian(self, x, y):
        _, log_abs_det = self.res_block.forward(y, 0.0)
        return log_abs_det.squeeze(-1)

    def _call(self, x):
        return self.res_block._inverse_fixed_point(x, atol=self.atol, rtol=self.rtol)
