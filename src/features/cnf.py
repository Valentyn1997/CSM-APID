import torchdyn
from torchdyn.models import NeuralODE, REQUIRES_NOISE
from torchdyn.nn import Augmenter
from pyro.nn import DenseNN
from torch.distributions import constraints
import torch

from pyro.distributions.torch_transform import TransformModule


def autograd_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    for i in range(x_in.shape[-1]):
        trJ += torch.autograd.grad(x_out[..., i].sum(), x_in, allow_unused=False, create_graph=True)[0][..., i]
    return trJ


class CNF(torch.nn.Module):
    def __init__(self, net, trace_estimator=None, noise_dist=None):
        super().__init__()
        self.net = net
        self.trace_estimator = trace_estimator if trace_estimator is not None else autograd_trace
        self.noise_dist, self.noise = noise_dist, None
        if self.trace_estimator in REQUIRES_NOISE:
            assert self.noise_dist is not None, 'This type of trace estimator requires specification of a noise distribution'

    def forward(self, t, x):
        with torch.set_grad_enabled(True):
            x_in = torch.autograd.Variable(x[..., 1:], requires_grad=True).to(x)  # 1st dim reserved for divergence propagation
            # the neural network will handle the data-dynamics here
            x_out = self.net(t, x_in)

            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return torch.cat([-trJ[..., None], x_out], -1) + 0 * x
        # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph


class VectorField(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VectorField, self).__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, hidden_dim),
            # torch.nn.Softplus(),
            # torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t, x):
        x = self.f(torch.cat([x, torch.zeros_like(x[..., :1]) + t], -1))
        return x


class CNFTransform(TransformModule):

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim: int, hidden_dim: int, cache_size=0):
        super(CNFTransform, self).__init__(cache_size=cache_size)
        # self.f = DenseNN(input_dim, [hidden_dim], param_dims=[input_dim])
        self.cnf = CNF(VectorField(input_dim, hidden_dim))
        self.nde = NeuralODE(self.cnf, solver='dopri5', atol=1e-3, rtol=1e-3, return_t_eval=False)
        self.augmenter = Augmenter(augment_idx=-1, augment_dims=1)

        self._y_cached, self._log_abs_det_jacobian_cached = None, None

    def _inverse(self, y):
        self._y_cached = y
        y = self.augmenter(y)
        sol = self.nde(y, t_span=torch.tensor([0.0, 1.0]))
        self._log_abs_det_jacobian_cached = sol[1][..., 0]
        return sol[1][..., 1:]

    def log_abs_det_jacobian(self, x, y):
        if (self._y_cached != y).any():
            self._inverse(y)
        return self._log_abs_det_jacobian_cached

    def _call(self, x):
        x = self.augmenter(x)
        sol = self.nde(x, t_span=torch.tensor([1.0, 0.0]))
        return sol[1][..., 1:]
