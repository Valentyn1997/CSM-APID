import torch
from torch.distributions.distribution import Distribution
from torch.distributions import TransformedDistribution
import pyro.distributions as dist
import pyro.distributions.transforms as T


class ProjectionAugmentation(T.Transform):
    def __init__(self, inputs_size, proj_dist, return_aug=False):
        super(ProjectionAugmentation, self).__init__()
        self.inputs_size = inputs_size
        self.proj_dist = proj_dist
        self.return_aug = return_aug

        # Cache
        self._p_delta_cached = None
        self._y_cached, self._mu_cached, self._r_cached = None, None, None

    def _inverse(self, y):  # Augmentation
        assert self.inputs_size - 1 == y.shape[-1]
        self._mu_cached = self.proj_dist.g(y)

        self._y_cached = y
        self._p_delta_cached = dist.Normal(0.0, self.proj_dist.eps * torch.ones_like(self._mu_cached))  # Dirac delta measure

        if self.proj_dist.aug_mode == 's':  # Sampling based level set
            self._r_cached = self._mu_cached + self._p_delta_cached.sample()
            x = torch.cat([y, self._r_cached], dim=-1)
        elif self.proj_dist.aug_mode == 'q':  # Quantiles based level set
            self._r_cached = self._mu_cached + self._p_delta_cached.icdf(torch.linspace(0.0, 1.0, self.proj_dist.n_quantiles + 2)[1:-1])
            x = torch.cat([(y.unsqueeze(-2) + torch.zeros_like(self._r_cached).unsqueeze(-1)), self._r_cached.unsqueeze(-1)], dim=-1)
        else:
            raise NotImplementedError()
        return x

    def aug_log_prob(self, y):
        assert self.inputs_size - 1 == y.shape[-1]
        if (self._y_cached != y).any():
            self._inverse(y)
        return self._p_delta_cached.log_prob(self._r_cached - self._mu_cached).detach()

    def _call(self, x):  # Projection
        assert self.inputs_size == x.shape[-1]
        if not self.return_aug:
            return x[..., :-1]
        else:
            return x[..., :-1], x[..., -1]

    def clear_cache(self):
        self._p_delta_cached = None
        self._y_cached, self._mu_cached, self._r_cached = None, None, None


class ProjectedAugmentedDistribution(Distribution):

    def __init__(self, base_distribution: Distribution, g: torch.nn.Module, eps: float, aug_mode='s', n_quantiles=20,
                 validate_args=None):
        super(ProjectedAugmentedDistribution, self).__init__(base_distribution.batch_shape,
                                                             (base_distribution.event_shape[0] - 1, ),
                                                             validate_args=validate_args)
        self.base_distribution = base_distribution
        self.g = g  # g: event_shape - 1 -> 1
        self.aug_mode = aug_mode
        self.eps = eps
        self.n_quantiles = n_quantiles
        self.transforms = [ProjectionAugmentation(base_distribution.event_shape[0], self)]

    def clear_cache(self):
        self.transforms[0].clear_cache()

    def log_prob(self, value):  # Augmentation + log_prob calculation
        aug_value = self.transforms[0].inv(value)
        log_prob_r = self.transforms[0].aug_log_prob(value)
        if self.aug_mode == 's':  # Sampling based level set
            return self.base_distribution.log_prob(aug_value) - log_prob_r.squeeze(-1)
        elif self.aug_mode == 'q':  # Quantiles based level set
            return (self.base_distribution.log_prob(aug_value) - log_prob_r).nanmean(-1)
        else:
            raise NotImplementedError()

