from sklearn import preprocessing
import pyro.distributions as dist
import torch


class Normal:
    def __init__(self, n_samples=1000, cov_shift=0.0, **kwargs):
        self.p_y0 = dist.Normal(torch.zeros((1,)), torch.ones((1,)))
        self.p_y1 = dist.Normal(torch.zeros((1,)) + cov_shift, torch.ones((1,)))
        self.n_samples = n_samples

    def get_data(self):
        Y0 = self.p_y0.sample((self.n_samples,))
        Y1 = self.p_y1.sample((self.n_samples,))
        return {'Y0': Y0, 'Y1': Y1}

