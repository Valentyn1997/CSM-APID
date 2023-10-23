import pyro.distributions as dist
import torch


class MultiModal:
    def __init__(self, n_samples=1000, **kwargs):
        weigths_dist = dist.OneHotCategorical(probs=torch.tensor([0.7, 0.3]))
        comp_dist = dist.Normal(torch.tensor([-0.5, 1.5]), torch.tensor([1.5, 0.5]))
        self.p_y0 = dist.MixtureSameFamily(weigths_dist._categorical, comp_dist)

        weigths_dist = dist.OneHotCategorical(probs=torch.tensor([0.3, 0.4, 0.3]))
        comp_dist = dist.Normal(torch.tensor([-2.5, 0.5, 2.0]), torch.tensor([0.35, 0.75, 0.5]))
        self.p_y1 = dist.MixtureSameFamily(weigths_dist._categorical, comp_dist)

        self.n_samples = n_samples

    def get_data(self):
        Y0 = self.p_y0.sample((self.n_samples,)).unsqueeze(-1)
        Y1 = self.p_y1.sample((self.n_samples,)).unsqueeze(-1)
        return {'Y0': Y0, 'Y1': Y1}
