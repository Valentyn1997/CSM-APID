import torch
from omegaconf import DictConfig
from torch.distributions import TransformedDistribution, Independent
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import DenseNN
from pytorch_lightning.loggers import MLFlowLogger
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from ot import wasserstein_1d
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage

from src.features.projection import ProjectedAugmentedDistribution, ProjectionAugmentation
from src.features.residual import ResidualTransform


class ConditionalAPID(torch.nn.Module):

    def __init__(self, args: DictConfig = None, **kwargs):
        super(ConditionalAPID, self).__init__()

        # ConditionalAPID basic parameters
        self.dim_u = args.model.dim_u
        self.n_trans = args.model.n_trans
        self.hid_dim = 5

        # Projection-Augmentation parameters
        self.aug_mode = args.model.aug_mode
        self.n_quantiles = args.model.n_quantiles
        self.eps = args.model.eps

        # Base distributions
        self.flows = []
        self.base_unif = Independent(dist.Uniform(torch.zeros(self.dim_u), torch.ones(self.dim_u)), -1)

        # Init of transformations u -> y
        self.gs, self.proj_dists, self.trans = [], [], []

        for i in range(self.dim_u):
            trans = []
            if i == 0:  # unit cube support -> real support
                trans += [T.SigmoidTransform().inv]
            if i < self.dim_u - 1:  # dim in [2, 3, ..., dim_u]
                trans += [ResidualTransform(self.dim_u - i, self.hid_dim * self.dim_u, atol=args.model.tol, rtol=args.model.tol) for _ in range(self.n_trans)]
            else:  # dim == 1 (last dimension)
                trans += [ResidualTransform(self.dim_u - i, self.hid_dim * self.dim_u, atol=args.model.tol, rtol=args.model.tol)]

            self.trans.extend(trans)

            if i == 0:
                self.flows.append(TransformedDistribution(self.base_unif, trans))
            else:
                self.flows.append(TransformedDistribution(self.flows[-1], trans))

            if i < self.dim_u - 1:  # projection <-> augmentation
                self.gs.append(DenseNN(self.dim_u - i - 1, [self.hid_dim * self.dim_u], param_dims=[1]).float())
                proj_dist = ProjectedAugmentedDistribution(self.flows[-1], self.gs[-1], eps=self.eps, aug_mode=self.aug_mode,
                                                           n_quantiles=self.n_quantiles)
                self.flows.append(proj_dist)
                self.proj_dists.append(proj_dist)

        self.gs = torch.nn.ModuleList(self.gs)
        self.trans = torch.nn.ModuleList([t for t in self.trans if isinstance(t, torch.nn.Module)])


    def log_prob(self, y):
        # Resetting proj_dists
        for proj_dist in self.proj_dists:
            proj_dist.aug_mode, proj_dist.eps, proj_dist.n_quantiles = self.aug_mode, self.eps, self.n_quantiles
        return self.flows[-1].log_prob(y)

    def clear_cache(self):
        for f in self.flows:
            f.clear_cache()

    def forward(self, u, return_aug=False):
        x = u
        x_augs = []
        for f in self.flows:
            for t in f.transforms:
                if not isinstance(t, ProjectionAugmentation) or not return_aug:
                    x = t(x)
                else:
                    t.return_aug = True
                    x, x_aug = t(x)
                    x_augs.append(x_aug)
                    t.return_aug = False
        if len(x_augs) > 0:
            x = torch.cat([x, torch.stack(x_augs, -1)], -1)
        return x

    def backward(self, y, eps=None, aug_mode=None, n_quantiles=None):
        # Resetting proj_dists
        eps = self.eps if eps is None else eps
        aug_mode = self.aug_mode if aug_mode is None else aug_mode
        n_quantiles = self.n_quantiles if n_quantiles is None else n_quantiles
        for proj_dist in self.proj_dists:
            proj_dist.aug_mode, proj_dist.eps, proj_dist.n_quantiles = aug_mode, eps, n_quantiles

        x = y
        for f in reversed(self.flows):
            for t in reversed(f.transforms):
                x = t.inv(x)
        return x

    def max_abs_principal_curvature(self, u):
        assert self.dim_u == 2

        u = u.reshape(-1, self.dim_u)
        u.requires_grad_(True)

        def unit_normal(u):
            _, grad = torch.autograd.functional.vjp(self.forward, u, v=torch.ones((u.shape[0], 1)), create_graph=True)
            return grad / grad.norm(dim=1, keepdim=True)

        unit_normal_grad = torch.autograd.functional.jacobian(unit_normal, u, create_graph=True)
        H = 0.5 * unit_normal_grad.diagonal(dim1=1, dim2=3).diagonal(dim1=0, dim2=1).sum(0)  # Curvature
        return H.abs()


class APID(torch.nn.Module):
    def __init__(self, args: DictConfig = None, **kwargs):
        super(APID, self).__init__()
        self.apids = torch.nn.ModuleList([ConditionalAPID(args), ConditionalAPID(args)])  # treatments 0 and 1
        self.max_apids, self.min_apids = [], []

        # Model hyparams
        self.hparams = args

        # Common latent noise variables
        self.p_u = self.apids[0].base_unif

        # Training hparams
        self.burn_in_epochs = args.model.burn_in_epochs
        self.q_epochs = args.model.q_epochs
        self.curv_epochs = args.model.curv_epochs
        self.lr = args.model.lr
        self.noise_std = args.model.noise_std
        self.batch_size = args.model.batch_size
        self.n_quantiles = args.model.n_quantiles
        self.q_coeff = args.model.q_coeff
        self.curv_coeff = args.model.curv_coeff
        self.cf_only = args.model.cf_only
        self.ema_q_beta = args.model.ema_q

        self.device = args.exp.device
        self.to(self.device)

        # MlFlow Logger
        if args.exp.logging:
            experiment_name = f'{args.model.name}/{args.dataset.name}'
            self.mlflow_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri)

    def get_optimizer(self, params):
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=0.0)

    def get_train_dataloader(self, train_data_dict: dict, batch_size):
        training_data0, training_data1 = TensorDataset(train_data_dict['Y0']), TensorDataset(train_data_dict['Y1'])
        return DataLoader(training_data0, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=self.device)), \
            DataLoader(training_data1, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=self.device))

    def plot_forward(self, Y, apid):
        u = self.p_u.sample((Y.shape[0],))
        _, ax = plt.subplots()
        sns.distplot(Y.cpu().numpy(), ax=ax)
        sns.distplot(apid.forward(u).detach().cpu().numpy(), ax=ax)
        plt.show()

    def plot_level_sets(self, apids, t_f, t_cf, y_f, values, task):
        fig = plt.figure(figsize=(10, 10))
        with torch.no_grad():
            x1 = apids[t_f].backward(y_f, aug_mode='q', n_quantiles=100).cpu().numpy()
            plt.plot(x1[0, :, 0], x1[0, :, 1], s=1, c='tab:orange')
            for q in values:
                x2 = apids[t_cf].backward(q * torch.ones((1, 1)), aug_mode='q', n_quantiles=100).cpu().numpy()
                # c2 = p2.mean_curvature(torch.tensor(x2)).numpy()
                plt.plot(x2[0, :, 0], x2[0, :, 1], s=1, c='tab:blue')
        plt.title(task)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.show()

    def linear_rise(self, step, max_steps, max_value):
        p = float(step + 1) / float(max_steps)
        return p * max_value

    def fit(self, train_data_dict: dict, f_dict: dict, log: bool):
        train_dataloaders = self.get_train_dataloader(train_data_dict, self.batch_size)
        y_f, t_f, t_cf = f_dict['Y_f'], f_dict['T_f'], 1 - f_dict['T_f']
        support = [(train_data_dict['Y0'].min(), train_data_dict['Y0'].max()), (train_data_dict['Y1'].min(), train_data_dict['Y1'].max())]
        cf_support = torch.tensor(support[t_cf])
        task_non_inf = {'min': False, 'max': False}

        # Logging
        self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        # Burn-in for both models
        optimizers = [self.get_optimizer(apid.parameters()) for apid in self.apids]
        for step in tqdm(range(self.burn_in_epochs)):
            ys = torch.tensor(next(iter(train_dataloaders[0]))[0]), torch.tensor(next(iter(train_dataloaders[1]))[0])

            [optimizer.zero_grad() for optimizer in optimizers]
            for i, (y, optimizer, apid) in enumerate(zip(ys, optimizers, self.apids)):
                # Fitting distribution
                # Log-prob losses
                y_noisy = y + self.noise_std * torch.randn_like(y)
                log_prob = - apid.log_prob(y_noisy).mean()
                # Wasserstein losses
                u = self.p_u.sample((self.batch_size,))
                wd = wasserstein_1d(y, apid.forward(u)).mean()
                loss = log_prob + wd

                loss.backward()
                optimizer.step()
                apid.clear_cache()

                if step % 5 == 0 and log:
                    self.mlflow_logger.log_metrics({f'train_neg_log_prob{i}': log_prob.item()}, step=step)
                    self.mlflow_logger.log_metrics({f'train_wd{i}': wd.item()}, step=step)

        self.plot_forward(train_data_dict['Y0'], self.apids[0])
        self.plot_forward(train_data_dict['Y1'], self.apids[1])

        self.max_apids, self.min_apids = deepcopy(self.apids), deepcopy(self.apids)
        max_optimizers = [self.get_optimizer(apid.parameters()) for apid in self.max_apids]
        min_optimizers = [self.get_optimizer(apid.parameters()) for apid in self.min_apids]

        self.ema_q = ExponentialMovingAverage([p for apid in self.max_apids + self.min_apids for p in apid.parameters()],
                                              decay=self.ema_q_beta)

        for step in tqdm(range(self.q_epochs + self.curv_epochs)):
            ys = torch.tensor(next(iter(train_dataloaders[0]))[0]), torch.tensor(next(iter(train_dataloaders[1]))[0])
            [[optimizer.zero_grad() for optimizer in optimizers] for optimizers in [self.max_apids, self.min_apids]]

            for task, optimizers, apids in zip(['max', 'min'], [max_optimizers, min_optimizers], [self.max_apids, self.min_apids]):
                if task_non_inf[task] and step < self.q_epochs:
                    continue

                loss = 0.0

                # Max/Min ECOU(ECOT)
                # y_f_noisy = (y_f + self.noise_std * torch.randn((self.batch_size, 1)))
                # f_level_set = apids[t_f].backward(y_f + torch.zeros((self.batch_size, 1)))
                f_level_set = apids[t_f].backward(y_f, aug_mode='q', n_quantiles=self.n_quantiles)
                if self.cf_only:
                    f_level_set = f_level_set.detach()
                q = apids[t_cf].forward(f_level_set).mean()

                with torch.no_grad():
                    with self.ema_q.average_parameters():
                        # f_level_set = apids[t_f].backward(y_f + torch.zeros((4 * self.batch_size, 1)))
                        f_level_set = apids[t_f].backward(y_f, aug_mode='q', n_quantiles=4 * self.n_quantiles)
                        q_smooth = apids[t_cf].forward(f_level_set).mean()
                    task_non_inf[task] = (q < cf_support[0]) or (q > cf_support[1])

                if step % 5 == 0 and log:
                    self.mlflow_logger.log_metrics({f'q_{task}': q.item()}, step=step + self.burn_in_epochs)
                    self.mlflow_logger.log_metrics({f'q_smooth_{task}': q_smooth.item()}, step=step + self.burn_in_epochs)

                if task == 'max' and not task_non_inf[task]:
                    loss += self.q_coeff * torch.nn.functional.softplus(-q)
                elif not task_non_inf[task]:
                    loss += self.q_coeff * torch.nn.functional.softplus(q)

                # Fitting distribution
                for i, (y, apid) in enumerate(zip(ys, apids)):
                    if self.cf_only and i == t_f:
                        continue
                    # Log-prob losses
                    y_noisy = y + self.noise_std * torch.randn_like(y)
                    log_prob = - apid.log_prob(y_noisy).mean()
                    # Wasserstein losses
                    u = self.p_u.sample((self.batch_size,))
                    wd = wasserstein_1d(y, apid.forward(u)).mean()
                    loss += log_prob + wd

                    if ((step % 5 == 0) or step == self.q_epochs + self.curv_epochs - 1) and log:
                        self.mlflow_logger.log_metrics({f'train_neg_log_prob{i}_{task}': log_prob.item()},
                                                       step=step + self.burn_in_epochs)
                        self.mlflow_logger.log_metrics({f'train_wd{i}_{task}': wd.item()}, step=step + self.burn_in_epochs)

                # # Penalizing curvature
                if self.curv_coeff > 0.0 and (step >= self.q_epochs):
                    # mean_cf_level_set = apids[t_cf].backward(q.detach() + torch.zeros((self.batch_size, 1)))
                    mean_cf_level_set = apids[t_cf].backward(q.detach(), aug_mode='q', n_quantiles=self.n_quantiles)
                    cf_curv = apids[t_cf].max_abs_principal_curvature(mean_cf_level_set).mean()
                    curv_coeff = self.linear_rise(step - self.q_epochs, self.curv_epochs, self.curv_coeff)
                    loss += curv_coeff * cf_curv
                else:
                    cf_curv = torch.tensor(0.0)

                loss.backward()
                [optimizer.step() for optimizer in optimizers]
                [apid.clear_cache() for apid in apids]

                if step % 5 == 0 and log:
                    self.mlflow_logger.log_metrics({f'cf_curv_{task}': cf_curv.item()}, step=step + self.burn_in_epochs)
                    self.mlflow_logger.log_metrics({f'loss_{task}': loss.item()}, step=step + self.burn_in_epochs)

                # if step % 50 == 0 and log:
                #     self.plot_level_sets(apids, t_f, t_cf, y_f, [0.0, 1.5, 2.0] if task == 'max' else [-0.0, -1.5, -2.0], task)

            self.ema_q.update()


if __name__ == '__main__':
    args = DictConfig({'model': {'dim_u': 2, 'n_trans': 1, 'aug_mode': 's', 'n_quantiles': 20, 'eps': 0.5}})
    p1 = ConditionalAPID(args)
    p1.log_prob(torch.tensor([[0.1], [1.0]]))
    p1.log_prob(torch.zeros((100, 1))).mean().exp()
    p1.max_abs_principal_curvature(p1.backward(torch.tensor([[0.1], [1.0]]), aug_mode='q'))
    p1.max_abs_principal_curvature(torch.tensor([[0.1, 0.2], [0.5, 0.5], [0.5, 0.5]]))

    p_y = dist.Normal(torch.zeros((1, )) + 0.0, torch.ones((1, )))
    y = p_y.sample((1000, ))
    p1.log_prob(y)
