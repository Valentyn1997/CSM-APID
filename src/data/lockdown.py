import torch
import numpy as np
import pandas as pd

from src import ROOT_PATH


class LockdownData:
    def __init__(self, **kwargs):
        self.df = pd.read_csv(f'{ROOT_PATH}/data/lockdown/data_preprocessed.csv')

        # Relevant columns
        self.df = self.df[['country', 'date', 'lockdown', 'incidence_rate', 'cases']].dropna()
        self.df['year'] = pd.DatetimeIndex(self.df['date']).year
        self.df['month'] = pd.DatetimeIndex(self.df['date']).month
        self.df['week'] = pd.DatetimeIndex(self.df['date']).week

        # Filtering only days with (cumulative) cases > 50
        self.df = self.df[(self.df['cases'] > 50) & (self.df['incidence_rate'] > 0.0)]

        # Averaging on a weekly-basis
        self.df_week = self.df.groupby(['country', 'week']).mean().reset_index()
        self.df_week['lockdown_bin'] = (self.df_week['lockdown'] > 0.5).astype(int)
        self.df_week['log_incidence_rate'] = np.log(self.df_week['incidence_rate'])
        self.mean_log_incidence_rate = self.df_week['log_incidence_rate'].mean()

        self.n_samples = len(self.df_week)

    def get_data(self):
        Y0 = torch.tensor((self.df_week[self.df_week['lockdown_bin'] == 0].log_incidence_rate - self.mean_log_incidence_rate).values, dtype=torch.float32)
        Y1 = torch.tensor((self.df_week[self.df_week['lockdown_bin'] == 1].log_incidence_rate - self.mean_log_incidence_rate).values, dtype=torch.float32)
        return {'Y0': Y0.reshape(-1, 1), 'Y1': Y1.reshape(-1, 1)}
