import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class RetrievalTool:
    """
    Multi-view cosine-similarity retrieval with trend/seasonal decomposition.

    Constructs independent memory banks for raw / trend / seasonal views and
    retrieves Top-M candidates per view using mean-centered cosine similarity.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
        decomp_kernel=25,
        use_multiview=True,
        n_views=3,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        self.temperature = temperature
        self.topm = topm
        self.with_dec = with_dec
        self.return_key = return_key
        self.decomp_kernel = decomp_kernel
        self.use_multiview = use_multiview
        self.n_views = n_views
        self.view_names = ['raw', 'trend', 'season'][:self.n_views]

    def decompose_trend_seasonal(self, x):
        """
        Trend-seasonal decomposition via moving average.

        X_trend = MovingAvg(X, tau),  X_seasonal = X - X_trend
        """
        x_t = x.permute(0, 2, 1)
        tau = self.decomp_kernel
        moving_avg = nn.AvgPool1d(
            kernel_size=tau, stride=1,
            padding=(tau - 1) // 2, count_include_pad=False)
        x_trend_t = moving_avg(x_t)
        x_trend = x_trend_t.permute(0, 2, 1)
        x_seasonal = x - x_trend
        return x_trend, x_seasonal

    def _decompose_future(self, y_data):
        """Apply trend-seasonal decomposition to future labels."""
        return self.decompose_trend_seasonal(y_data)

    def build_multiview_library(self, train_data):
        """Build raw / trend / seasonal memory banks from training data."""
        train_data_all, y_data_all = [], []
        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[1])
            y_data_all.append(
                td[2][-(train_data.pred_len + train_data.label_len):] if self.with_dec
                else td[2][-train_data.pred_len:])

        self.x_all = torch.tensor(np.stack(train_data_all, axis=0)).float()
        self.y_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        self.n_train = self.x_all.shape[0]

        self.libs = {}
        if 'raw' in self.view_names:
            self.libs['raw'] = (self.x_all, self.y_all)
        if 'trend' in self.view_names:
            x_trend, _ = self.decompose_trend_seasonal(self.x_all)
            y_trend, _ = self._decompose_future(self.y_all)
            self.libs['trend'] = (x_trend, y_trend)
        if 'season' in self.view_names:
            _, x_seasonal = self.decompose_trend_seasonal(self.x_all)
            _, y_seasonal = self._decompose_future(self.y_all)
            self.libs['season'] = (x_seasonal, y_seasonal)

    def _retrieve_single_view(self, x, index, view_name, train=True):
        """
        Retrieve Top-M candidates from a single view using cosine similarity.

        Args:
            x: (B, T, V) queries
            index: (B,) sample indices (for self-masking)
            view_name: 'raw' / 'trend' / 'season'
            train: whether to apply self-masking

        Returns:
            topm_index: (B, topm) indices into library
            topm_values: (B, topm, P, V) future segments
        """
        K_lib, V_lib = self.libs[view_name]
        n_candidates = K_lib.shape[0]

        if view_name == 'trend':
            x_view, _ = self.decompose_trend_seasonal(x)
        elif view_name == 'season':
            _, x_view = self.decompose_trend_seasonal(x)
        else:
            x_view = x

        bsz, seq_len, channels = x_view.shape
        assert seq_len == self.seq_len and channels == self.channels

        x_flat = x_view.reshape(bsz, -1)
        K_flat = K_lib.reshape(n_candidates, -1).to(x.device)

        # Mean-centered cosine similarity
        x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
        K_centered = K_flat - K_flat.mean(dim=1, keepdim=True)
        x_norm = F.normalize(x_centered, dim=1)
        K_norm = F.normalize(K_centered, dim=1)
        sim = torch.mm(x_norm, K_norm.t())

        if train:
            index = index.to(x.device)
            bsz_idx = len(index)
            sliding_index = torch.arange(
                2 * (self.seq_len + self.pred_len) - 1, device=x.device)
            sliding_index = sliding_index.unsqueeze(0).repeat(bsz_idx, 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(1)
            self_mask = torch.zeros((bsz_idx, n_candidates), device=x.device)
            safe_index = sliding_index.clamp(0, n_candidates - 1)
            self_mask.scatter_(1, safe_index, 1., reduce='add')
            sim = sim.masked_fill(self_mask.bool(), float('-inf'))

        topm_sim, topm_index = torch.topk(sim, self.topm, dim=1)
        topm_values = V_lib[topm_index.cpu()].to(x.device)
        return topm_index, topm_values

    def retrieve_multiview(self, x, index, train=True):
        """Retrieve Top-M from all views in parallel."""
        assert hasattr(self, 'libs') and len(self.libs) > 0
        results = {}
        for view_name in self.view_names:
            topm_index, topm_values = self._retrieve_single_view(
                x, index, view_name, train=train)
            results[view_name] = (topm_index, topm_values)
        return results

    def retrieve_all_multiview(self, data, train=False, device=torch.device('cpu')):
        """
        Precompute retrieval results for an entire dataset.

        Returns:
            dict of {view_name: (index_table, value_table)}
        """
        assert hasattr(self, 'libs') and len(self.libs) > 0
        rt_loader = DataLoader(data, batch_size=1024, shuffle=False,
                               num_workers=0, drop_last=False)
        index_tables = {v: [] for v in self.view_names}
        value_tables = {v: [] for v in self.view_names}

        with torch.no_grad():
            for index, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(rt_loader):
                results = self.retrieve_multiview(
                    batch_x.float().to(device), index, train=train)
                for view_name in self.view_names:
                    topm_index, topm_values = results[view_name]
                    index_tables[view_name].append(topm_index.cpu())
                    value_tables[view_name].append(topm_values.cpu())

        multiview_results = {}
        for view_name in self.view_names:
            idx_table = torch.cat(index_tables[view_name], dim=0)
            val_table = torch.cat(value_tables[view_name], dim=0)
            multiview_results[view_name] = (idx_table, val_table)
        return multiview_results
