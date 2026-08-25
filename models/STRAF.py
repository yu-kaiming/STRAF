import torch
import torch.nn as nn
from layers.Retrieval import RetrievalTool


class Model(nn.Module):
    """
    STRAF: Structure-Aware Retrieval-Augmented Time Series Forecasting.

    Combines a direct linear predictor with multi-view retrieval-augmented prediction,
    fused via Reliability-aware Retrieval Fusion (RRF) -- a two-stage attention
    mechanism across raw / trend / seasonal views.
    """

    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.device = torch.device(f'cuda:{configs.gpu}')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len if self.task_name not in['long_term_forecast', 'short_term_forecast'] else configs.seq_len
        self.channels = configs.enc_in

        # Direct linear predictor (baseline path)
        self.linear_x = nn.Linear(self.seq_len, self.pred_len)

        self.n_period = configs.n_period
        self.topm = configs.topm

        # Retrieval module
        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
            decomp_kernel=getattr(configs, 'decomp_kernel', 25),
            use_multiview=getattr(configs, 'use_multiview', True),
            n_views=getattr(configs, 'n_views', 3),
        )
        self.view_names = self.rt.view_names
        self.n_views = len(self.view_names)

        # RRF parameters
        self.d_model = self.pred_len
        self.n_heads = getattr(configs, 'rrf_n_heads', 4)
        self.use_rrf = getattr(configs, 'use_reliability_fusion', True)

        # Query projection (Step 1 of RRF)
        self.linear_q = nn.Linear(self.seq_len, self.pred_len)

        # Value projections per view (Step 2 of RRF)
        self.linear_v_raw = nn.Linear(self.pred_len, self.pred_len)
        if 'trend' in self.view_names:
            self.linear_v_trend = nn.Linear(self.pred_len, self.pred_len)
        if 'season' in self.view_names:
            self.linear_v_season = nn.Linear(self.pred_len, self.pred_len)

        # Intra-branch MHA (Step 2) with learnable temperature
        self.fragment_attention = nn.MultiheadAttention(
            embed_dim=self.channels, num_heads=1, dropout=0.0, batch_first=True)
        self.fragment_scale = nn.Parameter(torch.tensor(10.0))

        # Inter-branch MHA (Step 3)
        self.branch_attention = nn.MultiheadAttention(
            embed_dim=self.channels, num_heads=1, dropout=0.0, batch_first=True)
        self.branch_scale = nn.Parameter(torch.tensor(10.0))

        # Independent branch-level query projection (ablation)
        self.rrf_branch_independent_proj = bool(
            getattr(configs, 'rrf_branch_independent_proj', 0))
        if self.rrf_branch_independent_proj:
            self.branch_q_proj = nn.Linear(self.seq_len, self.pred_len)
            with torch.no_grad():
                self.branch_q_proj.weight.zero_()
                self.branch_q_proj.bias.zero_()
        else:
            self.branch_q_proj = None

        # Output projection
        self.linear_out = nn.Linear(self.pred_len, self.pred_len)

    def rrf(self, query, multiview_values):
        """
        Reliability-aware Retrieval Fusion: two-stage attention.

        Args:
            query: (B, T, V) input sequence
            multiview_values: dict of {view_name: (B, M, P, V)} top-M candidates

        Returns:
            H_final: (B, P, V) fused prediction residual
        """
        # Step 1: future proxy Z_query
        z_query = self.linear_q(query.permute(0, 2, 1)).permute(0, 2, 1)

        # Step 2: intra-branch MHA per view
        view_summaries = []
        for v_name in self.view_names:
            V_v = multiview_values[v_name]
            B, M, P, V = V_v.shape
            V_flat = V_v.reshape(B * M, P, V)

            proj_layer = (self.linear_v_raw if v_name == 'raw'
                          else self.linear_v_trend if v_name == 'trend'
                          else self.linear_v_season)
            z_v_flat = proj_layer(V_flat.permute(0, 2, 1)).permute(0, 2, 1)

            # Attention Q/K/V
            z_query_rep = z_query.unsqueeze(1).expand(B, M, P, V).reshape(B * M, P, V)
            in_w = self.fragment_attention.in_proj_weight
            in_b = self.fragment_attention.in_proj_bias
            q_w, k_w, v_w = in_w.chunk(3)
            q_b, k_b, v_b = in_b.chunk(3)
            q = torch.nn.functional.linear(z_query_rep, q_w, q_b)
            k = torch.nn.functional.linear(z_v_flat, k_w, k_b)
            v = torch.nn.functional.linear(z_v_flat, v_w, v_b)
            scores = torch.bmm(q, k.transpose(1, 2)) * self.fragment_scale
            attn_w = torch.nn.functional.softmax(scores, dim=-1)
            attn_out = torch.bmm(attn_w, v)
            attn_out = torch.nn.functional.linear(attn_out,
                self.fragment_attention.out_proj.weight,
                self.fragment_attention.out_proj.bias)
            attn_out = attn_out.reshape(B, M, P, V)
            h_v = attn_out.mean(dim=1)
            view_summaries.append((v_name, h_v))

        # Step 3: inter-branch MHA
        h_v_stack = torch.stack([h for (_, h) in view_summaries], dim=1)
        B = h_v_stack.shape[0]
        V_views = h_v_stack.shape[1]
        h_v_flat = h_v_stack.reshape(B, V_views * self.pred_len, self.channels)
        z_query_branch = (z_query.unsqueeze(1).expand(B, V_views, self.pred_len, self.channels)
                          .reshape(B, V_views * self.pred_len, self.channels))

        if self.rrf_branch_independent_proj and self.branch_q_proj is not None:
            x_query_branch = self.branch_q_proj(query.permute(0, 2, 1)).permute(0, 2, 1)
            x_query_branch = (x_query_branch.unsqueeze(1)
                              .expand(B, V_views, self.pred_len, self.channels)
                              .reshape(B, V_views * self.pred_len, self.channels))
            z_query_branch = z_query_branch + x_query_branch

        in_w = self.branch_attention.in_proj_weight
        in_b = self.branch_attention.in_proj_bias
        q_w, k_w, v_w = in_w.chunk(3)
        q_b, k_b, v_b = in_b.chunk(3)
        q = torch.nn.functional.linear(z_query_branch, q_w, q_b)
        k = torch.nn.functional.linear(h_v_flat, k_w, k_b)
        v = torch.nn.functional.linear(h_v_flat, v_w, v_b)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.branch_scale
        attn_w = torch.nn.functional.softmax(scores, dim=-1)
        branch_out = torch.bmm(attn_w, v)
        branch_out = torch.nn.functional.linear(branch_out,
            self.branch_attention.out_proj.weight,
            self.branch_attention.out_proj.bias)
        branch_out = branch_out.reshape(B, V_views, self.pred_len, self.channels)
        H_final = branch_out.sum(dim=1)
        return H_final

    def prepare_dataset(self, train_data, valid_data, test_data):
        """Build multi-view retrieval library and precompute retrieval results."""
        self.rt.build_multiview_library(train_data)
        self.retrieval_dict = {}

        print('Doing Train Multi-view Retrieval')
        train_rt = self.rt.retrieve_all_multiview(train_data, train=True, device=self.device)
        print('Doing Valid Multi-view Retrieval')
        valid_rt = self.rt.retrieve_all_multiview(valid_data, train=False, device=self.device)
        print('Doing Test Multi-view Retrieval')
        test_rt = self.rt.retrieve_all_multiview(test_data, train=False, device=self.device)

        del self.rt
        torch.cuda.empty_cache()

        self.retrieval_dict['train'] = {
            v: val.detach().to(self.device) for v, (_, val) in train_rt.items()}
        self.retrieval_dict['valid'] = {
            v: val.detach().to(self.device) for v, (_, val) in valid_rt.items()}
        self.retrieval_dict['test'] = {
            v: val.detach().to(self.device) for v, (_, val) in test_rt.items()}

    def encoder(self, x, index, mode):
        """
        Forward pass: combines direct linear prediction with retrieval-augmented fusion.

        Args:
            x: (B, T, V) input
            index: (B,) sample indices in dataset
            mode: 'train' / 'valid' / 'test'

        Returns:
            pred: (B, P, V) forecast
        """
        index = index.to(self.device)
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset
        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        if not hasattr(self, 'retrieval_dict') or self.retrieval_dict is None or len(self.retrieval_dict) == 0:
            return (x_pred_from_x + x_offset).reshape(bsz, self.pred_len, self.channels)

        index_dev = index.to(self.retrieval_dict[mode][next(iter(self.retrieval_dict[mode]))].device)
        multiview_values = {v: val[index_dev] for v, val in self.retrieval_dict[mode].items()}

        if self.use_rrf:
            H_final = self.rrf(x_norm, multiview_values)
        else:
            view_hiddens = []
            for v_name in multiview_values:
                V_v = multiview_values[v_name]
                h_v = V_v.mean(dim=1)
                view_hiddens.append(h_v)
            H_final = torch.stack(view_hiddens, dim=1).mean(dim=1)

        pred = x_pred_from_x + H_final
        pred = self.linear_out(pred.permute(0, 2, 1)).permute(0, 2, 1)
        pred = pred.reshape(bsz, self.pred_len, self.channels)
        pred = pred + x_offset
        return pred

    def forecast(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def imputation(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def anomaly_detection(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def classification(self, x_enc, index, mode):
        enc_out = self.encoder(x_enc, index, mode)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, index, mode='train'):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x_enc, index, mode)[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, index, mode)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc, index, mode)
        if self.task_name == 'classification':
            return self.classification(x_enc, index, mode)
        return None
