import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 0
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='STRAF')

    # Basic
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='temp')
    parser.add_argument('--model', type=str, default='RAFT')

    # Data
    parser.add_argument('--data', type=str, default='traffic')
    parser.add_argument('--root_path', type=str, default='./data/traffic')
    parser.add_argument('--data_path', type=str, default='traffic.csv')
    parser.add_argument('--features', type=str, default='M',
                        help='M: multivariate predict multivariate, S: univariate predict univariate, MS: multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)

    # Model dimensions
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--down_sampling_method', type=str, default=None)
    parser.add_argument('--seg_len', type=int, default=48)
    parser.add_argument('--output_attention', action='store_true', default=False)
    parser.add_argument('--n_period', type=int, default=3)
    parser.add_argument('--topm', type=int, default=20)

    # STRAF retrieval hyperparameters
    parser.add_argument('--use_multiview', type=int, default=1,
                        help='Enable multi-view retrieval (1=True, 0=False)')
    parser.add_argument('--n_views', type=int, default=3,
                        help='Number of views: 1=raw, 2=raw+trend, 3=raw+trend+season')
    parser.add_argument('--decomp_kernel', type=int, default=25,
                        help='Moving-average window tau for trend-seasonal decomposition')
    parser.add_argument('--use_reliability_fusion', type=int, default=1,
                        help='Enable RRF two-stage attention (1=True, 0=simple mean fusion)')
    parser.add_argument('--rrf_n_heads', type=int, default=4,
                        help='Attention heads in RRF modules')
    parser.add_argument('--rrf_d_model', type=int, default=0,
                        help='Hidden dim for RRF. 0 = auto (pred_len)')
    parser.add_argument('--rrf_branch_independent_proj', type=int, default=0,
                        help='Independent projection for inter-branch Q (ablation)')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    # De-stationary projector
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)

    # Metrics
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='DTW metric (slow, not recommended)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--jitter', action='store_true', default=False)
    parser.add_argument('--scaling', action='store_true', default=False)
    parser.add_argument('--permutation', action='store_true', default=False)
    parser.add_argument('--randompermutation', action='store_true', default=False)
    parser.add_argument('--magwarp', action='store_true', default=False)
    parser.add_argument('--timewarp', action='store_true', default=False)
    parser.add_argument('--windowslice', action='store_true', default=False)
    parser.add_argument('--windowwarp', action='store_true', default=False)
    parser.add_argument('--rotation', action='store_true', default=False)
    parser.add_argument('--spawner', action='store_true', default=False)
    parser.add_argument('--dtwwarp', action='store_true', default=False)
    parser.add_argument('--shapedtwwarp', action='store_true', default=False)
    parser.add_argument('--wdba', action='store_true', default=False)
    parser.add_argument('--discdtw', action='store_true', default=False)
    parser.add_argument('--discsdtw', action='store_true', default=False)
    parser.add_argument('--extra_tag', type=str, default='')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        args.device_ids = [int(id_) for id_ in args.devices.split(',')]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name, args.model_id, args.model, args.data,
                args.features, args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
                args.factor, args.embed, args.distil, args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp = Exp(args)
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data,
            args.features, args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
            args.factor, args.embed, args.distil, args.des, ii)
        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
