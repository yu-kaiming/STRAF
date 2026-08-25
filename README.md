# STRAF: Structure-Aware Retrieval-Augmented Time Series Forecasting


## Overview

STRAF combines a direct linear predictor with multi-view retrieval augmentation, fused via a two-stage attention mechanism:

1. **Multi-view decomposition** (Section 3.3): queries and candidates are decomposed into raw / trend / seasonal components, building three independent memory banks.
2. **Reliability-aware Retrieval Fusion (RRF)** (Section 3.4-3.5): intra-branch MHA aggregates Top-M candidates within each view; inter-branch MHA learns optimal view weights.
3. **Residual fusion**: retrieval predictions are added to the linear baseline and projected to output.

---

## Requirements

```bash
pip install -r requirements.txt
```

Core: Python ≥ 3.9, torch ≥ 1.10, numpy ≥ 1.24, pandas ≥ 1.5, scikit-learn ≥ 1.0, tqdm ≥ 4.65, sktime ≥ 0.16

---

## Quick Start

### 1. Prepare Data

```bash
mkdir -p data/ETTh1
# Place your CSV file (date column + features) at data/ETTh1/ETTh1.csv
```

### 2. Train

```bash
python run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id ETTh1_T96 --model RAFT --data ETTh1 \
  --root_path ./data/ETTh1/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --topm 20 --n_period 3 \
  --use_multiview 1 --n_views 3 \
  --use_reliability_fusion 1 \
  --batch_size 32 --train_epochs 10 --learning_rate 0.001
```

### 3. Test

```bash
python run.py \
  --task_name long_term_forecast --is_training 0 \
  --model_id ETTh1_T96 --model RAFT --data ETTh1 \
  --root_path ./data/ETTh1/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --topm 20 --n_period 3 \
  --use_multiview 1 --n_views 3 --use_reliability_fusion 1
```

### 4. Illness & Solar (one-shot)

```bash
bash run_illness_solar.sh
```

---

## Project Structure

```
STRAF/
├── run.py                          # Entry point: argparse + train/test loop
├── run_illness_solar.sh            # One-shot training for illness & solar
├── requirements.txt
├── README.md
├── LICENSE
├── models/
│   └── RAFT.py                    # Core model
├── layers/
│   └── Retrieval.py                # Multi-view retrieval module
├── data_provider/
│   ├── data_factory.py            # Dataset factory
│   └── data_loader.py             # Dataset classes
├── exp/
│   ├── exp_basic.py              # Base experiment class
│   └── exp_long_term_forecasting.py  # Training / validation / testing
└── utils/
    ├── tools.py                   # LR scheduler, EarlyStopping, visualizer
    ├── metrics.py                 # MAE, MSE, RMSE, MAPE, MSPE
    ├── timefeatures.py            # TimeFeature (gluonts)
    ├── print_args.py             # Argument printer
    ├── masking.py               # Attention masks
    ├── losses.py                # MAPE/SMAPE/MASE (N-BEATS)
    └── dtw_metric.py            # DTW and accelerated DTW
```


