# PFR Cross-Asset Generalizability & Replication Repository

This repository contains the frozen datasets and econometric pipeline for the Price Friction Response (PFR) strategy.

## Repository Contents
- `pfr_cross_asset.py`: Cross-asset Spearman IC analysis and zero-tuning strategy transfer engine (BTC, ETH, SOL).
- `pfr_v04.py`: Flagship single-asset engine with parametric 128-trial grid, Deflated Sharpe Ratio (DSR), and trade-level execution metrics.
- `pfr_vol_targeting.py`: Dynamic portfolio volatility targeting module.
- `pfr_output/`:
  - `data_btcusdt_1h.csv`: Frozen hourly Binance klines (2020-01-01 to 2026-09-16 08:00:00, 58,777 bars)
  - `data_ethusdt_1h.csv`: Frozen hourly Binance klines (2020-01-01 to 2026-09-16 08:00:00, 58,777 bars)
  - `data_solusdt_1h.csv`: Frozen hourly Binance klines (2020-08-11 to 2026-09-16 08:00:00, 53,431 bars)

## Exact Replication Command
Run directly in offline mode (no network access to Binance required):
```bash
python pfr_cross_asset.py
```

### Deterministic Replication Targets (Frozen at 2026-09-16 08:00:00 UTC)
#### 1. Spearman IC (h=20 hours)
- **BTCUSDT**: `+0.0392` (p = 1.92e-21)
- **ETHUSDT**: `+0.0282` (p = 8.83e-12)
- **SOLUSDT**: `+0.0149` (p = 5.99e-04)

#### 2. Out-of-Sample Performance (2024–2026 @ 5 bps cost)
- **BTCUSDT**: Return: `+53.2%`, Sharpe: `1.07`, Max Drawdown: `-8.7%`
- **ETHUSDT**: Return: `+8.3%`, Sharpe: `0.24`, Max Drawdown: `-48.4%`
- **SOLUSDT**: Return: `+49.2%`, Sharpe: `0.64`, Max Drawdown: `-39.4%`
