# ============================================================
# PFR Cross-Asset Generalizability Engine
# Assets: BTCUSDT, ETHUSDT, SOLUSDT
#
# Objectives:
#   1. Dynamic filesystem resolution via pathlib.Path.
#   2. Cross-Asset Econometric Validation:
#      Verify if Directional Friction Resid produces positive,
#      statistically significant IC across distinct crypto assets.
#   3. Zero-Tuning Strategy Transfer:
#      Apply the exact v0.4 flagship architecture to ETH and SOL
#      without overfitting or asset-specific parameter tuning.
#   4. Multi-Asset Portfolio Construction:
#      Combine BTC, ETH, and SOL into an equal-weight portfolio
#      to measure diversification and risk-adjusted return benefits.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import time
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.api as sm

# ============================================================
# CONFIGURATION & DYNAMIC PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "pfr_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
START_DATE = "2020-01-01"
SPLIT_DATE = "2024-01-01"
COSTS_BPS = [0.0, 2.5, 5.0, 10.0]
ANN_FACTOR = np.sqrt(365 * 24)

LOOKBACK = 48
MEMORY_LAMBDA = 0.90
HALF_LIFE = 12
DIRECTION_LOOKBACK = 24

# ============================================================
# DATA DOWNLOAD & CACHING
# ============================================================

def download_asset_data(symbol, start_date=START_DATE):
    cache_file = OUTPUT_DIR / f"data_{symbol.lower()}_1h.csv"
    if not cache_file.exists():
        fallback = BASE_DIR / f"data_{symbol.lower()}_1h.csv"
        if fallback.exists():
            cache_file = fallback
            
    # Check if local cache exists
    if cache_file.exists():
        try:
            print(f"Loading {symbol} from {cache_file} (Offline Frozen Mode)...")
            df = pd.read_csv(cache_file)
            df["open_time"] = pd.to_datetime(df["open_time"], format="ISO8601")
            df["close_time"] = pd.to_datetime(df["close_time"], format="ISO8601")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
            print(f"Ready: {len(df):,} clean rows for {symbol} (Start: {df['open_time'].min()} End: {df['open_time'].max()})")
            return df
        except Exception as e:
            print(f"Error loading {cache_file}: {e}, downloading afresh...")
            cached_df = None
            start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    elif symbol == "BTCUSDT" and (OUTPUT_DIR / "data.csv").exists():
        print(f"Loading {symbol} from {OUTPUT_DIR / 'data.csv'} (Offline Frozen Mode)...")
        df = pd.read_csv(OUTPUT_DIR / "data.csv")
        df["open_time"] = pd.to_datetime(df["open_time"], format="ISO8601")
        df["close_time"] = pd.to_datetime(df["close_time"], format="ISO8601")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        print(f"Ready: {len(df):,} clean rows for {symbol} (Start: {df['open_time'].min()} End: {df['open_time'].max()})")
        return df
    else:
        cached_df = None
        start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)

    end_ts = int(pd.Timestamp.utcnow().timestamp() * 1000)
    
    # Download delta or full series if missing or outdated
    session = requests.Session()
    all_rows = []
    current_start = start_ts
    
    if current_start < end_ts - 3600 * 1000:
        print(f"Checking updates for {symbol} from {pd.to_datetime(current_start, unit='ms', utc=True)}...")
        
        while current_start < end_ts:
            params = {
                "symbol": symbol,
                "interval": "1h",
                "limit": 1000,
                "startTime": current_start,
                "endTime": end_ts,
            }
            try:
                r = session.get(BINANCE_URL, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"Request notice for {symbol}: {e}")
                break
                
            if not data:
                break
                
            all_rows.extend(data)
            last_open_time = data[-1][0]
            next_start = last_open_time + 1
            if next_start <= current_start:
                break
            current_start = next_start
            if len(data) < 1000:
                break

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]
    numeric_columns = ["open", "high", "low", "close", "volume"]

    if all_rows:
        new_df = pd.DataFrame(all_rows, columns=columns)
        new_df["open_time"] = pd.to_datetime(new_df["open_time"], unit="ms", utc=True)
        new_df["close_time"] = pd.to_datetime(new_df["close_time"], unit="ms", utc=True)
        for col in numeric_columns:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
        if cached_df is not None and not cached_df.empty:
            df = pd.concat([cached_df, new_df], ignore_index=True)
        else:
            df = new_df
    elif cached_df is not None and not cached_df.empty:
        df = cached_df
    else:
        raise RuntimeError(f"No data returned from Binance for {symbol}.")

    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    save_path = OUTPUT_DIR / f"data_{symbol.lower()}_1h.csv"
    df.to_csv(save_path, index=False)
    print(f"Ready: {len(df):,} clean rows for {symbol} (Start: {df['open_time'].min()} End: {df['open_time'].max()})")
    return df

# ============================================================
# FEATURE ENGINEERING & RESIDUALIZATION
# ============================================================

def rolling_zscore(series, window=LOOKBACK):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)

def compute_pfr_features(df):
    x = df.copy()
    
    # Returns
    x["log_return"] = np.log(x["close"] / x["close"].shift(1))
    x["pct_return"] = x["close"].pct_change().fillna(0.0)
    
    # ATR 48
    prev_close = x["close"].shift(1)
    tr1 = x["high"] - x["low"]
    tr2 = (x["high"] - prev_close).abs()
    tr3 = (x["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    x["atr_48"] = tr.rolling(LOOKBACK).mean()
    
    # Effort & Response
    x["effort_z"] = rolling_zscore(x["volume"], LOOKBACK)
    x["response_raw"] = x["log_return"].abs() / x["atr_48"].replace(0, np.nan)
    x["response_z"] = rolling_zscore(x["response_raw"], LOOKBACK)
    
    # Friction & Memory
    x["raw_friction"] = x["effort_z"] - x["response_z"]
    x["friction_memory"] = x["raw_friction"].ewm(alpha=1.0 - MEMORY_LAMBDA, adjust=False).mean()
    x["signed_friction"] = x["friction_memory"] * np.sign(x["log_return"])
    
    # Directional Friction
    alpha = 1.0 - 2.0 ** (-1.0 / HALF_LIFE)
    x["directional_friction"] = x["signed_friction"].ewm(alpha=alpha, adjust=False).mean()
    col_idx = x.columns.get_loc("directional_friction")
    x.iloc[:DIRECTION_LOOKBACK, col_idx] = np.nan
    
    # Momentum controls
    x["momentum_24"] = x["close"] / x["close"].shift(24) - 1.0
    x["sma_200"] = x["close"].rolling(200).mean()
    
    # Forward returns for IC analysis
    for h in [1, 5, 10, 20]:
        x[f"future_return_{h}"] = x["close"].shift(-h) / x["close"] - 1.0
        
    # Residualization: directional_friction ~ momentum_24
    mask = x[["directional_friction", "momentum_24"]].notna().all(axis=1)
    x["directional_friction_resid"] = np.nan
    if mask.sum() > 100:
        X = sm.add_constant(x.loc[mask, ["momentum_24"]])
        y = x.loc[mask, "directional_friction"]
        model = sm.OLS(y, X).fit()
        x.loc[mask, "directional_friction_resid"] = y - model.predict(X)
        
    return x

# ============================================================
# CROSS-ASSET IC ANALYSIS
# ============================================================

def analyze_cross_asset_ic(asset_dfs):
    print("\n" + "=" * 70)
    print("CROSS-ASSET SPEARMAN IC & STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    
    rows = []
    for symbol, df in asset_dfs.items():
        for factor in ["directional_friction", "directional_friction_resid", "response_z", "momentum_24"]:
            for h in [1, 5, 10, 20]:
                mask = df[[factor, f"future_return_{h}"]].notna().all(axis=1)
                if mask.sum() > 100:
                    rho, pval = spearmanr(df.loc[mask, factor], df.loc[mask, f"future_return_{h}"])
                    rows.append({
                        "symbol": symbol,
                        "factor": factor,
                        "horizon": h,
                        "Spearman_IC": rho,
                        "p_value": pval,
                        "N": mask.sum()
                    })
                    
    ic_df = pd.DataFrame(rows)
    
    # Focus table on directional_friction_resid
    resid_ic = ic_df[ic_df["factor"] == "directional_friction_resid"].pivot(
        index="symbol", columns="horizon", values="Spearman_IC"
    )
    resid_p = ic_df[ic_df["factor"] == "directional_friction_resid"].pivot(
        index="symbol", columns="horizon", values="p_value"
    )
    
    print("\n--- DIRECTIONAL FRICTION RESID SPEARMAN IC ---")
    print(resid_ic.round(4).to_string())
    
    print("\n--- P-VALUES ---")
    print(resid_p.map(lambda p: f"{p:.2e}").to_string())
    
    return ic_df

# ============================================================
# STRATEGY SIMULATION (FLAGSHIP ARCHITECTURE)
# ============================================================

def simulate_asset_strategy(df, q90_th, min_hold=16, max_hold=36):
    """
    Executes the exact v0.4 Flagship strategy:
    Entry: sig >= q90_th AND response_z > 0 AND close > sma_200
    Exit: bars >= max_hold OR (bars >= min_hold AND sig < 0)
    Next-bar execution (shift 1) to eliminate lookahead bias.
    """
    N = len(df)
    close = df["close"].values
    signal = df["directional_friction_resid"].values
    resp = df["response_z"].values
    sma200 = df["sma_200"].values
    
    position = np.zeros(N, dtype=float)
    trade_ids = np.zeros(N, dtype=int)
    in_pos = False
    bars_in_trade = 0
    current_trade_id = 0
    
    for i in range(1, N):
        if not in_pos:
            cond = (signal[i-1] >= q90_th) and (resp[i-1] > 0) and (close[i-1] > sma200[i-1])
            if cond:
                in_pos = True
                bars_in_trade = 0
                current_trade_id += 1
        else:
            bars_in_trade += 1
            if bars_in_trade >= max_hold or (bars_in_trade >= min_hold and signal[i-1] < 0):
                in_pos = False
                bars_in_trade = 0
                
        position[i] = 1.0 if in_pos else 0.0
        if in_pos:
            trade_ids[i] = current_trade_id
            
    return position, trade_ids

def calculate_performance(returns, position, trade_ids=None, cost_bps=5.0):
    cost_rate = cost_bps / 10000.0
    turnover = np.abs(np.diff(position, prepend=0.0))
    costs = turnover * cost_rate
    net_ret = position * returns - costs
    
    equity = np.cumprod(1.0 + net_ret)
    total_ret = equity[-1] - 1.0
    
    years = len(net_ret) / (365.25 * 24.0)
    cagr = ((1.0 + total_ret) ** (1.0 / years) - 1.0) if (years > 0 and total_ret > -1.0) else -1.0
    
    ret_std = np.std(net_ret)
    sharpe = (np.mean(net_ret) / ret_std * ANN_FACTOR) if ret_std > 0 else 0.0
    
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = np.min(drawdown)
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0
    
    # Trade-level metrics
    if trade_ids is not None and np.max(trade_ids) > 0:
        unique_tids = np.unique(trade_ids)
        unique_tids = unique_tids[unique_tids > 0]
        trade_pnls = []
        for tid in unique_tids:
            mask = (trade_ids == tid)
            t_ret = np.prod(1.0 + net_ret[mask]) - 1.0
            trade_pnls.append(t_ret)
            
        trade_pnls = np.array(trade_pnls)
        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls < 0]
        
        trade_win_rate = len(wins) / len(trade_pnls) if len(trade_pnls) > 0 else 0.0
        gross_gains = wins.sum() if len(wins) > 0 else 0.0
        gross_losses = -losses.sum() if len(losses) > 0 else 0.0
        trade_profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else np.nan
        payoff_ratio = (wins.mean() / abs(losses.mean())) if (len(losses) > 0 and len(wins) > 0 and abs(losses.mean()) > 0) else np.nan
        total_trades = len(trade_pnls)
    else:
        trade_win_rate = (net_ret > 0).sum() / (position > 0).sum() if (position > 0).sum() > 0 else 0.0
        gains = net_ret[net_ret > 0].sum()
        losses = -net_ret[net_ret < 0].sum()
        trade_profit_factor = (gains / losses) if losses > 0 else np.nan
        payoff_ratio = np.nan
        total_trades = (np.diff(position, prepend=0.0) > 0).sum()
        
    avg_hold = (np.sum(position > 0) / total_trades) if total_trades > 0 else 0.0
    
    return {
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "trade_win_rate": trade_win_rate,
        "trade_profit_factor": trade_profit_factor,
        "payoff_ratio": payoff_ratio,
        "trades": total_trades,
        "turnover": turnover.sum(),
        "avg_hold_h": avg_hold,
        "equity": equity,
        "net_returns": net_ret,
        "drawdown": drawdown
    }

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("PFR CROSS-ASSET GENERALIZABILITY & PORTFOLIO ENGINE")
    print("=" * 70)
    
    asset_dfs = {}
    for sym in SYMBOLS:
        df_raw = download_asset_data(sym)
        df_feat = compute_pfr_features(df_raw)
        asset_dfs[sym] = df_feat
        
    # 1. IC Analysis
    ic_results = analyze_cross_asset_ic(asset_dfs)
    ic_path = OUTPUT_DIR / "pfr_cross_asset_ic.csv"
    ic_results.to_csv(ic_path, index=False)
    print(f"\nSaved Spearman IC results to {ic_path}")
    
    # 2. Strategy evaluation per asset
    print("\n" + "=" * 70)
    print("ASSET-BY-ASSET STRATEGY BACKTEST (IN-SAMPLE & OUT-OF-SAMPLE)")
    print("=" * 70)
    
    strategy_results = []
    positions = {}
    asset_equities_5bps = {}
    bh_equities = {}
    
    # Find common date range for portfolio
    common_start = max(df["open_time"].min() for df in asset_dfs.values())
    common_end = min(df["open_time"].max() for df in asset_dfs.values())
    print(f"Common portfolio date range: {common_start} to {common_end}")
    
    for sym, df in asset_dfs.items():
        train_mask = (df["open_time"] < SPLIT_DATE).values
        test_mask = (df["open_time"] >= SPLIT_DATE).values
        full_mask = np.ones(len(df), dtype=bool)
        
        # In-sample Q90 threshold strictly on training data
        sig_train = df.loc[train_mask, "directional_friction_resid"].dropna().values
        q90_th = np.quantile(sig_train, 0.90)
        print(f"[{sym}] In-Sample Q90 Threshold: {q90_th:.6f}")
        
        pos, trade_ids = simulate_asset_strategy(df, q90_th, min_hold=16, max_hold=36)
        positions[sym] = (df["open_time"], pos, df["pct_return"])
        
        for slice_name, mask in [("IN_SAMPLE (2020-2023)", train_mask), 
                                 ("OUT_OF_SAMPLE (2024-2026)", test_mask), 
                                 ("FULL_SAMPLE", full_mask)]:
            sub_ret = df.loc[mask, "pct_return"].values
            sub_pos = pos[mask]
            sub_tids = trade_ids[mask]
            
            # Buy & Hold benchmark
            bh = calculate_performance(sub_ret, np.ones(len(sub_ret)), cost_bps=0.0)
            strategy_results.append({
                "symbol": sym,
                "sample": slice_name,
                "strategy": "Buy & Hold",
                "cost_bps": 0.0,
                **{k: bh[k] for k in ["total_return", "cagr", "sharpe", "max_drawdown", "calmar", "trade_win_rate", "trade_profit_factor", "payoff_ratio", "trades", "turnover", "avg_hold_h"]}
            })
            
            # PFR v0.4 at various cost levels
            for cost in COSTS_BPS:
                res = calculate_performance(sub_ret, sub_pos, trade_ids=sub_tids, cost_bps=cost)
                strategy_results.append({
                    "symbol": sym,
                    "sample": slice_name,
                    "strategy": "PFR v0.4 Flagship",
                    "cost_bps": cost,
                    **{k: res[k] for k in ["total_return", "cagr", "sharpe", "max_drawdown", "calmar", "trade_win_rate", "trade_profit_factor", "payoff_ratio", "trades", "turnover", "avg_hold_h"]}
                })
                
                if slice_name == "FULL_SAMPLE" and cost == 5.0:
                    asset_equities_5bps[sym] = (df["open_time"], res["equity"])
                    bh_equities[sym] = (df["open_time"], bh["equity"])
                    
    strat_df = pd.DataFrame(strategy_results)
    strat_results_path = OUTPUT_DIR / "pfr_cross_asset_results.csv"
    strat_df.to_csv(strat_results_path, index=False)
    print(f"Saved asset-by-asset strategy results to {strat_results_path}")
    
    # Print comparison table for 5 bps
    print("\n--- PERFORMANCE ACROSS ASSETS (At Standard 5 Bps Cost) ---")
    comp_table = strat_df[(strat_df["cost_bps"] == 5.0) | (strat_df["strategy"] == "Buy & Hold")].copy()
    display_cols = ["symbol", "sample", "strategy", "cost_bps", "total_return", "sharpe", "max_drawdown", "trade_win_rate", "trade_profit_factor", "trades", "turnover"]
    disp = comp_table[display_cols].copy()
    disp["total_return"] = (disp["total_return"] * 100).round(1).astype(str) + "%"
    disp["max_drawdown"] = (disp["max_drawdown"] * 100).round(1).astype(str) + "%"
    disp["trade_win_rate"] = (disp["trade_win_rate"] * 100).round(1).astype(str) + "%"
    disp["trade_profit_factor"] = disp["trade_profit_factor"].round(2)
    disp["sharpe"] = disp["sharpe"].round(2)
    disp["turnover"] = disp["turnover"].round(0).astype(int)
    print(disp.to_string(index=False))
    
    # ========================================================
    # MULTI-ASSET EQUAL-WEIGHT PORTFOLIO
    # ========================================================
    print("\n" + "=" * 70)
    print("EQUAL-WEIGHT MULTI-ASSET PORTFOLIO (BTC + ETH + SOL)")
    print("=" * 70)
    
    aligned_series = []
    aligned_bh = []
    
    for sym in SYMBOLS:
        times, pos, ret = positions[sym]
        cost_rate = 5.0 / 10000.0
        turnover = np.abs(np.diff(pos, prepend=0.0))
        net_ret = pos * ret - turnover * cost_rate
        
        s_strat = pd.Series(net_ret.values if hasattr(net_ret, "values") else net_ret, index=times, name=f"{sym}_pfr")
        s_bh = pd.Series(ret.values if hasattr(ret, "values") else ret, index=times, name=f"{sym}_bh")
        aligned_series.append(s_strat)
        aligned_bh.append(s_bh)
        
    df_port = pd.concat(aligned_series, axis=1).dropna()
    df_bh = pd.concat(aligned_bh, axis=1).dropna()
    
    port_ret = df_port.mean(axis=1)
    bh_port_ret = df_bh.mean(axis=1)
    
    def eval_portfolio(p_ret, name):
        eq = (1.0 + p_ret).cumprod()
        tot_ret = eq.iloc[-1] - 1.0
        years = len(p_ret) / (365.25 * 24.0)
        cagr = ((1.0 + tot_ret) ** (1.0 / years) - 1.0) if (years > 0 and tot_ret > -1.0) else -1.0
        std = p_ret.std()
        sharpe = (p_ret.mean() / std * ANN_FACTOR) if std > 0 else 0.0
        running_max = eq.cummax()
        dd = (eq - running_max) / running_max
        mdd = dd.min()
        calmar = (cagr / abs(mdd)) if mdd < 0 else 0.0
        return tot_ret, cagr, sharpe, mdd, calmar, eq, dd
        
    p_tot, p_cagr, p_sh, p_mdd, p_cal, port_eq, port_dd = eval_portfolio(port_ret, "PFR v0.4 Portfolio")
    bh_tot, bh_cagr, bh_sh, bh_mdd, bh_cal, bh_eq, bh_dd = eval_portfolio(bh_port_ret, "Buy & Hold Basket")
    
    print("\n--- PORTFOLIO SUMMARY (Common Horizon: Aug 2020 - Sep 2026) ---")
    print(f"PFR v0.4 Multi-Asset Portfolio (5 bps):")
    print(f"  Total Return: +{p_tot*100:.1f}% | CAGR: {p_cagr*100:.1f}% | Sharpe: {p_sh:.2f} | Max DD: {p_mdd*100:.1f}% | Calmar: {p_cal:.2f}")
    print(f"\nEqual-Weight Buy & Hold Basket (BTC + ETH + SOL):")
    print(f"  Total Return: +{bh_tot*100:.1f}% | CAGR: {bh_cagr*100:.1f}% | Sharpe: {bh_sh:.2f} | Max DD: {bh_mdd*100:.1f}% | Calmar: {bh_cal:.2f}")
    
    print("\n--- CORRELATION OF STRATEGY RETURNS ---")
    corr_matrix = df_port.corr()
    print(corr_matrix.round(3).to_string())
    
    # Charts
    print("\nGenerating cross-asset charts...")
    
    plt.figure(figsize=(14, 7))
    plt.plot(port_eq.index, port_eq.values, label=f"PFR v0.4 Portfolio (5 bps) [Sharpe {p_sh:.2f}]", color="green", linewidth=2.2)
    plt.plot(bh_eq.index, bh_eq.values, label=f"Buy & Hold Basket [Sharpe {bh_sh:.2f}]", color="black", linestyle="--", linewidth=1.5, alpha=0.6)
    
    for sym, color in [("BTCUSDT", "blue"), ("ETHUSDT", "purple"), ("SOLUSDT", "orange")]:
        times, eq = asset_equities_5bps[sym]
        s = pd.Series(eq, index=pd.to_datetime(times)).reindex(port_eq.index).ffill()
        s = s / s.iloc[0]
        plt.plot(s.index, s.values, label=f"{sym} PFR v0.4 (5 bps)", color=color, linewidth=1.2, alpha=0.7)
        
    plt.yscale("log")
    plt.title("Cross-Asset PFR v0.4 Performance (Log Scale, 2020 - 2026)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Equity Multiple (Log Scale)", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    eq_chart_path = OUTPUT_DIR / "pfr_cross_asset_equity.png"
    plt.savefig(eq_chart_path, dpi=150)
    plt.close()
    
    plt.figure(figsize=(14, 5))
    plt.plot(port_dd.index, port_dd.values * 100, label=f"PFR v0.4 Portfolio [Max DD: {p_mdd*100:.1f}%]", color="green", linewidth=1.8)
    plt.plot(bh_dd.index, bh_dd.values * 100, label=f"Buy & Hold Basket [Max DD: {bh_mdd*100:.1f}%]", color="red", linestyle="--", linewidth=1.2, alpha=0.5)
    plt.fill_between(bh_dd.index, bh_dd.values * 100, 0, color="red", alpha=0.1)
    
    plt.title("Drawdown Comparison: PFR v0.4 Portfolio vs Buy & Hold Basket (%)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Drawdown %", fontsize=11)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    dd_chart_path = OUTPUT_DIR / "pfr_cross_asset_drawdowns.png"
    plt.savefig(dd_chart_path, dpi=150)
    plt.close()
    
    print(f"Saved plots to {eq_chart_path} and {dd_chart_path}")
    print("\n" + "=" * 70)
    print("CROSS-ASSET PIPELINE COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
