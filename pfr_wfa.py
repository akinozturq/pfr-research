# ============================================================
# PFR Walk-Forward Analysis (WFA) Engine
# Target: BTCUSDT (Isolated Variable Discipline)
#
# Methodology:
#   1. Rolling 18-month Train / 9-month Test / 9-month Step.
#   2. Zero-Leakage Protocol:
#      - OLS residualization alpha/beta fit strictly on Train.
#      - Entry quantile thresholds calibrated strictly on Train.
#   3. Systematic 128-Trial Grid Search on Train fold.
#   4. Dynamic Model Selection (Best IS Sharpe @ 5 bps) evaluated on Test.
#   5. Continuous Stitched Walk-Forward Out-of-Sample Equity Curve.
#   6. Parameter Stability Diagnostic (Tracking parameter migration).
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "pfr_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = OUTPUT_DIR / "data_btcusdt_1h.csv"
ANN_FACTOR = np.sqrt(365 * 24)
COST_BPS = 5.0
COST_RATE = COST_BPS / 10000.0

TRAIN_MONTHS = 18
TEST_MONTHS = 9
STEP_MONTHS = 9

LOOKBACK = 48
MEMORY_LAMBDA = 0.90
HALF_LIFE = 12
DIRECTION_LOOKBACK = 24

# ============================================================
# FEATURE COMPUTATION (ROLLING BASE FEATURES)
# ============================================================

def rolling_zscore(series, window=LOOKBACK):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)

def compute_base_features(df):
    x = df.copy()
    x["log_return"] = np.log(x["close"] / x["close"].shift(1))
    x["pct_return"] = x["close"].pct_change().fillna(0.0)
    
    prev_close = x["close"].shift(1)
    tr1 = x["high"] - x["low"]
    tr2 = (x["high"] - prev_close).abs()
    tr3 = (x["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    x["atr_48"] = tr.rolling(LOOKBACK).mean()
    
    x["effort_z"] = rolling_zscore(x["volume"], LOOKBACK)
    x["response_raw"] = x["log_return"].abs() / x["atr_48"].replace(0, np.nan)
    x["response_z"] = rolling_zscore(x["response_raw"], LOOKBACK)
    
    x["raw_friction"] = x["effort_z"] - x["response_z"]
    x["friction_memory"] = x["raw_friction"].ewm(alpha=1.0 - MEMORY_LAMBDA, adjust=False).mean()
    x["signed_friction"] = x["friction_memory"] * np.sign(x["log_return"])
    
    alpha = 1.0 - 2.0 ** (-1.0 / HALF_LIFE)
    x["directional_friction"] = x["signed_friction"].ewm(alpha=alpha, adjust=False).mean()
    col_idx = x.columns.get_loc("directional_friction")
    x.iloc[:DIRECTION_LOOKBACK, col_idx] = np.nan
    
    x["momentum_24"] = x["close"] / x["close"].shift(24) - 1.0
    x["sma_200"] = x["close"].rolling(200).mean()
    return x

# ============================================================
# SIMULATION & PERFORMANCE CALCULATION
# ============================================================

def simulate_strategy_fast(close, signal, resp, sma200, pct_ret, 
                           entry_th, min_hold, max_hold, req_resp, use_macro):
    N = len(close)
    position = np.zeros(N, dtype=float)
    trade_ids = np.zeros(N, dtype=int)
    in_pos = False
    bars_held = 0
    current_tid = 0
    
    for i in range(1, N):
        if not in_pos:
            c_sig = (signal[i-1] >= entry_th)
            c_resp = (resp[i-1] > 0) if req_resp else True
            c_macro = (close[i-1] > sma200[i-1]) if use_macro else True
            
            if c_sig and c_resp and c_macro:
                in_pos = True
                bars_held = 0
                current_tid += 1
        else:
            bars_held += 1
            if bars_held >= max_hold or (bars_held >= min_hold and signal[i-1] < 0.0):
                in_pos = False
                bars_held = 0
                
        position[i] = 1.0 if in_pos else 0.0
        if in_pos:
            trade_ids[i] = current_tid
            
    turnover = np.abs(np.diff(position, prepend=0.0))
    net_ret = position * pct_ret - turnover * COST_RATE
    return position, trade_ids, net_ret, turnover

def quick_metrics(net_ret, trade_ids=None):
    ret_std = np.std(net_ret)
    sharpe = (np.mean(net_ret) / ret_std * ANN_FACTOR) if ret_std > 0 else 0.0
    equity = np.cumprod(1.0 + net_ret)
    tot_ret = equity[-1] - 1.0 if len(equity) > 0 else 0.0
    
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0
    
    trades = 0
    win_rate = 0.0
    profit_factor = 0.0
    if trade_ids is not None and np.max(trade_ids) > 0:
        unique_tids = np.unique(trade_ids[trade_ids > 0])
        trade_pnls = []
        for tid in unique_tids:
            mask = (trade_ids == tid)
            trade_pnls.append(np.prod(1.0 + net_ret[mask]) - 1.0)
        trade_pnls = np.array(trade_pnls)
        trades = len(trade_pnls)
        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls < 0]
        win_rate = len(wins) / trades if trades > 0 else 0.0
        gross_gains = wins.sum() if len(wins) > 0 else 0.0
        gross_losses = -losses.sum() if len(losses) > 0 else 0.0
        profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else np.nan
        
    return {
        "sharpe": sharpe,
        "total_return": tot_ret,
        "max_drawdown": max_dd,
        "trades": trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "equity": equity,
        "drawdown": drawdown
    }

# ============================================================
# 128-COMBINATION GRID GENERATOR
# ============================================================

def build_grid(q_dict):
    grid = []
    for q_label, q_val in q_dict.items():
        for min_h, max_h in [(8, 24), (8, 36), (8, 48), (16, 24), (16, 36), (16, 48), (24, 36), (24, 48)]:
            for req_r in [False, True]:
                for use_m in [False, True]:
                    grid.append({
                        "name": f"{q_label}_h{min_h}-{max_h}_{'resp' if req_r else 'noresp'}_{'sma200' if use_m else 'nomacro'}",
                        "q_label": q_label,
                        "entry_th": q_val,
                        "min_hold": min_h,
                        "max_hold": max_h,
                        "require_resp": req_r,
                        "use_macro": use_m
                    })
    return grid

# ============================================================
# WALK-FORWARD EXECUTION PIPELINE
# ============================================================

def run_wfa():
    print("=" * 75)
    print("PFR WALK-FORWARD ANALYSIS (WFA) — BTCUSDT")
    print(f"Train: {TRAIN_MONTHS} Months | Test: {TEST_MONTHS} Months | Step: {STEP_MONTHS} Months (Non-Overlapping)")
    print(f"Execution Cost: {COST_BPS} bps | Model Space: 128 Trials per Fold")
    print("=" * 75)
    
    df = pd.read_csv(DATA_FILE)
    df["open_time"] = pd.to_datetime(df["open_time"], format="ISO8601")
    df = df.sort_values("open_time").reset_index(drop=True)
    
    print(f"Data Loaded: {len(df):,} bars from {df['open_time'].min()} to {df['open_time'].max()}")
    
    # 1. Compute rolling base features
    df = compute_base_features(df)
    
    end_dt = df["open_time"].max()
    cur_start = pd.Timestamp("2020-01-01", tz="UTC")
    
    fold_records = []
    stitched_oos_returns = []
    stitched_flagship_returns = []
    stitched_bh_returns = []
    stitched_times = []
    
    fold_idx = 1
    while True:
        train_end = cur_start + pd.DateOffset(months=TRAIN_MONTHS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)
        if test_end > end_dt + pd.DateOffset(days=5):
            test_end = end_dt
            
        train_mask = (df["open_time"] >= cur_start) & (df["open_time"] < train_end)
        test_mask = (df["open_time"] >= train_end) & (df["open_time"] <= test_end)
        
        train_df = df.loc[train_mask].copy().reset_index(drop=True)
        test_df = df.loc[test_mask].copy().reset_index(drop=True)
        
        if len(test_df) < 500:
            break
            
        # ----------------------------------------------------
        # LEAK-FREE OLS RESIDUALIZATION FIT ON TRAIN
        # ----------------------------------------------------
        mask_ols = train_df[["directional_friction", "momentum_24"]].notna().all(axis=1)
        X_train = sm.add_constant(train_df.loc[mask_ols, "momentum_24"])
        y_train = train_df.loc[mask_ols, "directional_friction"]
        ols_model = sm.OLS(y_train, X_train).fit()
        
        alpha_fit = ols_model.params["const"]
        beta_fit = ols_model.params["momentum_24"]
        
        # Apply to Train & Test
        train_df["directional_friction_resid"] = train_df["directional_friction"] - (alpha_fit + beta_fit * train_df["momentum_24"])
        test_df["directional_friction_resid"] = test_df["directional_friction"] - (alpha_fit + beta_fit * test_df["momentum_24"])
        
        # Calibrate entry quantiles strictly on train residuals
        sig_train = train_df["directional_friction_resid"].dropna().values
        q_dict = {
            "Q80": np.quantile(sig_train, 0.80),
            "Q85": np.quantile(sig_train, 0.85),
            "Q90": np.quantile(sig_train, 0.90),
            "Q95": np.quantile(sig_train, 0.95)
        }
        
        # ----------------------------------------------------
        # IN-SAMPLE 128-TRIAL GRID OPTIMIZATION
        # ----------------------------------------------------
        grid = build_grid(q_dict)
        
        close_tr = train_df["close"].values
        sig_tr = train_df["directional_friction_resid"].values
        resp_tr = train_df["response_z"].values
        sma200_tr = train_df["sma_200"].values
        ret_tr = train_df["pct_return"].values
        
        best_model = None
        best_is_sr = -999.0
        best_is_metrics = None
        
        for p in grid:
            pos, tids, net_r, _ = simulate_strategy_fast(
                close_tr, sig_tr, resp_tr, sma200_tr, ret_tr,
                p["entry_th"], p["min_hold"], p["max_hold"], p["require_resp"], p["use_macro"]
            )
            m = quick_metrics(net_r, tids)
            # Selection criteria: highest IS Sharpe with at least 10 trades
            if m["trades"] >= 10 and m["sharpe"] > best_is_sr:
                best_is_sr = m["sharpe"]
                best_model = p
                best_is_metrics = m
                
        # ----------------------------------------------------
        # OUT-OF-SAMPLE EVALUATION ON TEST SLICE
        # ----------------------------------------------------
        close_te = test_df["close"].values
        sig_te = test_df["directional_friction_resid"].values
        resp_te = test_df["response_z"].values
        sma200_te = test_df["sma_200"].values
        ret_te = test_df["pct_return"].values
        
        # 1. Evaluate Dynamically Selected Best Model
        pos_oos, tids_oos, net_r_oos, _ = simulate_strategy_fast(
            close_te, sig_te, resp_te, sma200_te, ret_te,
            best_model["entry_th"], best_model["min_hold"], best_model["max_hold"],
            best_model["require_resp"], best_model["use_macro"]
        )
        oos_m = quick_metrics(net_r_oos, tids_oos)
        
        # 2. Evaluate Static Benchmark (Flagship v0.4: Q90, 16-36h, resp, sma200)
        flagship_pos, flagship_tids, flagship_r, _ = simulate_strategy_fast(
            close_te, sig_te, resp_te, sma200_te, ret_te,
            q_dict["Q90"], 16, 36, True, True
        )
        flagship_m = quick_metrics(flagship_r, flagship_tids)
        
        # 3. Buy & Hold Benchmark
        bh_ret = ret_te
        
        # Append for continuous stitched curve
        stitched_oos_returns.extend(net_r_oos)
        stitched_flagship_returns.extend(flagship_r)
        stitched_bh_returns.extend(bh_ret)
        stitched_times.extend(test_df["open_time"])
        
        # WFE (Walk-Forward Efficiency)
        wfe = (oos_m["sharpe"] / best_is_metrics["sharpe"]) if best_is_metrics["sharpe"] > 0 else 0.0
        
        train_lbl = f"{cur_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}"
        test_lbl = f"{train_end.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}"
        
        fold_records.append({
            "fold": fold_idx,
            "train_period": train_lbl,
            "test_period": test_lbl,
            "selected_model": best_model["name"],
            "selected_q": best_model["q_label"],
            "selected_hold": f"{best_model['min_hold']}-{best_model['max_hold']}h",
            "selected_resp": best_model["require_resp"],
            "selected_macro": "SMA200" if best_model["use_macro"] else "None",
            "is_sharpe": round(best_is_metrics["sharpe"], 2),
            "is_trades": best_is_metrics["trades"],
            "oos_sharpe": round(oos_m["sharpe"], 2),
            "oos_return": f"{oos_m['total_return']*100:+.1f}%",
            "oos_max_dd": f"{oos_m['max_drawdown']*100:.1f}%",
            "oos_trades": oos_m["trades"],
            "oos_win_rate": f"{oos_m['win_rate']*100:.1f}%",
            "flagship_oos_sharpe": round(flagship_m["sharpe"], 2),
            "flagship_oos_return": f"{flagship_m['total_return']*100:+.1f}%",
            "wfe": round(wfe, 2)
        })
        
        if test_end >= end_dt:
            break
        cur_start = cur_start + pd.DateOffset(months=STEP_MONTHS)
        fold_idx += 1
        
    df_folds = pd.DataFrame(fold_records)
    
    # --------------------------------------------------------
    # FULL STITCHED OUT-OF-SAMPLE METRICS (5+ YEARS)
    # --------------------------------------------------------
    stitched_oos = np.array(stitched_oos_returns)
    stitched_flag = np.array(stitched_flagship_returns)
    stitched_bh = np.array(stitched_bh_returns)
    
    total_oos_m = quick_metrics(stitched_oos)
    total_flag_m = quick_metrics(stitched_flag)
    total_bh_m = quick_metrics(stitched_bh)
    
    # Plotting continuous walk-forward curve
    plt.figure(figsize=(14, 7))
    plt.plot(stitched_times, total_oos_m["equity"], label=f"WFA Adaptive Model (Sharpe: {total_oos_m['sharpe']:.2f}, MaxDD: {total_oos_m['max_drawdown']*100:.1f}%)", color="blue", lw=1.8)
    plt.plot(stitched_times, total_flag_m["equity"], label=f"Static Flagship v0.4 (Sharpe: {total_flag_m['sharpe']:.2f}, MaxDD: {total_flag_m['max_drawdown']*100:.1f}%)", color="green", lw=1.8, linestyle="--")
    plt.plot(stitched_times, total_bh_m["equity"], label=f"BTC Buy & Hold (Sharpe: {total_bh_m['sharpe']:.2f}, MaxDD: {total_bh_m['max_drawdown']*100:.1f}%)", color="gray", lw=1.2, alpha=0.6)
    
    plt.title("BTCUSDT Walk-Forward Continuous Out-of-Sample Equity Curves (2021 - 2026)", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Cumulative Growth (Base = 1.0)", fontsize=11)
    plt.grid(alpha=0.25)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "pfr_wfa_btc_equity.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    
    # Save CSV
    csv_path = OUTPUT_DIR / "pfr_wfa_btc_results.csv"
    df_folds.to_csv(csv_path, index=False)
    
    # --------------------------------------------------------
    # REPORTING & PARAMETER STABILITY SUMMARY
    # --------------------------------------------------------
    print("\n" + "=" * 75)
    print("WALK-FORWARD FOLD-BY-FOLD AUDIT TABLE (BTCUSDT)")
    print("=" * 75)
    cols_display = ["fold", "train_period", "test_period", "selected_q", "selected_hold", "selected_resp", "selected_macro", "is_sharpe", "oos_sharpe", "flagship_oos_sharpe", "wfe", "oos_trades"]
    print(df_folds[cols_display].to_string(index=False))
    
    print("\n" + "=" * 75)
    print("OVERALL 5-YEAR STITCHED OUT-OF-SAMPLE PERFORMANCE (July 2021 - Sep 2026)")
    print("=" * 75)
    print(f"Adaptive WFA Model : Total Return: {total_oos_m['total_return']*100:+.1f}% | Sharpe: {total_oos_m['sharpe']:.2f} | Max DD: {total_oos_m['max_drawdown']*100:.1f}%")
    print(f"Static Flagship v0.4: Total Return: {total_flag_m['total_return']*100:+.1f}% | Sharpe: {total_flag_m['sharpe']:.2f} | Max DD: {total_flag_m['max_drawdown']*100:.1f}%")
    print(f"BTC Buy & Hold      : Total Return: {total_bh_m['total_return']*100:+.1f}% | Sharpe: {total_bh_m['sharpe']:.2f} | Max DD: {total_bh_m['max_drawdown']*100:.1f}%")
    
    print("\n" + "=" * 75)
    print("PARAMETER STABILITY ANALYSIS")
    print("=" * 75)
    print("Selected Entry Quantile Distribution:")
    print(df_folds["selected_q"].value_counts().to_string())
    print("\nSelected Hold Duration Distribution:")
    print(df_folds["selected_hold"].value_counts().to_string())
    print("\nSelected Response Filter Distribution:")
    print(df_folds["selected_resp"].value_counts().to_string())
    print("\nSelected Macro Filter Distribution:")
    print(df_folds["selected_macro"].value_counts().to_string())
    
    return df_folds

if __name__ == "__main__":
    run_wfa()
