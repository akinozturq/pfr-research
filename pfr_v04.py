# ============================================================
# PFR v0.4 — Execution & Turnover Filtering Engine
# 
# Objectives:
#   1. Dynamic filesystem resolution via pathlib.Path(__file__).
#   2. Systematic Parameter Grid Search to eliminate hardcoded magic numbers.
#   3. Empirical Deflated Sharpe Ratio (DSR) using actual trial variance
#      and trial count across the parameter space.
#   4. Elimination of micro-churn via Hysteresis and Minimum Holding Periods.
#   5. Dual-Horizon execution:
#      - Directional Friction Resid (Macro absorption / Persistence)
#      - Response Z (Micro reaction trigger)
#      - Macro Trend Regime (SMA 200)
#   6. Strict In-Sample (2020-2023) vs Out-of-Sample (2024-2026) split.
#   7. Multi-cost stress testing: 0.0, 2.5, 5.0, 10.0 bps.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

# ============================================================
# DIRECTORY & FILE RESOLUTION (DYNAMIC PATHLIB)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "pfr_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = OUTPUT_DIR / "pfr_v03_full_dataset.csv"
if not DATA_FILE.exists():
    DATA_FILE = BASE_DIR / "pfr_v03_full_dataset.csv"

SPLIT_DATE = "2024-01-01"
COSTS_BPS = [0.0, 2.5, 5.0, 10.0]
ANN_FACTOR = np.sqrt(365 * 24)

# ============================================================
# DATA LOADING
# ============================================================

def load_data(filepath=DATA_FILE):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found. Please verify dataset location.")
    
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    df["open_time"] = pd.to_datetime(df["open_time"], format="ISO8601")
    df = df.sort_values("open_time").reset_index(drop=True)
    
    close_series = df["close"]
    df["sma_200"] = close_series.rolling(200).mean()
    df["sma_720"] = close_series.rolling(720).mean()
    df["return"] = df["close"].pct_change().fillna(0.0)
    
    print(f"Loaded {len(df):,} bars from {df['open_time'].min()} to {df['open_time'].max()}")
    return df

# ============================================================
# SIMULATION ENGINE
# ============================================================

def simulate_strategy(df, entry_threshold, exit_threshold=0.0, 
                      min_hold=16, max_hold=36, 
                      require_response=True, macro_filter="sma_200"):
    """
    Simulates a long-only strategy with Hysteresis, Minimum Holding Period,
    and optional timing & macro filters.
    All signal checks use bar i-1 to avoid lookahead bias (executes at bar i open/close).
    """
    N = len(df)
    close = df["close"].values
    signal = df["directional_friction_resid"].values
    resp = df["response_z"].values
    sma200 = df["sma_200"].values
    sma720 = df["sma_720"].values
    
    position = np.zeros(N, dtype=float)
    trade_ids = np.zeros(N, dtype=int)
    
    in_position = False
    bars_in_trade = 0
    current_trade_id = 0
    
    for i in range(1, N):
        sig_prev = signal[i-1]
        resp_prev = resp[i-1]
        close_prev = close[i-1]
        
        if not in_position:
            cond = (sig_prev >= entry_threshold)
            
            if require_response:
                cond = cond and (resp_prev > 0)
                
            if macro_filter == "sma_200":
                cond = cond and (close_prev > sma200[i-1])
            elif macro_filter == "sma_720":
                cond = cond and (close_prev > sma720[i-1])
                
            if cond:
                in_position = True
                bars_in_trade = 0
                current_trade_id += 1
        else:
            bars_in_trade += 1
            
            # Exit conditions
            exit_by_max = (bars_in_trade >= max_hold)
            exit_by_signal = (bars_in_trade >= min_hold) and (sig_prev < exit_threshold)
            
            if exit_by_max or exit_by_signal:
                in_position = False
                bars_in_trade = 0
                
        position[i] = 1.0 if in_position else 0.0
        if in_position:
            trade_ids[i] = current_trade_id
            
    return position, trade_ids

# ============================================================
# PERFORMANCE & DSR METRICS
# ============================================================

def deflated_sharpe_ratio(net_returns, n_trials, var_trials):
    """
    Computes Bailey & López de Prado (2014) Deflated Sharpe Ratio.
    Adjusts the estimated Sharpe ratio for:
      - Non-normality (skewness and kurtosis of returns)
      - Selection bias / Multiple testing (n_trials and empirical variance of trials var_trials)
    """
    T = len(net_returns)
    std = np.std(net_returns)
    if std == 0 or n_trials <= 1 or var_trials <= 0:
        return 0.0, 0.0
    
    sr_hourly = np.mean(net_returns) / std
    sk = skew(net_returns)
    kt = kurtosis(net_returns, fisher=False)
    
    # Expected maximum Sharpe under null hypothesis (Euler-Mascheroni approx)
    gamma = 0.5772156649
    e_max_sr = ((1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials) + 
                gamma * norm.ppf(1.0 - 1.0 / (n_trials * np.e))) * np.sqrt(var_trials)
    
    # Convert annual E[max SR] to hourly scale for comparison
    sr_zero = e_max_sr / ANN_FACTOR
    
    num = (sr_hourly - sr_zero) * np.sqrt(T - 1.0)
    denom = np.sqrt(1.0 - sk * sr_hourly + ((kt - 1.0) / 4.0) * (sr_hourly ** 2))
    dsr = norm.cdf(num / denom) if denom > 0 else 0.0
    return dsr, e_max_sr

def calculate_metrics(returns, position, trade_ids=None, costs_bps=0.0, n_trials=1, var_trials=0.0):
    cost_rate = costs_bps / 10000.0
    turnover_series = np.abs(np.diff(position, prepend=0.0))
    costs_series = turnover_series * cost_rate
    net_returns = position * returns - costs_series
    
    equity = np.cumprod(1.0 + net_returns)
    total_return = equity[-1] - 1.0
    
    n_bars = len(net_returns)
    years = n_bars / (365.25 * 24.0)
    cagr = ((1.0 + total_return) ** (1.0 / years) - 1.0) if (years > 0 and total_return > -1.0) else -1.0
    
    ret_std = np.std(net_returns)
    ret_mean = np.mean(net_returns)
    ann_vol = ret_std * ANN_FACTOR
    
    sharpe = (ret_mean / ret_std * ANN_FACTOR) if ret_std > 0 else 0.0
    
    downside = net_returns[net_returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 0.0
    sortino = (ret_mean / downside_std * ANN_FACTOR) if downside_std > 0 else 0.0
    
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = np.min(drawdown)
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0
    
    # Trade-level statistics (closed trade PnL)
    if trade_ids is not None and np.max(trade_ids) > 0:
        unique_tids = np.unique(trade_ids)
        unique_tids = unique_tids[unique_tids > 0]
        trade_pnls = []
        for tid in unique_tids:
            mask = (trade_ids == tid)
            t_ret = np.prod(1.0 + net_returns[mask]) - 1.0
            trade_pnls.append(t_ret)
            
        trade_pnls = np.array(trade_pnls)
        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls < 0]
        
        trade_win_rate = len(wins) / len(trade_pnls) if len(trade_pnls) > 0 else 0.0
        gross_gains = wins.sum() if len(wins) > 0 else 0.0
        gross_losses = -losses.sum() if len(losses) > 0 else 0.0
        trade_profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else np.nan
        payoff_ratio = (wins.mean() / abs(losses.mean())) if (len(losses) > 0 and len(wins) > 0 and abs(losses.mean()) > 0) else np.nan
        avg_trade_ret = trade_pnls.mean() if len(trade_pnls) > 0 else 0.0
        total_trades = len(trade_pnls)
    else:
        trade_win_rate = (net_returns > 0).sum() / (position > 0).sum() if (position > 0).sum() > 0 else 0.0
        gains = net_returns[net_returns > 0].sum()
        losses = -net_returns[net_returns < 0].sum()
        trade_profit_factor = (gains / losses) if losses > 0 else np.nan
        payoff_ratio = np.nan
        avg_trade_ret = 0.0
        total_trades = (np.diff(position, prepend=0.0) > 0).sum()
    
    total_turnover = turnover_series.sum()
    avg_hold_hours = (np.sum(position > 0) / total_trades) if total_trades > 0 else 0.0
    
    # Compute DSR if trials info provided
    if n_trials > 1 and var_trials > 0:
        dsr, e_max_sr = deflated_sharpe_ratio(net_returns, n_trials=n_trials, var_trials=var_trials)
    else:
        dsr, e_max_sr = 0.0, 0.0
    
    return {
        "cost_bps": costs_bps,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "trade_win_rate": trade_win_rate,
        "trade_profit_factor": trade_profit_factor,
        "payoff_ratio": payoff_ratio,
        "avg_trade_ret": avg_trade_ret,
        "dsr": dsr,
        "e_max_sr": e_max_sr,
        "trades": total_trades,
        "turnover": total_turnover,
        "avg_hold_h": avg_hold_hours,
        "equity": equity,
        "net_returns": net_returns,
        "drawdown": drawdown
    }

# ============================================================
# PARAMETRIC GRID GENERATOR
# ============================================================

def generate_parameter_grid(quantiles_dict):
    """
    Generates a systematic combinatorial grid across:
      - entry_quantile: Q80, Q85, Q90, Q95
      - min_hold: 8, 16, 24 hours
      - max_hold: 24, 36, 48 hours (constrained to max_hold > min_hold)
      - require_resp: False, True
      - macro_filter: 'none', 'sma_200'
    """
    grid = []
    for q_label, q_val in quantiles_dict.items():
        for min_h in [8, 16, 24]:
            for max_h in [24, 36, 48]:
                if max_h <= min_h:
                    continue
                for req_r in [False, True]:
                    for macro in ["none", "sma_200"]:
                        grid.append({
                            "name": f"{q_label}_h{min_h}-{max_h}_{'resp' if req_r else 'noresp'}_{macro}",
                            "q_label": q_label,
                            "entry_th": q_val,
                            "exit_th": 0.0,
                            "min_hold": min_h,
                            "max_hold": max_h,
                            "require_resp": req_r,
                            "macro_filter": macro
                        })
    return grid

# ============================================================
# MAIN EXPERIMENT PIPELINE
# ============================================================

def run_v04_pipeline():
    print("=" * 70)
    print("PFR v0.4 — PARAMETRIC GRID & EXECUTION ENGINE")
    print("=" * 70)
    
    df = load_data()
    
    # Train / Test split definition
    train_mask = (df["open_time"] < SPLIT_DATE).values
    test_mask = (df["open_time"] >= SPLIT_DATE).values
    full_mask = np.ones(len(df), dtype=bool)
    
    # In-Sample threshold calibration (ZERO lookahead bias into test period)
    sig_train = df.loc[train_mask, "directional_friction_resid"].dropna().values
    quantiles_in_sample = {
        "Q80": np.quantile(sig_train, 0.80),
        "Q85": np.quantile(sig_train, 0.85),
        "Q90": np.quantile(sig_train, 0.90),
        "Q95": np.quantile(sig_train, 0.95),
    }
    
    print("\n" + "=" * 70)
    print("CALIBRATION (IN-SAMPLE: 2020-01-01 to 2023-12-31)")
    print("=" * 70)
    for ql, qv in quantiles_in_sample.items():
        print(f"In-Sample {ql} Threshold: {qv:.6f}")
    print(f"In-Sample Bars: {train_mask.sum():,} | Out-of-Sample Bars: {test_mask.sum():,}")
    
    # Generate systematic grid
    param_grid = generate_parameter_grid(quantiles_in_sample)
    n_trials = len(param_grid)
    print(f"\nGenerated Systematic Parameter Grid: {n_trials} candidate combinations.")
    
    # ------------------------------------------------------------
    # STEP 1: RUN PARAMETER GRID ON IN-SAMPLE TO OBTAIN EMPIRICAL DSR
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"PHASE 1: GRID EVALUATION (IN-SAMPLE 2020-2023 at 5.0 bps cost)")
    print("=" * 70)
    
    sub_df_train = df[train_mask].reset_index(drop=True)
    ret_train = sub_df_train["return"].values
    
    grid_rows = []
    train_sharpes = []
    
    for idx, m in enumerate(param_grid):
        pos_full, trade_ids = simulate_strategy(
            df,
            entry_threshold=m["entry_th"],
            exit_threshold=m["exit_th"],
            min_hold=m["min_hold"],
            max_hold=m["max_hold"],
            require_response=m["require_resp"],
            macro_filter=m["macro_filter"]
        )
        sub_pos = pos_full[train_mask]
        sub_tids = trade_ids[train_mask]
        
        # In-sample evaluation at 5.0 bps
        res = calculate_metrics(ret_train, sub_pos, trade_ids=sub_tids, costs_bps=5.0)
        train_sharpes.append(res["sharpe"])
        
        grid_rows.append({
            "model_idx": idx,
            "name": m["name"],
            "q_label": m["q_label"],
            "min_hold": m["min_hold"],
            "max_hold": m["max_hold"],
            "require_resp": m["require_resp"],
            "macro_filter": m["macro_filter"],
            "is_sharpe": res["sharpe"],
            "is_cagr": res["cagr"],
            "is_max_dd": res["max_drawdown"],
            "is_win_rate": res["trade_win_rate"],
            "is_profit_factor": res["trade_profit_factor"],
            "is_payoff_ratio": res["payoff_ratio"],
            "is_trades": res["trades"],
            "is_turnover": res["turnover"],
            "is_avg_hold_h": res["avg_hold_h"]
        })
        
    grid_df = pd.DataFrame(grid_rows)
    
    # Calculate Empirical Trial Variance across In-Sample Grid
    train_sharpes = np.array(train_sharpes)
    var_trials_empirical = float(np.var(train_sharpes, ddof=1))
    mean_sr_is = float(np.mean(train_sharpes))
    max_sr_is = float(np.max(train_sharpes))
    
    gamma = 0.5772156649
    e_max_sr_empirical = float(((1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials) + 
                                gamma * norm.ppf(1.0 - 1.0 / (n_trials * np.e))) * np.sqrt(var_trials_empirical))
    
    print(f"\nEmpirical Grid Distribution Statistics (N={n_trials} trials):")
    print(f"  Mean In-Sample Sharpe:     {mean_sr_is:.4f}")
    print(f"  Empirical Sharpe Variance: {var_trials_empirical:.4f} (Std: {np.sqrt(var_trials_empirical):.4f})")
    print(f"  Max In-Sample Sharpe:      {max_sr_is:.4f}")
    print(f"  Expected Max SR by Chance: {e_max_sr_empirical:.4f}")
    
    # Top 5 In-Sample Models from Grid
    top_grid = grid_df.sort_values("is_sharpe", ascending=False).head(5)
    print("\nTop 5 Grid Candidates by In-Sample Sharpe (5 bps):")
    display_top = top_grid[["name", "is_sharpe", "is_cagr", "is_max_dd", "is_win_rate", "is_profit_factor", "is_trades"]]
    print(display_top.to_string(index=False))
    
    # Save complete grid trials
    grid_csv_path = OUTPUT_DIR / "pfr_v04_grid_trials.csv"
    grid_df.to_csv(grid_csv_path, index=False)
    print(f"\nSaved all {n_trials} parameter combinations to {grid_csv_path}")
    
    # ------------------------------------------------------------
    # STEP 2: REPRESENTATIVE & BENCHMARK ARCHITECTURE COMPARISON
    # ------------------------------------------------------------
    # Representative models spanning naive baseline to flagship configurations
    rep_models = [
        {
            "name": "M0: Naive Long-Only (signal > 0)",
            "entry_th": 0.0,
            "exit_th": 0.0,
            "min_hold": 0,
            "max_hold": 99999,
            "require_resp": False,
            "macro_filter": "none"
        },
        {
            "name": "M1: Hysteresis (Q85 in, 0 out, min 16h)",
            "entry_th": quantiles_in_sample["Q85"],
            "exit_th": 0.0,
            "min_hold": 16,
            "max_hold": 36,
            "require_resp": False,
            "macro_filter": "none"
        },
        {
            "name": "M2: Dual Timing (Q90 + Resp>0, min 16h)",
            "entry_th": quantiles_in_sample["Q90"],
            "exit_th": 0.0,
            "min_hold": 16,
            "max_hold": 36,
            "require_resp": True,
            "macro_filter": "none"
        },
        {
            "name": "M3: Dual + Macro SMA200 (Q85 + Resp>0, Hold 16-36h)",
            "entry_th": quantiles_in_sample["Q85"],
            "exit_th": 0.0,
            "min_hold": 16,
            "max_hold": 36,
            "require_resp": True,
            "macro_filter": "sma_200"
        },
        {
            "name": "M4: Flagship PFR v0.4 (Q90 + Resp>0 + SMA200, Hold 16-36h)",
            "entry_th": quantiles_in_sample["Q90"],
            "exit_th": 0.0,
            "min_hold": 16,
            "max_hold": 36,
            "require_resp": True,
            "macro_filter": "sma_200"
        },
        {
            "name": "M5: Extended PFR v0.4 (Q90 + Resp>0 + SMA200, Hold 20-48h)",
            "entry_th": quantiles_in_sample["Q90"],
            "exit_th": 0.0,
            "min_hold": 20,
            "max_hold": 48,
            "require_resp": True,
            "macro_filter": "sma_200"
        },
    ]
    
    sample_slices = [
        ("IN_SAMPLE (2020-2023)", train_mask),
        ("OUT_OF_SAMPLE (2024-2026)", test_mask),
        ("FULL_SAMPLE (2020-2026)", full_mask)
    ]
    
    all_summary_rows = []
    curves_to_plot = {}
    
    for slice_name, mask in sample_slices:
        print("\n" + "=" * 70)
        print(f"EVALUATION: {slice_name}")
        print("=" * 70)
        
        sub_df = df[mask].reset_index(drop=True)
        sub_ret = sub_df["return"].values
        
        # Benchmark: Buy & Hold BTC
        bh_metrics = calculate_metrics(
            sub_ret, np.ones(len(sub_ret)), 
            costs_bps=0.0, 
            n_trials=1, 
            var_trials=0.0
        )
        all_summary_rows.append({
            "sample": slice_name,
            "model": "Benchmark: Buy & Hold BTC",
            "cost_bps": 0.0,
            "total_return": bh_metrics["total_return"],
            "cagr": bh_metrics["cagr"],
            "ann_vol": bh_metrics["ann_vol"],
            "sharpe": bh_metrics["sharpe"],
            "sortino": bh_metrics["sortino"],
            "max_drawdown": bh_metrics["max_drawdown"],
            "calmar": bh_metrics["calmar"],
            "trade_win_rate": bh_metrics["trade_win_rate"],
            "trade_profit_factor": bh_metrics["trade_profit_factor"],
            "payoff_ratio": bh_metrics["payoff_ratio"],
            "avg_trade_ret": bh_metrics["avg_trade_ret"],
            "dsr": bh_metrics["dsr"],
            "trades": bh_metrics["trades"],
            "turnover": bh_metrics["turnover"],
            "avg_hold_h": bh_metrics["avg_hold_h"]
        })
        
        if slice_name == "FULL_SAMPLE (2020-2026)":
            curves_to_plot["Buy & Hold BTC"] = bh_metrics["equity"]
        
        for m in rep_models:
            pos_full, trade_ids = simulate_strategy(
                df,
                entry_threshold=m["entry_th"],
                exit_threshold=m["exit_th"],
                min_hold=m["min_hold"],
                max_hold=m["max_hold"],
                require_response=m["require_resp"],
                macro_filter=m["macro_filter"]
            )
            
            sub_pos = pos_full[mask]
            sub_trade_ids = trade_ids[mask]
            
            for cost in COSTS_BPS:
                # Pass EMPIRICAL n_trials and var_trials
                res = calculate_metrics(
                    sub_ret, sub_pos, 
                    trade_ids=sub_trade_ids, 
                    costs_bps=cost,
                    n_trials=n_trials,
                    var_trials=var_trials_empirical
                )
                
                all_summary_rows.append({
                    "sample": slice_name,
                    "model": m["name"],
                    "cost_bps": cost,
                    "total_return": res["total_return"],
                    "cagr": res["cagr"],
                    "ann_vol": res["ann_vol"],
                    "sharpe": res["sharpe"],
                    "sortino": res["sortino"],
                    "max_drawdown": res["max_drawdown"],
                    "calmar": res["calmar"],
                    "trade_win_rate": res["trade_win_rate"],
                    "trade_profit_factor": res["trade_profit_factor"],
                    "payoff_ratio": res["payoff_ratio"],
                    "avg_trade_ret": res["avg_trade_ret"],
                    "dsr": res["dsr"],
                    "trades": res["trades"],
                    "turnover": res["turnover"],
                    "avg_hold_h": res["avg_hold_h"]
                })
                
                if slice_name == "FULL_SAMPLE (2020-2026)":
                    if "M4: Flagship" in m["name"] and cost in [0.0, 5.0, 10.0]:
                        curves_to_plot[f"M4 Flagship ({cost:.0f} bps)"] = res["equity"]
                    if "M5: Extended" in m["name"] and cost == 5.0:
                        curves_to_plot[f"M5 Extended (5 bps)"] = res["equity"]
                        
    summary_df = pd.DataFrame(all_summary_rows)
    
    # Formatted tables
    for slice_name, _ in sample_slices:
        print(f"\n--- {slice_name} PERFORMANCE TABLE ---")
        slice_table = summary_df[summary_df["sample"] == slice_name]
        
        display_cols = ["model", "cost_bps", "total_return", "sharpe", "max_drawdown", "trade_win_rate", "trade_profit_factor", "payoff_ratio", "dsr", "trades", "turnover"]
        display_df = slice_table[display_cols].copy()
        
        display_df["total_return"] = (display_df["total_return"] * 100).round(1).astype(str) + "%"
        display_df["max_drawdown"] = (display_df["max_drawdown"] * 100).round(1).astype(str) + "%"
        display_df["trade_win_rate"] = (display_df["trade_win_rate"] * 100).round(1).astype(str) + "%"
        display_df["trade_profit_factor"] = display_df["trade_profit_factor"].round(2)
        display_df["payoff_ratio"] = display_df["payoff_ratio"].round(2)
        display_df["dsr"] = (display_df["dsr"] * 100).round(2).astype(str) + "%"
        display_df["sharpe"] = display_df["sharpe"].round(2)
        display_df["turnover"] = display_df["turnover"].round(0).astype(int)
        
        print(display_df.to_string(index=False))
        
    # Save CSV results directly into OUTPUT_DIR
    results_csv_path = OUTPUT_DIR / "pfr_v04_results.csv"
    summary_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved comprehensive results to {results_csv_path}")
    
    # Generate Charts directly into OUTPUT_DIR
    print("\nGenerating charts...")
    
    # 1. Equity Curves
    plt.figure(figsize=(14, 7))
    for label, eq in curves_to_plot.items():
        if "Buy & Hold" in label:
            plt.plot(eq, label=label, color="black", linestyle="--", alpha=0.6, linewidth=1.5)
        elif "M4 Flagship (0 bps)" in label:
            plt.plot(eq, label=label, color="green", linewidth=1.5)
        elif "M4 Flagship (5 bps)" in label:
            plt.plot(eq, label=label, color="blue", linewidth=2.0)
        elif "M4 Flagship (10 bps)" in label:
            plt.plot(eq, label=label, color="orange", linewidth=1.5)
        else:
            plt.plot(eq, label=label, linewidth=1.2)
            
    plt.yscale("log")
    plt.title("PFR v0.4 — Equity Curves (Log Scale, 2020 - 2026)", fontsize=14, fontweight="bold")
    plt.xlabel("Hourly Bars", fontsize=11)
    plt.ylabel("Equity Multiple (Log Scale)", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    equity_png_path = OUTPUT_DIR / "pfr_v04_equity.png"
    plt.savefig(equity_png_path, dpi=150)
    plt.close()
    print(f"Saved equity curves to {equity_png_path}")
    
    # 2. Drawdown comparison chart
    plt.figure(figsize=(14, 5))
    bh_eq = curves_to_plot["Buy & Hold BTC"]
    bh_dd = (bh_eq - np.maximum.accumulate(bh_eq)) / np.maximum.accumulate(bh_eq)
    
    flagship_eq = curves_to_plot.get("M4 Flagship (5 bps)")
    if flagship_eq is not None:
        flagship_dd = (flagship_eq - np.maximum.accumulate(flagship_eq)) / np.maximum.accumulate(flagship_eq)
        plt.plot(flagship_dd * 100, label="PFR v0.4 Flagship (5 bps fee)", color="blue", linewidth=1.5)
        
    plt.plot(bh_dd * 100, label="Buy & Hold BTC", color="red", alpha=0.5, linestyle="--", linewidth=1.2)
    plt.fill_between(range(len(bh_dd)), bh_dd * 100, 0, color="red", alpha=0.1)
    
    plt.title("Drawdown Comparison: PFR v0.4 vs Buy & Hold BTC (%)", fontsize=14, fontweight="bold")
    plt.xlabel("Hourly Bars", fontsize=11)
    plt.ylabel("Drawdown %", fontsize=11)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    dd_png_path = OUTPUT_DIR / "pfr_v04_drawdown.png"
    plt.savefig(dd_png_path, dpi=150)
    plt.close()
    print(f"Saved drawdown chart to {dd_png_path}")
    
    print("\n" + "=" * 70)
    print("PFR v0.4 PIPELINE COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_v04_pipeline()
