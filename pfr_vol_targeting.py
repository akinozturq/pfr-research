# ============================================================
# PFR Dynamic Capital & Volatility Targeting Engine (v0.4.2)
#
# Objectives:
#   1. Risk Parity Sizing (Inverse Volatility):
#      Prevent high-beta assets (SOL, ETH) from dominating portfolio
#      variance by weighting each asset inversely to its realized volatility.
#   2. Dynamic Volatility Targeting (Vol Targeting):
#      Scale portfolio gross exposure dynamically to maintain a stable
#      annualized volatility target (e.g. 25% or 35%), dampening exposure
#      during high-volatility panic and expanding during low-volatility calm.
#   3. Turnover-Aware Rebalancing Buffer:
#      Implement a rebalancing deadband (5% weight buffer) to prevent
#      continuous fee drag from micro-rebalancing.
#   4. Multi-Cost Stress Test: 0.0, 2.5, 5.0, 10.0 bps.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SPLIT_DATE = "2024-01-01"
COSTS_BPS = [0.0, 2.5, 5.0, 10.0]
ANN_FACTOR = np.sqrt(365 * 24)

VOL_WINDOW = 48  # 48-hour rolling volatility window
TARGET_VOLS = [0.25, 0.35]  # 25% conservative, 35% balanced
MAX_LEVERAGE = 1.0  # Cash-only (no leverage), max gross exposure 100%
REBALANCE_BUFFER = 0.05  # Only rebalance if weight change > 5%

# ============================================================
# DATA & FEATURE LOADING
# ============================================================

def load_processed_assets():
    from pfr_cross_asset import download_asset_data, compute_pfr_features, simulate_asset_strategy
    
    asset_data = {}
    signals = {}
    
    for sym in SYMBOLS:
        raw = download_asset_data(sym)
        feat = compute_pfr_features(raw)
        
        # In-sample calibration for Q90
        train_mask = (feat["open_time"] < SPLIT_DATE).values
        sig_train = feat.loc[train_mask, "directional_friction_resid"].dropna().values
        q90_th = np.quantile(sig_train, 0.90)
        
        # Run flagship binary signal (1 = Long active, 0 = Cash)
        active_flag = simulate_asset_strategy(feat, q90_th, min_hold=16, max_hold=36)
        
        # Realized annualized volatility
        feat["ann_vol"] = feat["pct_return"].rolling(VOL_WINDOW).std() * ANN_FACTOR
        feat["active_flag"] = active_flag
        
        asset_data[sym] = feat
        signals[sym] = active_flag
        
    return asset_data, signals

# ============================================================
# POSITION SIZING ALGORITHMS
# ============================================================

def build_portfolio_weights(asset_data, sizing_method="equal_weight", 
                            target_vol=0.25, max_leverage=1.0, rebal_buffer=0.05):
    """
    Computes hourly weight matrix W(t) across assets based on:
    - 'equal_weight': Fixed 1/N per active asset
    - 'risk_parity': Weight proportional to 1 / realized_vol_i
    - 'vol_target': Risk parity scaled by target_vol / portfolio_vol
    Includes a turnover-minimizing rebalancing buffer.
    """
    # Align all assets on common datetime index
    dfs = []
    for sym, df in asset_data.items():
        sub = df[["open_time", "pct_return", "ann_vol", "active_flag"]].copy()
        sub.columns = ["open_time", f"{sym}_ret", f"{sym}_vol", f"{sym}_active"]
        sub = sub.set_index("open_time")
        dfs.append(sub)
        
    merged = pd.concat(dfs, axis=1).dropna()
    times = merged.index
    N = len(merged)
    K = len(SYMBOLS)
    
    weights = np.zeros((N, K), dtype=float)
    current_w = np.zeros(K, dtype=float)
    
    for t in range(1, N):
        active_t = np.array([merged[f"{sym}_active"].iloc[t-1] for sym in SYMBOLS])
        vols_t = np.array([merged[f"{sym}_vol"].iloc[t-1] for sym in SYMBOLS])
        
        n_active = np.sum(active_t)
        
        if n_active == 0:
            target_w = np.zeros(K)
        else:
            if sizing_method == "equal_weight":
                target_w = np.where(active_t > 0, 1.0 / n_active, 0.0)
                
            elif sizing_method in ["risk_parity", "vol_target"]:
                # Inverse volatility weights
                inv_vol = np.where((active_t > 0) & (vols_t > 0.01), 1.0 / vols_t, 0.0)
                sum_inv = np.sum(inv_vol)
                if sum_inv > 0:
                    base_w = inv_vol / sum_inv
                else:
                    base_w = np.where(active_t > 0, 1.0 / n_active, 0.0)
                    
                if sizing_method == "risk_parity":
                    # Allocate 100% across active assets (or fractional if less than K active)
                    target_w = base_w * min(1.0, n_active / float(K) * 1.5)
                    target_w = target_w / max(1.0, np.sum(target_w))
                    
                elif sizing_method == "vol_target":
                    # Estimate portfolio volatility using weighted average asset volatility
                    est_port_vol = np.sum(base_w * vols_t)
                    scalar = (target_vol / est_port_vol) if est_port_vol > 0.01 else 1.0
                    scalar = min(scalar, max_leverage)
                    target_w = base_w * scalar
                    
        # Apply rebalancing deadband buffer to avoid small micro-trades
        for k in range(K):
            diff = abs(target_w[k] - current_w[k])
            # Always execute complete entry (from 0) or complete exit (to 0)
            if (current_w[k] == 0 and target_w[k] > 0) or (target_w[k] == 0 and current_w[k] > 0):
                current_w[k] = target_w[k]
            elif diff >= rebal_buffer:
                current_w[k] = target_w[k]
            # Otherwise keep current_w[k] unchanged
            
        weights[t, :] = current_w.copy()
        
    df_weights = pd.DataFrame(weights, index=times, columns=SYMBOLS)
    df_returns = merged[[f"{sym}_ret" for sym in SYMBOLS]]
    df_returns.columns = SYMBOLS
    
    return df_weights, df_returns

# ============================================================
# PERFORMANCE SIMULATOR
# ============================================================

def evaluate_portfolio(df_weights, df_returns, cost_bps=5.0):
    cost_rate = cost_bps / 10000.0
    
    # Next-bar execution
    w_exec = df_weights.values
    r_exec = df_returns.values
    
    gross_pnl = np.sum(w_exec * r_exec, axis=1)
    turnover = np.sum(np.abs(np.diff(w_exec, axis=0, prepend=np.zeros((1, len(SYMBOLS))))), axis=1)
    costs = turnover * cost_rate
    net_pnl = gross_pnl - costs
    
    equity = np.cumprod(1.0 + net_pnl)
    total_ret = equity[-1] - 1.0
    
    years = len(net_pnl) / (365.25 * 24.0)
    cagr = ((1.0 + total_ret) ** (1.0 / years) - 1.0) if (years > 0 and total_ret > -1.0) else -1.0
    
    ret_std = np.std(net_pnl)
    ann_vol = ret_std * ANN_FACTOR
    sharpe = (np.mean(net_pnl) / ret_std * ANN_FACTOR) if ret_std > 0 else 0.0
    
    downside = net_pnl[net_pnl < 0]
    sortino = (np.mean(net_pnl) / np.std(downside) * ANN_FACTOR) if len(downside) > 1 and np.std(downside) > 0 else 0.0
    
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = np.min(drawdown)
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0
    
    # Average gross exposure
    avg_exposure = np.mean(np.sum(w_exec, axis=1))
    
    return {
        "total_return": total_ret,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "turnover": np.sum(turnover),
        "annual_turnover": np.sum(turnover) / years if years > 0 else 0.0,
        "avg_exposure": avg_exposure,
        "equity": pd.Series(equity, index=df_weights.index),
        "drawdown": pd.Series(drawdown, index=df_weights.index),
        "net_pnl": pd.Series(net_pnl, index=df_weights.index)
    }

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("PFR DYNAMIC CAPITAL & VOLATILITY TARGETING ENGINE")
    print("=" * 70)
    
    asset_data, signals = load_processed_assets()
    
    # Build models
    sizing_configs = [
        ("M1: Equal-Weight (1/N Active)", "equal_weight", 0.0),
        ("M2: Risk Parity (Inverse Vol)", "risk_parity", 0.0),
        ("M3: Vol Target 25% (Conservative)", "vol_target", 0.25),
        ("M4: Vol Target 35% (Balanced)", "vol_target", 0.35),
    ]
    
    model_weights = {}
    df_returns = None
    
    for label, method, vt in sizing_configs:
        print(f"Constructing weight engine for {label}...")
        w, r = build_portfolio_weights(
            asset_data, 
            sizing_method=method, 
            target_vol=vt, 
            max_leverage=MAX_LEVERAGE, 
            rebal_buffer=REBALANCE_BUFFER
        )
        model_weights[label] = w
        df_returns = r
        
    # Evaluate across sample slices
    all_results = []
    
    # Define date masks
    times = df_returns.index
    train_mask = (times < SPLIT_DATE)
    test_mask = (times >= SPLIT_DATE)
    full_mask = np.ones(len(times), dtype=bool)
    
    sample_slices = [
        ("IN_SAMPLE (2020-2023)", train_mask),
        ("OUT_OF_SAMPLE (2024-2026)", test_mask),
        ("FULL_SAMPLE (2020-2026)", full_mask)
    ]
    
    equity_curves_5bps = {}
    drawdown_curves_5bps = {}
    
    # Benchmark: Equal-weight Buy & Hold Basket
    bh_weights = pd.DataFrame(np.ones_like(df_returns) / len(SYMBOLS), index=times, columns=SYMBOLS)
    
    for slice_name, mask in sample_slices:
        sub_returns = df_returns[mask]
        sub_bh_w = bh_weights[mask]
        
        # Benchmark evaluation
        bh_eval = evaluate_portfolio(sub_bh_w, sub_returns, cost_bps=0.0)
        all_results.append({
            "sample": slice_name,
            "model": "Benchmark: Buy & Hold Basket",
            "cost_bps": 0.0,
            **{k: bh_eval[k] for k in ["total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "turnover", "avg_exposure"]}
        })
        
        if slice_name == "FULL_SAMPLE (2020-2026)":
            equity_curves_5bps["Buy & Hold Basket"] = bh_eval["equity"]
            drawdown_curves_5bps["Buy & Hold Basket"] = bh_eval["drawdown"]
            
        for label, _, _ in sizing_configs:
            w = model_weights[label]
            sub_w = w[mask]
            
            for cost in COSTS_BPS:
                res = evaluate_portfolio(sub_w, sub_returns, cost_bps=cost)
                all_results.append({
                    "sample": slice_name,
                    "model": label,
                    "cost_bps": cost,
                    **{k: res[k] for k in ["total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "turnover", "avg_exposure"]}
                })
                
                if slice_name == "FULL_SAMPLE (2020-2026)" and cost == 5.0:
                    equity_curves_5bps[label] = res["equity"]
                    drawdown_curves_5bps[label] = res["drawdown"]
                    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("pfr_vol_target_results.csv", index=False)
    
    # Print formatted comparative tables
    for slice_name, _ in sample_slices:
        print("\n" + "=" * 70)
        print(f"PERFORMANCE TABLE: {slice_name} (At Standard 5 Bps Cost)")
        print("=" * 70)
        
        sub = results_df[(results_df["sample"] == slice_name) & ((results_df["cost_bps"] == 5.0) | (results_df["model"].str.startswith("Benchmark")))].copy()
        display_cols = ["model", "cost_bps", "total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "calmar", "turnover", "avg_exposure"]
        disp = sub[display_cols].copy()
        disp["total_return"] = (disp["total_return"] * 100).round(1).astype(str) + "%"
        disp["cagr"] = (disp["cagr"] * 100).round(1).astype(str) + "%"
        disp["ann_vol"] = (disp["ann_vol"] * 100).round(1).astype(str) + "%"
        disp["max_drawdown"] = (disp["max_drawdown"] * 100).round(1).astype(str) + "%"
        disp["avg_exposure"] = (disp["avg_exposure"] * 100).round(1).astype(str) + "%"
        disp["sharpe"] = disp["sharpe"].round(2)
        disp["calmar"] = disp["calmar"].round(2)
        disp["turnover"] = disp["turnover"].round(0).astype(int)
        print(disp.to_string(index=False))
        
    # ========================================================
    # VISUALIZATIONS
    # ========================================================
    print("\nGenerating comprehensive charts...")
    
    # 1. Equity Curves
    plt.figure(figsize=(14, 7))
    for label, eq in equity_curves_5bps.items():
        if "Buy & Hold" in label:
            plt.plot(eq.index, eq.values, label=label, color="black", linestyle="--", linewidth=1.5, alpha=0.6)
        elif "Equal-Weight" in label:
            plt.plot(eq.index, eq.values, label=label, color="gray", linewidth=1.5, alpha=0.8)
        elif "Risk Parity" in label:
            plt.plot(eq.index, eq.values, label=label, color="blue", linewidth=1.8)
        elif "Vol Target 25%" in label:
            plt.plot(eq.index, eq.values, label=label, color="green", linewidth=2.0)
        elif "Vol Target 35%" in label:
            plt.plot(eq.index, eq.values, label=label, color="darkorange", linewidth=2.2)
            
    plt.yscale("log")
    plt.title("PFR Multi-Asset: Dynamic Sizing & Volatility Targeting (Log Scale, 5 Bps Cost)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Equity Multiple (Log Scale)", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig("pfr_vol_target_equity.png", dpi=150)
    plt.close()
    
    # 2. Drawdowns
    plt.figure(figsize=(14, 5))
    for label, dd in drawdown_curves_5bps.items():
        if "Buy & Hold" in label:
            plt.plot(dd.index, dd.values * 100, label=label, color="red", linestyle="--", linewidth=1.2, alpha=0.5)
            plt.fill_between(dd.index, dd.values * 100, 0, color="red", alpha=0.08)
        elif "Vol Target 25%" in label:
            plt.plot(dd.index, dd.values * 100, label=label, color="green", linewidth=1.8)
        elif "Risk Parity" in label:
            plt.plot(dd.index, dd.values * 100, label=label, color="blue", linewidth=1.4, alpha=0.7)
            
    plt.title("Portfolio Drawdown Comparison: Volatility Targeting vs Buy & Hold (%)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Drawdown %", fontsize=11)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("pfr_vol_target_drawdowns.png", dpi=150)
    plt.close()
    
    # 3. Dynamic Asset Allocation Chart (Stacked Area for Risk Parity & Vol Target)
    plt.figure(figsize=(14, 6))
    w_vt = model_weights["M4: Vol Target 35% (Balanced)"]
    plt.stackplot(w_vt.index, w_vt["BTCUSDT"], w_vt["ETHUSDT"], w_vt["SOLUSDT"], 
                  labels=["BTCUSDT Weight", "ETHUSDT Weight", "SOLUSDT Weight"],
                  colors=["#1f77b4", "#9467bd", "#ff7f0e"], alpha=0.85)
    plt.title("Dynamic Capital Allocation Over Time (PFR Vol Target 35%)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Portfolio Exposure (0.0 to 1.0)", fontsize=11)
    plt.ylim(0, 1.05)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("pfr_vol_target_allocations.png", dpi=150)
    plt.close()
    
    print("Saved plots to:")
    print("  - pfr_vol_target_equity.png")
    print("  - pfr_vol_target_drawdowns.png")
    print("  - pfr_vol_target_allocations.png")
    print("\n" + "=" * 70)
    print("DYNAMIC SIZING & VOL TARGETING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
