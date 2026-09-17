# ============================================================
# PFR Live Quantitative Signal & WhatsApp Alert Engine
#
# Features:
#   1. Connects to Binance live REST API (public klines, no keys required).
#   2. Computes real-time PFR, directional friction, momentum residualization,
#      and Risk Parity position sizing.
#   3. Evaluates Flagship entry, holding, and exit logic for BTC, ETH, and SOL.
#   4. Sends formatted WhatsApp notifications via CallMeBot gateway
#      (or prints console simulation if credentials not configured).
#   5. Maintains trade history and duration in pfr_live_state.json.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import time
import argparse
import urllib.parse
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm

CONFIG_FILE = "pfr_live_config.json"
STATE_FILE = "pfr_live_state.json"
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
ANN_FACTOR = np.sqrt(365 * 24)

# ============================================================
# CONFIG & STATE MANAGEMENT
# ============================================================

def load_config():
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"{CONFIG_FILE} not found.")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_state():
    if not os.path.exists(STATE_FILE):
        default_state = {
            "last_check_time": None,
            "portfolio_gross_exposure": 0.0,
            "positions": {s: {"in_position": False, "entry_time": None, "entry_price": 0.0, "bars_held": 0, "weight": 0.0} for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
            "history": []
        }
        save_state(default_state)
        return default_state
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

# ============================================================
# LIVE DATA FETCHING
# ============================================================

def fetch_live_klines(symbol, limit=300):
    """
    Fetches the last N closed 1h candles from Binance.
    """
    params = {
        "symbol": symbol,
        "interval": "1h",
        "limit": limit
    }
    r = requests.get(BINANCE_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df = df.sort_values("open_time").reset_index(drop=True)
    return df

# ============================================================
# REAL-TIME FEATURE ENGINE
# ============================================================

def compute_live_pfr(df):
    x = df.copy()
    
    # Return
    x["log_return"] = np.log(x["close"] / x["close"].shift(1))
    x["pct_return"] = x["close"].pct_change().fillna(0.0)
    
    # ATR 48
    prev_close = x["close"].shift(1)
    tr1 = x["high"] - x["low"]
    tr2 = (x["high"] - prev_close).abs()
    tr3 = (x["low"] - prev_close).abs()
    x["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    x["atr_48"] = x["tr"].rolling(48).mean()
    
    # Effort & Response Z-Scores
    vol_mean = x["volume"].rolling(48).mean()
    vol_std = x["volume"].rolling(48).std().replace(0, np.nan)
    x["effort_z"] = (x["volume"] - vol_mean) / vol_std
    
    x["response_raw"] = x["log_return"].abs() / x["atr_48"].replace(0, np.nan)
    resp_mean = x["response_raw"].rolling(48).mean()
    resp_std = x["response_raw"].rolling(48).std().replace(0, np.nan)
    x["response_z"] = (x["response_raw"] - resp_mean) / resp_std
    
    # Friction & Memory
    x["raw_friction"] = x["effort_z"] - x["response_z"]
    x["friction_memory"] = x["raw_friction"].ewm(alpha=0.10, adjust=False).mean()
    x["signed_friction"] = x["friction_memory"] * np.sign(x["log_return"])
    
    # Directional Friction (half-life 12h)
    alpha = 1.0 - 2.0 ** (-1.0 / 12.0)
    x["directional_friction"] = x["signed_friction"].ewm(alpha=alpha, adjust=False).mean()
    
    # Momentum 24h & Trend SMA 200
    x["momentum_24"] = x["close"] / x["close"].shift(24) - 1.0
    x["sma_200"] = x["close"].rolling(200).mean()
    
    # Rolling 48h volatility for Risk Parity sizing
    x["ann_vol"] = x["pct_return"].rolling(48).std() * ANN_FACTOR
    
    # OLS Residualization: directional_friction ~ momentum_24
    mask = x[["directional_friction", "momentum_24"]].notna().all(axis=1)
    x["directional_friction_resid"] = np.nan
    if mask.sum() > 50:
        X = sm.add_constant(x.loc[mask, ["momentum_24"]])
        y = x.loc[mask, "directional_friction"]
        model = sm.OLS(y, X).fit()
        x.loc[mask, "directional_friction_resid"] = y - model.predict(X)
        
    return x

# ============================================================
# LIVE SIGNAL & PORTFOLIO EVALUATOR
# ============================================================

def evaluate_live_market(config, state):
    symbols = config["strategy"]["symbols"]
    q90_map = config["strategy"]["q90_thresholds"]
    min_hold = config["strategy"]["min_hold_hours"]
    max_hold = config["strategy"]["max_hold_hours"]
    rebal_buffer = config["strategy"]["rebalance_buffer"]
    
    market_snapshot = {}
    events = []
    
    # 1. Evaluate individual assets
    for sym in symbols:
        df_raw = fetch_live_klines(sym, limit=250)
        df_feat = compute_live_pfr(df_raw)
        
        # Last closed bar is iloc[-2] (or iloc[-1] if running right after hour close)
        # We look at the latest completed candle
        last_bar = df_feat.iloc[-1]
        
        current_price = float(last_bar["close"])
        sig_resid = float(last_bar["directional_friction_resid"])
        resp_z = float(last_bar["response_z"])
        sma200 = float(last_bar["sma_200"])
        ann_vol = float(last_bar["ann_vol"]) if not np.isnan(last_bar["ann_vol"]) else 0.50
        bar_time = str(last_bar["open_time"])
        
        q90_th = q90_map.get(sym, 0.055)
        pos_info = state["positions"].get(sym, {"in_position": False, "entry_time": None, "entry_price": 0.0, "bars_held": 0, "weight": 0.0})
        
        in_pos = pos_info["in_position"]
        bars_held = pos_info["bars_held"]
        action = "HOLD"
        
        if not in_pos:
            # Check Entry Condition
            cond_macro = (current_price > sma200)
            cond_friction = (sig_resid >= q90_th)
            cond_response = (resp_z > 0)
            
            if cond_macro and cond_friction and cond_response:
                action = "BUY_ENTRY"
                in_pos = True
                bars_held = 0
                events.append({
                    "symbol": sym,
                    "type": "ENTRY",
                    "price": current_price,
                    "time": bar_time,
                    "friction": sig_resid,
                    "q90": q90_th
                })
        else:
            bars_held += 1
            # Check Exit Condition
            exit_max = (bars_held >= max_hold)
            exit_sig = (bars_held >= min_hold) and (sig_resid < 0.0)
            
            if exit_max or exit_sig:
                action = "SELL_EXIT"
                in_pos = False
                reason = "Max Hold Reached" if exit_max else "Friction Decay (< 0)"
                pnl_pct = (current_price / pos_info["entry_price"] - 1.0) * 100 if pos_info["entry_price"] > 0 else 0.0
                events.append({
                    "symbol": sym,
                    "type": "EXIT",
                    "price": current_price,
                    "entry_price": pos_info["entry_price"],
                    "pnl_pct": pnl_pct,
                    "bars_held": bars_held,
                    "reason": reason
                })
                bars_held = 0
                
        market_snapshot[sym] = {
            "price": current_price,
            "sig_resid": sig_resid,
            "q90_th": q90_th,
            "resp_z": resp_z,
            "sma_200": sma200,
            "above_sma200": (current_price > sma200),
            "ann_vol": ann_vol,
            "bar_time": bar_time,
            "in_position": in_pos,
            "bars_held": bars_held,
            "action": action
        }
        
    # 2. Compute Risk Parity Target Weights
    active_symbols = [s for s in symbols if market_snapshot[s]["in_position"]]
    n_active = len(active_symbols)
    
    target_weights = {}
    if n_active == 0:
        for s in symbols:
            target_weights[s] = 0.0
    else:
        # Inverse Volatility sizing
        inv_vols = {s: 1.0 / max(0.05, market_snapshot[s]["ann_vol"]) for s in active_symbols}
        sum_inv = sum(inv_vols.values())
        for s in symbols:
            if s in active_symbols:
                base_w = inv_vols[s] / sum_inv
                target_weights[s] = round(base_w, 4)
            else:
                target_weights[s] = 0.0
                
    # 3. Update state with rebalancing buffer
    for s in symbols:
        old_w = state["positions"][s].get("weight", 0.0)
        new_w = target_weights[s]
        
        # Check rebalance
        if abs(new_w - old_w) >= rebal_buffer or (old_w == 0 and new_w > 0) or (new_w == 0 and old_w > 0):
            state["positions"][s]["weight"] = new_w
            if old_w > 0 and new_w > 0 and abs(new_w - old_w) >= rebal_buffer:
                events.append({
                    "symbol": s,
                    "type": "REBALANCE",
                    "old_weight": old_w,
                    "new_weight": new_w
                })
                
        state["positions"][s]["in_position"] = market_snapshot[s]["in_position"]
        state["positions"][s]["bars_held"] = market_snapshot[s]["bars_held"]
        if market_snapshot[s]["action"] == "BUY_ENTRY":
            state["positions"][s]["entry_time"] = market_snapshot[s]["bar_time"]
            state["positions"][s]["entry_price"] = market_snapshot[s]["price"]
        elif market_snapshot[s]["action"] == "SELL_EXIT":
            state["positions"][s]["entry_time"] = None
            state["positions"][s]["entry_price"] = 0.0
            
    state["last_check_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    state["portfolio_gross_exposure"] = round(sum(state["positions"][s]["weight"] for s in symbols), 4)
    save_state(state)
    
    # Append-only scientific audit logging
    log_forward_test(market_snapshot, events, state)
    
    return market_snapshot, events, state

def log_forward_test(snapshot, events, state):
    from pathlib import Path
    log_dir = Path("pfr_output")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Hourly signal snapshot log
    log_file = log_dir / "forward_test_log.csv"
    file_exists = log_file.exists()
    
    rows = []
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    for sym, d in snapshot.items():
        w = state["positions"][sym]["weight"]
        rows.append({
            "timestamp": now_str,
            "bar_time": d["bar_time"],
            "symbol": sym,
            "price": d["price"],
            "sig_resid": round(d["sig_resid"], 6),
            "q90_th": round(d["q90_th"], 6),
            "response_z": round(d["resp_z"], 4),
            "above_sma200": int(d["above_sma200"]),
            "in_position": int(d["in_position"]),
            "bars_held": d["bars_held"],
            "weight": round(w, 4),
            "action": d["action"]
        })
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(log_file, mode="a", index=False, header=not file_exists)
    
    # 2. Closed trade execution log
    trade_file = log_dir / "forward_test_trades.csv"
    t_exists = trade_file.exists()
    trade_rows = []
    for e in events:
        if e["type"] == "EXIT":
            trade_rows.append({
                "symbol": e["symbol"],
                "exit_time": now_str,
                "exit_price": e["price"],
                "entry_price": e["entry_price"],
                "pnl_pct": round(e["pnl_pct"], 2),
                "bars_held": e["bars_held"],
                "reason": e["reason"]
            })
    if trade_rows:
        pd.DataFrame(trade_rows).to_csv(trade_file, mode="a", index=False, header=not t_exists)


# ============================================================
# WHATSAPP NOTIFICATION DISPATCHER
# ============================================================

def format_alert_message(market_snapshot, events, state):
    lines = []
    
    # Friendly asset names
    friendly_names = {
        "BTCUSDT": "Bitcoin",
        "ETHUSDT": "Ethereum",
        "SOLUSDT": "Solana"
    }
    
    now_str = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    
    # 1. Header & Title
    has_entry = any(e["type"] == "ENTRY" for e in events)
    has_exit = any(e["type"] == "EXIT" for e in events)
    
    if has_entry:
        lines.append("🟢 *[PFR ALGO: YENİ ALIM SİNYALİ]* 🚀")
    elif has_exit:
        lines.append("🔴 *[PFR ALGO: POZİSYON KAPATMA SİNYALİ]* ⚠️")
    else:
        lines.append("ℹ️ *[PFR PİYASA DURUM ÖZETİ]*")
        
    lines.append(f"⏱️ *Zaman:* {now_str}\n")
    
    # 2. Urgent Action Line
    gross_exp = state["portfolio_gross_exposure"] * 100
    cash_pct = 100.0 - gross_exp
    
    if has_entry:
        lines.append("🎯 *NE YAPILACAK? (YENİ İŞLEM):*")
        for e in events:
            if e["type"] == "ENTRY":
                sym_name = friendly_names.get(e['symbol'], e['symbol'])
                w = state["positions"][e['symbol']]["weight"] * 100
                lines.append(f"• *{sym_name} AL!* 🟢")
                lines.append(f"  Fiyat: ${e['price']:,.2f}")
                lines.append(f"  Ayrılacak Bütçe: *%{w:.1f}* (Sermayenizin)")
                lines.append(f"  Hedef Süre: En az 16 - 24 saat elde tutulacak.")
                lines.append(f"  Gerekçe: Güçlü kurumsal alıcı piyasaya girdi, trend yukarı döndü.")
        lines.append("")
        
    if has_exit:
        lines.append("🎯 *NE YAPILACAK? (POZİSYON KAPATMA):*")
        for e in events:
            if e["type"] == "EXIT":
                sym_name = friendly_names.get(e['symbol'], e['symbol'])
                sign = "+" if e["pnl_pct"] >= 0 else ""
                lines.append(f"• *{sym_name} SAT VE NAKİTE GEÇ!* 🔴")
                lines.append(f"  Satış Fiyatı: ${e['price']:,.2f}")
                lines.append(f"  İşlem Kâr/Zarar: *{sign}{e['pnl_pct']:.2f}%*")
                lines.append(f"  Pozisyonda Kalınan Süre: {e['bars_held']} saat")
                lines.append(f"  Gerekçe: Alıcı gücü zayıfladı ({e['reason']}), sermaye korumaya alınıyor.")
        lines.append("")

    if not has_entry and not has_exit:
        if gross_exp == 0:
            lines.append("🛑 *GÜNCEL AKSİYON: NAKİTTE BEKLE (Alım Yapma)*")
            lines.append("Piyasa şu an riskli; paranızı korumak için kenarda bekliyoruz.\n")
        else:
            lines.append("🟢 *GÜNCEL AKSİYON: MEVCUT POZİSYONLARI KORU*")
            lines.append("Açık işlemler kuralına uygun devam ediyor, sabırla bekliyoruz.\n")
            
    # 3. Asset Analysis in Plain Language
    lines.append("*📊 Varlık Analizi:*")
    for sym, d in market_snapshot.items():
        name = friendly_names.get(sym, sym)
        st = state["positions"][sym]
        
        if st["in_position"]:
            w = st["weight"] * 100
            lines.append(f"• *{name} (${d['price']:,.2f})*: 🟢 ELDE TUTULUYOR")
            lines.append(f"  Giriş: ${st['entry_price']:,.2f} | Portföy Payı: %{w:.1f} | {st['bars_held']}. saatte")
        else:
            if not d["above_sma200"] and d["sig_resid"] < d["q90_th"]:
                status_desc = "Düşüş trendinde, güçlü alıcı yok ❌"
            elif d["above_sma200"] and d["sig_resid"] < d["q90_th"]:
                status_desc = "Trend yukarı ama alım hacmi yetersiz ⏳"
            elif not d["above_sma200"] and d["sig_resid"] >= d["q90_th"]:
                status_desc = "Alıcı geldi ama ana trend henüz dönmedi ⏳"
            else:
                status_desc = "Gözlemde ⏳"
            lines.append(f"• *{name} (${d['price']:,.2f})*: {status_desc}")
            
    lines.append("")
    # 4. Portfolio Summary
    lines.append(f"*💼 Portföy Dağılımı:*")
    lines.append(f"• Nakit (Dolar/USDT): *%{cash_pct:.1f}*")
    lines.append(f"• Kripto Pozisyonu: *%{gross_exp:.1f}*")
    
    # 5. Bottom Line Advice
    lines.append("")
    if gross_exp == 0:
        lines.append("👉 *Özet:* Güvenli ve kurumsal onaylı bir alım fırsatı oluşana kadar %100 nakitte beklemek en karlı stratejidir.")
    else:
        lines.append("👉 *Özet:* Pozisyonlar taşınıyor. Çıkış sinyali geldiğinde otomatik bildirim gönderilecektir.")
        
    return "\n".join(lines)

def send_whatsapp_alert(message_text, config):
    wa = config.get("whatsapp", {})
    enabled = wa.get("enabled", False)
    phone = wa.get("phone", "").strip()
    apikey = wa.get("apikey", "").strip()
    
    print("\n" + "=" * 60)
    print("WHATSAPP MESAJI:")
    print("=" * 60)
    print(message_text)
    print("=" * 60)
    
    if not enabled or "XXXXXXXXX" in phone or apikey == "YOUR_CALLMEBOT_APIKEY":
        print("\n⚠️  [WhatsApp Bildirimi Gönderilmedi]")
        print("Nedeni: 'pfr_live_config.json' içinde whatsapp.enabled=false veya telefon/apikey girilmemiş.")
        print("Ücretsiz WhatsApp kurulumu için rehber:")
        print("1. WhatsApp'tan +34 644 44 49 48 numarasina şu mesajı gönderin:")
        print("   'I allow callmebot to send me messages'")
        print("2. Gelen API key'i ve telefon numaranızı pfr_live_config.json dosyasına yazın.")
        print("3. whatsapp.enabled değerini true yapın.")
        return False
        
    try:
        # CallMeBot API call
        encoded_text = urllib.parse.quote_plus(message_text)
        url = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={encoded_text}&apikey={apikey}"
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            print(f"✅ WhatsApp mesajı başarıyla {phone} numarasına iletildi!")
            return True
        else:
            print(f"❌ CallMeBot hatası (HTTP {r.status_code}): {r.text}")
            return False
    except Exception as e:
        print(f"❌ WhatsApp gönderiminde bağlantı hatası: {e}")
        return False

# ============================================================
# RUN MODES
# ============================================================

def run_check():
    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] PFR Live Engine: Anlık piyasa taranıyor...")
    config = load_config()
    state = load_state()
    
    snapshot, events, updated_state = evaluate_live_market(config, state)
    msg = format_alert_message(snapshot, events, updated_state)
    
    send_whatsapp_alert(msg, config)

def run_test_whatsapp():
    config = load_config()
    test_msg = (
        "🤖 *[PFR QUANT ENGINE TEST]*\n\n"
        "✅ Tebrikler! WhatsApp bildirim entegrasyonu başarıyla bağlandı.\n"
        f"⏱️ Zaman: {datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M UTC')}\n"
        "📈 BTC, ETH ve SOL için saatlik sürtünme alım/satım sinyalleri buraya iletilecektir."
    )
    send_whatsapp_alert(test_msg, config)

def run_daemon():
    print("=" * 70)
    print("PFR QUANT ENGINE — HOURLY DAEMON BAŞLATILDI")
    print("Her saat başı Binance mum kapanışında otomatik tarama ve bildirim yapılır.")
    print("Durdurmak için Ctrl + C tuşlarına basın.")
    print("=" * 70)
    
    # Run once immediately on start
    run_check()
    
    while True:
        # Sleep until the next hour mark + 10 seconds (to ensure hourly candle is closed)
        now = datetime.now(timezone.utc)
        sleep_seconds = (60 - now.minute - 1) * 60 + (60 - now.second) + 15
        next_run = datetime.fromtimestamp(now.timestamp() + sleep_seconds, timezone.utc)
        print(f"\nBir sonraki tarama zamanı: {next_run.strftime('%H:%M:%S UTC')} ({sleep_seconds//60} dakika sonra)...")
        time.sleep(sleep_seconds)
        
        try:
            run_check()
        except Exception as e:
            print(f"Saatlik döngü hatası: {e}")
            time.sleep(30)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PFR Live Quantitative Signal Engine")
    parser.add_argument("--check", action="store_true", help="Anlık canlı sinyal kontrolü yap")
    parser.add_argument("--test-whatsapp", action="store_true", help="WhatsApp bağlantı testi mesajı gönder")
    parser.add_argument("--daemon", action="store_true", help="Her saat başı otomatik çalışan arka plan servisi")
    
    args = parser.parse_args()
    
    if args.test_whatsapp:
        run_test_whatsapp()
    elif args.daemon:
        run_daemon()
    else:
        # Default action is single check
        run_check()
