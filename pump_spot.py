#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binance Spot Pump Bot 
"""

import os
import csv
import json
import time
import math
import atexit
import logging
import datetime
import traceback
import random
from decimal import Decimal, ROUND_DOWN
from logging.handlers import TimedRotatingFileHandler

import numpy as np
from talib import RSI, ROC
from functools import wraps

from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Create pump_spot directory if it doesn't exist
PUMP_SPOT_DIR = "pump_spot"
os.makedirs(PUMP_SPOT_DIR, exist_ok=True)

# ---------------- CONFIG ----------------
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Mode (safe by default)
PAPER_TRADING = True  # Set to False for real trading

# Trading settings
APPROVED = None          # None = allow all allowed USDT pairs above min volume
MIN_VOLUME = 2_000_000   # 24h quote volume threshold
MIN_PUMP = 2.0           # 2% daily move
TAKE_PROFIT = 0.02       # +2%
STOP_LOSS = 0.01         # -1%
MAX_RISK = 0.02          # Risk 2% of USDT balance per trade
MAX_SLIPPAGE = 0.005     # 0.5% max slippage at entry

# EMA confirmation
EMA_FAST = 20
EMA_SLOW = 50
EMA_CROSS_LOOKBACK = 3   # crossover must occur within last N bars

# Trade window
TRADE_WINDOW = 300       # seconds (5 min)
POLL_SEC = 5             # seconds between book/price polls (avoid 429)

# Symbols cache persistence
SYMBOLS_FILE = os.path.join(PUMP_SPOT_DIR, "symbols.json")
SYMBOLS_TTL = 900        # 15 minutes

# Open positions persistence
POSITIONS_FILE = os.path.join(PUMP_SPOT_DIR, "open_positions.json")

# Logging (rotate daily)
today_str = datetime.date.today().isoformat()
TRADE_LOG_CSV = os.path.join(PUMP_SPOT_DIR, f"trade_log_{today_str}.csv")
SKIPPED_LOG_CSV = os.path.join(PUMP_SPOT_DIR, f"skipped_log_{today_str}.csv")
RUNTIME_LOG = os.path.join(PUMP_SPOT_DIR, "run.log")

# ----------------------------------------

# ------------- Logging Setup -------------
logger = logging.getLogger("trader")
logger.setLevel(logging.INFO)

# File handler for persistent logs
_rot_handler = TimedRotatingFileHandler(RUNTIME_LOG, when="midnight", backupCount=14, encoding="utf-8")
_rot_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_rot_handler)

# Console handler to display output in terminal
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_console_handler)

def info(msg): logger.info(msg)
def warn(msg): logger.warning(msg)
def err(msg): logger.error(msg, exc_info=True)

def write_csv_row(path, header, row):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)
# ----------------------------------------

# ------------- Binance Client -----------
if not PAPER_TRADING and (not API_KEY or not API_SECRET):
    raise RuntimeError("Live trading requires API keys set in environment.")

client = Client(API_KEY, API_SECRET)

# -------- Retry Decorator --------
def _with_retry(
    attempts=5,
    base=1.0,
    max_wait=30.0,
    jitter=0.30,
    retry_exceptions=(BinanceAPIException, BinanceRequestException, Exception),
    logger_func=warn,
    label="binance call",
):
    def _decorator(fn):
        @wraps(fn)
        def _wrapped(*args, **kwargs):
            delay = base
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except KeyboardInterrupt:
                    raise
                except retry_exceptions as e:
                    if attempt >= attempts:
                        raise
                    sleep_s = min(delay, max_wait) * random.uniform(1 - jitter, 1 + jitter)
                    if logger_func:
                        logger_func(f"[retry] {label} failed (attempt {attempt}/{attempts - 1}): {e}. "
                                    f"Sleeping {sleep_s:.2f}s.")
                    time.sleep(sleep_s)
                    delay = min(delay * 2, max_wait)
        return _wrapped
    return _decorator

# Tenacity retry decorator for Binance calls
binance_retry = _with_retry(
    attempts=5,
    base=1.0,
    max_wait=30.0,
    jitter=0.30,
    retry_exceptions=(BinanceAPIException, BinanceRequestException, Exception),
    logger_func=warn,
    label="Binance API",
)
# ----------------------------------------

# ---------- Persistence Helpers ----------
def save_symbols(symbols, last_update):
    data = {"symbols": symbols, "last_update": last_update}
    tmp = SYMBOLS_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, SYMBOLS_FILE)
    except Exception as e:
        warn(f"Failed to persist symbols: {e}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def load_symbols():
    try:
        if os.path.isfile(SYMBOLS_FILE):
            with open(SYMBOLS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(s) for s in data], 0
            if isinstance(data, dict):
                syms = data.get("symbols", [])
                ts = data.get("last_update", 0)
                if not isinstance(syms, list):
                    syms = []
                syms = [str(s) for s in syms if isinstance(s, str)]
                if not isinstance(ts, (int, float)):
                    ts = 0
                return syms, ts
    except Exception as e:
        warn(f"Failed to load symbols.json, defaults used: {e}")
    return [], 0

def is_stale(ts, ttl=SYMBOLS_TTL):
    try:
        return (time.time() - float(ts)) > ttl
    except Exception:
        return True
# ----------------------------------------

# ------------- Position State -----------
OPEN_POSITIONS = {}  # symbol -> {"qty": float, "entry": float, "started_at": float}

def save_positions():
    try:
        tmp = POSITIONS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(OPEN_POSITIONS, f)
        os.replace(tmp, POSITIONS_FILE)
    except Exception as e:
        warn(f"Failed to persist OPEN_POSITIONS: {e}")
        
def check_existing_positions():
    """Monitor and exit any positions that were left open from previous run"""
    for symbol in list(OPEN_POSITIONS.keys()):
        try:
            price = safe_symbol_ticker_price(symbol)
            pos = OPEN_POSITIONS[symbol]
            entry = pos["entry"]
            qty = pos["qty"]
            
            pnl = (price - entry) / entry
            
            # Apply TP/SL to orphaned positions
            if pnl >= TAKE_PROFIT:
                sorder, spnl = market_sell(symbol, qty, entry)
                info(f"[ORPHAN] Take Profit: {symbol} +{spnl*100:.2f}%")
                log_trade([time.time(), symbol, "SELL_TP_ORPHAN", qty, price, spnl])
                del OPEN_POSITIONS[symbol]
            elif pnl <= -STOP_LOSS:
                sorder, spnl = market_sell(symbol, qty, entry)
                info(f"[ORPHAN] Stop Loss: {symbol} {spnl*100:.2f}%")
                log_trade([time.time(), symbol, "SELL_SL_ORPHAN", qty, price, spnl])
                del OPEN_POSITIONS[symbol]
            else:
                info(f"[ORPHAN] Monitoring {symbol}: PNL={pnl*100:.2f}%")
                
        except Exception as e:
            err(f"Error checking orphaned position {symbol}: {e}")
    
    save_positions()

def load_positions():
    global OPEN_POSITIONS
    try:
        if os.path.isfile(POSITIONS_FILE):
            with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                OPEN_POSITIONS = data
    except Exception as e:
        warn(f"Failed to load OPEN_POSITIONS: {e}")

atexit.register(save_positions)
load_positions()
# ----------------------------------------

# ---------- Utility & Filters -----------
def is_spot_usdt_symbol(s: str) -> bool:
    if not s.endswith("USDT"):
        return False
    # exclude leveraged/ETF-style tokens
    leveraged = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
    return not any(token == s or token in s for token in leveraged)

@binance_retry
def safe_get_symbol_info(symbol):
    return client.get_symbol_info(symbol)

def get_filters(symbol):
    info = safe_get_symbol_info(symbol)
    if not info:
        raise ValueError(f"No symbol info for {symbol}")

    lot = next((f for f in info["filters"] if f["filterType"] in ("MARKET_LOT_SIZE", "LOT_SIZE")), None)
    if not lot:
        raise ValueError(f"No LOT_SIZE/MARKET_LOT_SIZE for {symbol}")
    step = float(lot["stepSize"])
    min_qty = float(lot.get("minQty", "0"))
    max_qty = float(lot.get("maxQty", "0"))

    pf = next((f for f in info["filters"] if f["filterType"] == "PRICE_FILTER"), None)
    tick = float(pf["tickSize"]) if pf else 0.0

    nf = next((f for f in info["filters"] if f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL")), None)
    min_notional = float((nf or {}).get("minNotional", 0))
    apply_to_market = bool((nf or {}).get("applyToMarket", True))

    return {
        "step": step,
        "tick": tick,
        "min_qty": min_qty,
        "max_qty": max_qty,
        "min_notional": min_notional,
        "apply_to_market": apply_to_market,
    }

def adjust_qty(qty, step):
    """
    Floor to step multiple; clamp tiny floating artifacts; return 0 if below one step.
    """
    step = float(step)
    q = float(qty)
    if step <= 0:
        return max(0.0, q)
    floored = math.floor(q / step) * step
    # clamp to 8dp for safety
    floored = float(Decimal(str(floored)).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN))
    if 0 < floored < step:
        return 0.0
    return max(0.0, floored)

@binance_retry
def safe_get_balance(asset="USDT"):
    bal = client.get_asset_balance(asset=asset)
    return float(bal["free"])

@binance_retry
def safe_get_klines(symbol, interval="5m", limit=200):
    return client.get_klines(symbol=symbol, interval=interval, limit=limit)

@binance_retry
def safe_get_24h_ticker():
    return client.get_ticker()  # 24h change/volume array (python-binance wrapper)

@binance_retry
def safe_get_all_book_ticker():
    # Returns all best bid/ask in one shot
    return client.get_orderbook_ticker()

@binance_retry
def safe_symbol_ticker_price(symbol):
    t = client.get_symbol_ticker(symbol=symbol)
    return float(t["price"])

@binance_retry
def safe_market_buy_quote(symbol, quote_usdt_str):
    return client.create_order(symbol=symbol, side="BUY", type="MARKET", quoteOrderQty=quote_usdt_str)

@binance_retry
def safe_market_sell_qty(symbol, qty_str):
    return client.order_market_sell(symbol=symbol, quantity=qty_str)

def log_trade(row):
    write_csv_row(TRADE_LOG_CSV, ["timestamp", "symbol", "side", "qty", "price", "pnl"], row)

def log_skipped(row):
    write_csv_row(SKIPPED_LOG_CSV, ["timestamp", "symbol", "reason", "price", "extra"], row)
# ----------------------------------------

# ------------- Signals/Scanners ---------
def get_top_gainers(limit=10, min_volume=2_000_000, whitelist=None):
    tickers = safe_get_24h_ticker()
    gainers = []
    for t in tickers:
        s = t["symbol"]
        if not is_spot_usdt_symbol(s):
            continue
        if float(t["quoteVolume"]) < min_volume:
            continue
        if whitelist and s not in whitelist:
            continue
        gainers.append((s, float(t.get("priceChangePercent", 0.0))))
    gainers.sort(key=lambda x: x[1], reverse=True)
    return [g[0] for g in gainers[:limit]]

def get_pumping_spot(symbols):
    tickers = safe_get_24h_ticker()
    syms = set(symbols)
    pumps = []
    for t in tickers:
        s = t["symbol"]
        if s not in syms:
            continue
        try:
            if (
                float(t.get("priceChangePercent", 0.0)) > MIN_PUMP
                and float(t.get("quoteVolume", 0.0)) > MIN_VOLUME
            ):
                pumps.append(s)
        except Exception:
            continue
    return pumps

def get_last_closes(symbol, interval="5m", limit=200):
    klines = safe_get_klines(symbol, interval=interval, limit=limit)
    return np.array([float(k[4]) for k in klines], dtype=float)

def volume_spike_signal(symbol, threshold=3.0):
    """
    Detect if current volume is significantly higher than average
    threshold: multiplier (3.0 = 300% of average volume)
    """
    klines = safe_get_klines(symbol, interval="5m", limit=50)
    
    # Get volumes
    volumes = np.array([float(k[5]) for k in klines], dtype=float)
    
    # Current vs average volume
    current_vol = volumes[-1]
    avg_vol = np.mean(volumes[:-1])  # exclude current candle
    
    if avg_vol == 0:
        return False
    
    vol_ratio = current_vol / avg_vol
    
    # Volume must be 3x+ average
    return vol_ratio >= threshold

def rsi_momentum_signal(symbol):
    """
    Check if RSI shows strong momentum (50-80 range)
    """
    closes = get_last_closes(symbol, limit=100)
    
    if len(closes) < 14:
        return False
    
    rsi = RSI(closes, timeperiod=14)
    
    if np.isnan(rsi[-1]):
        return False
    
    # Strong momentum zone: 50-80
    # Above 80 = overextended, below 50 = weak
    return 50 < rsi[-1] < 80

def price_acceleration_signal(symbol, periods=5):
    """
    Detect if price is accelerating upward
    """
    closes = get_last_closes(symbol, limit=20)
    
    if len(closes) < periods + 1:
        return False
    
    # Rate of change
    roc = ((closes[-1] - closes[-periods-1]) / closes[-periods-1]) * 100
    
    # Must be gaining at least 1% in last 5 candles
    return roc > 1.0

def pump_signal(symbol):
    """
    Multi-factor pump detection:
    1. Volume spike (3x average) - PRIMARY
    2. Strong RSI momentum (50-80) - CONFIRMATION
    3. Price acceleration - CONFIRMATION
    """
    try:
        # PRIMARY: Volume must spike
        if not volume_spike_signal(symbol, threshold=3.0):
            return False
        
        # CONFIRMATION: RSI in momentum zone
        if not rsi_momentum_signal(symbol):
            return False
        
        # CONFIRMATION: Price accelerating
        if not price_acceleration_signal(symbol, periods=5):
            return False
        
        return True
        
    except Exception as e:
        warn(f"pump_signal error for {symbol}: {e}")
        return False
# ----------------------------------------

# ------------- Symbols Refresh ----------
def update_symbols(current_symbols, last_update):
    try:
        symbols = get_top_gainers(limit=5, min_volume=MIN_VOLUME, whitelist=APPROVED)
        ts = time.time()
        if symbols:
            save_symbols(symbols, ts)
            return symbols, ts
        else:
            warn("Fetched empty symbols list; keeping previous symbols.")
            return current_symbols, last_update
    except Exception as e:
        warn(f"update_symbols failed: {e}")
        return current_symbols, last_update
# ----------------------------------------

# ------------- Trading Actions ----------
def market_buy(symbol, usdt_amount, ask_price_snapshot):
    """
    BUY with quoteOrderQty. ask_price_snapshot is from bookTicker, used for slippage guard.
    """
    f = get_filters(symbol)
    # slippage guard vs the recent ask snapshot
    ref = safe_symbol_ticker_price(symbol)  # reference last trade price
    if ask_price_snapshot > ref * (1.0 + MAX_SLIPPAGE):
        log_skipped([time.time(), symbol, "Slippage too high", ask_price_snapshot, f"ref={ref}"])
        return None, None, None

    # notional guard
    min_notional_required = max(f["min_notional"], f["min_qty"] * ask_price_snapshot)
    if f["apply_to_market"] and usdt_amount < min_notional_required:
        log_skipped([time.time(), symbol, "Below min notional", ask_price_snapshot, 
                    f"usdt={usdt_amount}, required={min_notional_required:.2f}"])
        return None, None, None

    if PAPER_TRADING:
        qty = adjust_qty(usdt_amount / ask_price_snapshot, f["step"])
        entry = ask_price_snapshot
        info(f"[PAPER] Bought {qty} {symbol} at {entry}")
        return {"fills": [{"price": entry, "qty": qty}]}, qty, entry

    order = safe_market_buy_quote(symbol, str(usdt_amount))
    if order.get("fills"):
        qty = sum(float(x["qty"]) for x in order["fills"])
        entry = sum(float(x["price"]) * float(x["qty"]) for x in order["fills"]) / max(qty, 1e-12)
    else:
        executed_qty = float(order.get("executedQty", 0) or 0)
        quote_qty = float(order.get("cummulativeQuoteQty", 0) or 0)
        qty = executed_qty
        entry = quote_qty / max(executed_qty, 1e-12)

    qty = adjust_qty(qty, f["step"])
    return order, qty, entry

def market_sell(symbol, qty, entry):
    f = get_filters(symbol)
    qty = adjust_qty(qty, f["step"])
    # snapshot for paper pnl
    price = safe_symbol_ticker_price(symbol)

    if PAPER_TRADING:
        pnl = (price - entry) / entry
        info(f"[PAPER] Sold {qty} {symbol} at {price} (PNL {pnl*100:.2f}%)")
        return {"status": "FILLED"}, pnl

    # live market sell
    order = safe_market_sell_qty(symbol, str(qty))
    executed_qty = float(order.get("executedQty", 0) or 0)
    cummulative_quote = float(order.get("cummulativeQuoteQty", 0) or 0)
    avg_price = (cummulative_quote / max(executed_qty, 1e-12)) if executed_qty else price
    pnl = (avg_price - entry) / entry
    return order, pnl
# ----------------------------------------

# ----------------- Runner ----------------
SYMBOLS, last_update = load_symbols()
info(f"Loaded persisted SYMBOLS: {SYMBOLS}")

if (not SYMBOLS) or is_stale(last_update):
    SYMBOLS, last_update = update_symbols(SYMBOLS, last_update)
    info(f"Startup refresh SYMBOLS: {SYMBOLS}")

def run():
    global SYMBOLS, last_update, OPEN_POSITIONS
    
    # CHECK ORPHANED POSITIONS
    check_existing_positions()

    # refresh symbols if stale
    if is_stale(last_update) or not SYMBOLS:
        SYMBOLS, last_update = update_symbols(SYMBOLS, last_update)
        info(f"Updated SYMBOLS: {SYMBOLS}")

    # balance & risk sizing
    try:
        balance = safe_get_balance("USDT")
    except Exception:
        balance = 0.0
    risk_amount = balance * MAX_RISK
    info(f"Balance: {balance:.2f} USDT | Risk/trade: {risk_amount:.2f} USDT")

    # single snapshot of all bids/asks
    book = safe_get_all_book_ticker()
    prices_book = {}
    for d in book:
        try:
            s = d["symbol"]
            if is_spot_usdt_symbol(s):
                prices_book[s] = (float(d["bidPrice"]), float(d["askPrice"]))
        except Exception:
            continue

    # mid-window symbols refresh if needed
    if is_stale(last_update):
        SYMBOLS, last_update = update_symbols(SYMBOLS, last_update)

    # scan for pumps within current SYMBOLS
    pumps = get_pumping_spot(SYMBOLS)

    # process each pump
    for symbol in pumps:
        # EMA confirmation
        if not pump_signal(symbol):
            loop_price = prices_book.get(symbol, (0.0, 0.0))[1] or safe_symbol_ticker_price(symbol)
            log_skipped([time.time(), symbol, "Pump signal not confirmed", loop_price, ""])
            time.sleep(POLL_SEC)
            continue

        # avoid duplicate positions
        if symbol in OPEN_POSITIONS:
            loop_price = prices_book.get(symbol, (0.0, 0.0))[1] or safe_symbol_ticker_price(symbol)
            log_skipped([time.time(), symbol, "Already in position", loop_price, ""])
            time.sleep(POLL_SEC)
            continue

        # entry
        bid, ask = prices_book.get(symbol, (0.0, 0.0))
        if ask <= 0:
            # fallback to ticker if missing
            ask = safe_symbol_ticker_price(symbol)

        order, qty, entry = market_buy(symbol, risk_amount, ask)
        if not order:
            time.sleep(POLL_SEC)
            continue

        OPEN_POSITIONS[symbol] = {"qty": float(qty), "entry": float(entry), "started_at": time.time()}
        info(f"Entry {symbol} qty={qty} at {entry}")
        save_positions()

        # manage trade for up to TRADE_WINDOW seconds
        start = time.time()
        while time.time() - start < TRADE_WINDOW:
            # pacing to avoid 429
            time.sleep(POLL_SEC)

            # get fresh ask for pnl calc
            book = safe_get_all_book_ticker()
            prices_book = {d["symbol"]: (float(d["bidPrice"]), float(d["askPrice"])) for d in book if is_spot_usdt_symbol(d["symbol"])}
            cur_ask = prices_book.get(symbol, (0.0, 0.0))[1] or safe_symbol_ticker_price(symbol)

            pnl = (cur_ask - entry) / entry
            if pnl >= TAKE_PROFIT:
                sorder, spnl = market_sell(symbol, OPEN_POSITIONS[symbol]["qty"], entry)
                info(f"Take Profit: {symbol} +{spnl*100:.2f}%")
                log_trade([time.time(), symbol, "SELL_TP", OPEN_POSITIONS[symbol]["qty"], cur_ask, spnl])
                del OPEN_POSITIONS[symbol]
                save_positions()
                break
            elif pnl <= -STOP_LOSS:
                sorder, spnl = market_sell(symbol, OPEN_POSITIONS[symbol]["qty"], entry)
                info(f"Stop Loss: {symbol} {spnl*100:.2f}%")
                log_trade([time.time(), symbol, "SELL_SL", OPEN_POSITIONS[symbol]["qty"], cur_ask, spnl])
                del OPEN_POSITIONS[symbol]
                save_positions()
                break

        else:
            # time exit if while not broken
            # capture price once for consistent logging
            book = safe_get_all_book_ticker()
            prices_book = {d["symbol"]: (float(d["bidPrice"]), float(d["askPrice"])) for d in book if is_spot_usdt_symbol(d["symbol"])}
            loop_price = prices_book.get(symbol, (0.0, 0.0))[1] or safe_symbol_ticker_price(symbol)

            sorder, spnl = market_sell(symbol, OPEN_POSITIONS[symbol]["qty"], entry)
            info(f"Time Exit: {symbol} {spnl*100:.2f}%")
            log_trade([time.time(), symbol, "SELL_TIMEOUT", OPEN_POSITIONS[symbol]["qty"], loop_price, spnl])
            del OPEN_POSITIONS[symbol]
            save_positions()

        # brief pause before scanning next symbol
        time.sleep(POLL_SEC)

# -------------- Main Loop ---------------
if __name__ == "__main__":
    backoff = 60
    while True:
        try:
            run()
            backoff = 60
            time.sleep(60)  # outer cycle pace
        except KeyboardInterrupt:
            info("Shutting down by user request.")
            save_positions()
            break
        except Exception as e:
            err(f"Error in main loop: {e}")
            traceback.print_exc()
            info(f"Retrying in {backoff} sec...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 3600)
