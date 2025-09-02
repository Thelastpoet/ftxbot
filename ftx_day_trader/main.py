#!/usr/bin/env python3
"""
forex_bot.py
A pure-price-action breakout day-trading bot for MetaTrader 5.
Author: <you>
"""
import asyncio
import json
import logging
import argparse
import sys
from datetime import datetime, time
from decimal import Decimal
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not installed: pip install MetaTrader5")
    sys.exit(1)

# =============================================================================
# 0. Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("Bot")

# =============================================================================
# 1. Configuration helpers
# =============================================================================
DEFAULT_CONFIG = {
    "symbols": [
        {"name": "EURUSD", "timeframes": ["M5", "M15"]},
        {"name": "GBPUSD", "timeframes": ["M5", "M15"]},
    ],
    "strategy": {
        "lookback_bars": 250,          # how far back to look for swings
        "order": 5,                    # argrelextrema sensitivity
        "proximity_pips": 15,          # merge levels within this distance
        "min_rank": 3,                 # min swing-cluster size for a level
        "breakout_buffer_pips": 2,     # extra distance beyond level
        "trend_ema_fast": 21,          # fast EMA for trend filter
        "trend_ema_slow": 50,          # slow EMA for trend filter
    },
    "risk_management": {
        "risk_per_trade": 0.01,        # 1 % of equity
        "sl_buffer_pips": 10,          # stop-loss beyond level
        "rr_ratio": 2.0,               # risk-reward
    },
    "trading_sessions": [
        {"name": "London", "start": "08:00", "end": "17:00"},
        {"name": "NewYork", "start": "13:00", "end": "22:00"},
    ],
    "main_loop_interval_seconds": 5,
}

TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

def load_config() -> Dict:
    """Load config.json if present, else return DEFAULT_CONFIG."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-per-trade", type=float)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        log.warning("config.json not found – using built-in defaults.")
        cfg = DEFAULT_CONFIG

    # CLI overrides
    if args.risk_per_trade:
        cfg["risk_management"]["risk_per_trade"] = args.risk_per_trade
    return cfg

CONFIG = load_config()

# =============================================================================
# 2. MetaTrader5 API wrapper
# =============================================================================
class MT5Client:
    """Thin wrapper around MetaTrader5 package."""
    def __init__(self):
        if not mt5.initialize():
            log.error("MT5 initialization failed: %s", mt5.last_error())
            sys.exit(1)
        log.info("MT5 initialized. Account %s", mt5.account_info().login)

    def shutdown(self):
        mt5.shutdown()

    def symbol_info(self, symbol: str):
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol {symbol} not found")
        return info

    def get_rates(self, symbol: str, timeframe: int, count: int) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            raise RuntimeError(f"Cannot fetch data for {symbol}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df

    def place_order(self, *, symbol: str, order_type: int,
                    volume: float, price: float,
                    sl: float, tp: float,
                    comment: str = "forex_bot") -> bool:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("Order failed: %s (%s)", result.comment, result.retcode)
            return False
        log.info("Order placed: %s %s %.2f lots @%.5f sl=%.5f tp=%.5f",
                 symbol, "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL",
                 volume, price, sl, tp)
        return True

    def positions_get(self, symbol: str):
        return mt5.positions_get(symbol=symbol)

    def close_position(self, ticket: int, volume: float, symbol: str, order_type: int):
        price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask
        self.place_order(symbol=symbol, order_type=mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                         volume=volume, price=price, sl=0, tp=0, comment="close")

# =============================================================================
# 3. Strategy
# =============================================================================
class PurePriceActionStrategy:
    """Breakout strategy based on horizontal support/resistance."""
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _pips_to_price(pips: int, symbol_info) -> float:
        return pips * symbol_info.point

    def _levels(self, df: pd.DataFrame, symbol_info) -> Tuple[np.ndarray, np.ndarray]:
        """Return resistance and support levels (price arrays)."""
        cfg = self.cfg
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        order = cfg["order"]

        # swings
        highs_idx = argrelextrema(high, np.greater_equal, order=order)[0]
        lows_idx = argrelextrema(low, np.less_equal, order=order)[0]

        # price values
        highs = high[highs_idx]
        lows = low[lows_idx]

        # clustering helper
        def cluster(prices: np.ndarray) -> np.ndarray:
            if len(prices) == 0:
                return np.array([])
            prices = np.sort(prices)
            buffer = self._pips_to_price(cfg["proximity_pips"], symbol_info)
            clusters = []
            curr = [prices[0]]
            for p in prices[1:]:
                if abs(p - curr[-1]) <= buffer:
                    curr.append(p)
                else:
                    if len(curr) >= cfg["min_rank"]:
                        clusters.append(np.mean(curr))
                    curr = [p]
            if len(curr) >= cfg["min_rank"]:
                clusters.append(np.mean(curr))
            return np.array(clusters)

        resistances = cluster(highs)
        supports = cluster(lows)
        return resistances, supports

    def _trend(self, df: pd.DataFrame) -> str:
        """Return 'up', 'down', or 'sideways'."""
        fast = df["close"].ewm(span=self.cfg["trend_ema_fast"]).mean().iloc[-1]
        slow = df["close"].ewm(span=self.cfg["trend_ema_slow"]).mean().iloc[-1]
        if fast > slow:
            return "up"
        if fast < slow:
            return "down"
        return "sideways"

    # -------------------------------------------------------------------------
    # Main signal generator
    # -------------------------------------------------------------------------
    def signal(self, df: pd.DataFrame, symbol_info) -> Optional[Dict]:
        resistances, supports = self._levels(df, symbol_info)
        trend = self._trend(df)
        last = df.iloc[-1]
        close = last["close"]
        buffer = self._pips_to_price(self.cfg["breakout_buffer_pips"], symbol_info)

        # bullish breakout above resistance
        for r in resistances:
            if close > r + buffer and trend == "up":
                sl = r - self._pips_to_price(self.cfg["sl_buffer_pips"], symbol_info)
                tp = close + (close - sl) * self.cfg["rr_ratio"]
                return dict(direction="buy", entry=close, sl=sl, tp=tp)

        # bearish breakout below support
        for s in supports:
            if close < s - buffer and trend == "down":
                sl = s + self._pips_to_price(self.cfg["sl_buffer_pips"], symbol_info)
                tp = close - (sl - close) * self.cfg["rr_ratio"]
                return dict(direction="sell", entry=close, sl=sl, tp=tp)
        return None

# =============================================================================
# 4. Risk Manager
# =============================================================================
class RiskManager:
    def __init__(self, cfg: Dict):
        self.risk = cfg["risk_per_trade"]
        self.cfg = cfg

    def get_volume(self, symbol_info, sl_price: float, entry_price: float) -> float:
        """Return lot size rounded to symbol volume step."""
        risk_amount = mt5.account_info().equity * self.risk
        sl_distance = abs(entry_price - sl_price)
        if sl_distance == 0:
            return 0.0
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        volume = risk_amount / (sl_distance / tick_size * tick_value)
        step = symbol_info.volume_step
        volume = max(symbol_info.volume_min, round(volume / step) * step)
        return volume

# =============================================================================
# 5. Trading Session
# =============================================================================
class TradingSession:
    def __init__(self):
        self.mt5 = MT5Client()
        self.cfg = CONFIG
        self.strategy = PurePriceActionStrategy(self.cfg["strategy"])
        self.risk = RiskManager(self.cfg["risk_management"])

    # -------------------------------------------------------------------------
    def _in_session(self) -> bool:
        now = datetime.now().time()
        for sess in self.cfg["trading_sessions"]:
            start = time.fromisoformat(sess["start"])
            end = time.fromisoformat(sess["end"])
            if start <= now <= end:
                return True
        return False

    # -------------------------------------------------------------------------
    async def _process_symbol(self, sym_cfg: Dict):
        symbol = sym_cfg["name"]
        sym_info = self.mt5.symbol_info(symbol)
        for tf_str in sym_cfg["timeframes"]:
            tf = TIMEFRAME_MAP[tf_str]
            df = self.mt5.get_rates(symbol, tf, self.cfg["strategy"]["lookback_bars"])
            sig = self.strategy.signal(df, sym_info)
            if not sig:
                continue
            # check if already in position
            if self.mt5.positions_get(symbol):
                continue
            if not self._in_session():
                continue

            volume = self.risk.get_volume(sym_info, sig["sl"], sig["entry"])
            if volume == 0:
                continue

            order_type = mt5.ORDER_TYPE_BUY if sig["direction"] == "buy" else mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(symbol)
            price = tick.ask if sig["direction"] == "buy" else tick.bid
            self.mt5.place_order(symbol=symbol, order_type=order_type,
                                 volume=volume, price=price,
                                 sl=sig["sl"], tp=sig["tp"])

    # -------------------------------------------------------------------------
    async def run(self):
        log.info("Starting trading loop...")
        try:
            while True:
                for sym_cfg in self.cfg["symbols"]:
                    await self._process_symbol(sym_cfg)
                await asyncio.sleep(self.cfg["main_loop_interval_seconds"])
        except asyncio.CancelledError:
            log.info("Cancelled – shutting down...")
        finally:
            self.mt5.shutdown()

# =============================================================================
# 6. Entry point
# =============================================================================
if __name__ == "__main__":
    try:
        session = TradingSession()
        asyncio.run(session.run())
    except KeyboardInterrupt:
        log.info("User interrupt – exiting.")