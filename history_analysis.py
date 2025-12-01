"""
Historical market analysis utilities for per-symbol enrichment.

This module is intentionally decoupled from the main trading loop so it can be
wired in gradually.  It exposes a HistoricalAnalyzer that can be invoked with
an arbitrary market-data fetcher to compute ADR exhaustion, Donchian channels,
macro levels, and recent breakout statistics on a per-symbol basis.

Typical usage (pseudo-code):

    analyzer = HistoricalAnalyzer(config)
    snapshot = analyzer.refresh_symbol(
        symbol="EURUSD",
        fetcher=lambda sym, tf, bars: market_data.fetch_data_sync(sym, tf, bars),
        pip_size=0.0001,
        intraday_df=current_tf_df,
    )

Snapshots can later be consumed by the strategy/risk manager without running
heavy recomputations each loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, Optional, Tuple

import logging
import pandas as pd


HistoryFetcher = Callable[[str, str, int], Optional[pd.DataFrame]]

logger = logging.getLogger(__name__)


@dataclass
class HistoricalSnapshot:
    """Container for per-symbol historical metrics."""

    symbol: str
    generated_at: datetime
    params: Dict[str, Any]
    adr_value: Optional[float] = None
    adr_progress: Optional[float] = None
    adr_exhausted: bool = False
    donchian_upper: Optional[float] = None
    donchian_lower: Optional[float] = None
    donchian_mid: Optional[float] = None
    trend_bias: Optional[str] = None
    macro_high: Optional[float] = None
    macro_low: Optional[float] = None
    breakout_success_rate: Optional[float] = None
    breakout_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable representation."""
        return {
            "symbol": self.symbol,
            "generated_at": self.generated_at.isoformat(),
            "params": self.params,
            "adr_value": self.adr_value,
            "adr_progress": self.adr_progress,
            "adr_exhausted": self.adr_exhausted,
            "donchian_upper": self.donchian_upper,
            "donchian_lower": self.donchian_lower,
            "donchian_mid": self.donchian_mid,
            "trend_bias": self.trend_bias,
            "macro_high": self.macro_high,
            "macro_low": self.macro_low,
            "breakout_success_rate": self.breakout_success_rate,
            "breakout_samples": self.breakout_samples,
            "metadata": self.metadata,
        }


class HistoricalAnalyzer:
    """Compute per-symbol historical metrics with configurable knobs."""

    def __init__(self, config: Any):
        """
        Args:
            config: Either the global Config object or raw config dict.
                    Used purely to resolve defaults and per-symbol overrides.
        """
        self.config = config
        self._cache: Dict[str, HistoricalSnapshot] = {}
        self._global_defaults = self._extract_global_defaults(config)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def refresh_symbol(
        self,
        symbol: str,
        fetcher: HistoryFetcher,
        *,
        pip_size: Optional[float] = None,
        intraday_df: Optional[pd.DataFrame] = None,
    ) -> Optional[HistoricalSnapshot]:
        """
        Build/update the HistoricalSnapshot for `symbol`.

        Args:
            symbol: MT5 symbol name.
            fetcher: Callable that returns a pandas DataFrame indexed by time,
                     with columns ['open','high','low','close'] for the
                     requested timeframe and bar count.
            pip_size: Optional pip size override; defaults to symbol config or
                     generic 0.0001 if unavailable.
            intraday_df: Optional DataFrame for the trading timeframe so ADR
                     exhaustion can consider the current day's range progress.
        """
        params = self._resolve_symbol_params(symbol)
        if not params.get("enabled", False):
            return None

        daily_tf = params.get("daily_timeframe", "D1")
        macro_bars = int(params.get("macro_window_bars", 240))
        try:
            daily_df_raw = fetcher(symbol, daily_tf, macro_bars + 1)
        except Exception as exc:
            logger.warning("%s: failed to fetch %s data (%s)", symbol, daily_tf, exc)
            return None
        daily_df = self._prepare_price_df(
            daily_df_raw,
            symbol=symbol,
            timeframe=daily_tf,
        )
        if daily_df is None or len(daily_df) == 0:
            return None
        daily_closed = self._closed_bars(daily_df)
        if daily_closed is None or len(daily_closed) == 0:
            logger.warning("%s: no closed bars available for %s historical analysis", symbol, daily_tf)
            return None

        pip = pip_size or params.get("pip_size")
        if pip is None:
            logger.warning("%s: pip size missing for historical analysis", symbol)
            return None
        pip = float(pip)
        if pip <= 0:
            logger.warning("%s: invalid pip size %.6f for historical analysis", symbol, pip)
            return None

        adr_value, adr_progress, adr_exhausted = self._compute_adr(
            daily_closed,
            intraday_df=intraday_df,
            window=int(params.get("adr_window", 14)),
            exhaustion_pct=float(params.get("adr_exhaustion_pct", 0.9)),
        )

        donchian_upper, donchian_lower = self._compute_donchian(
            daily_closed, window=int(params.get("donchian_window", 55)), symbol=symbol
        )
        donchian_mid = (
            (donchian_upper + donchian_lower) / 2 if (donchian_upper and donchian_lower) else None
        )
        macro_high, macro_low = self._macro_extremes(
            daily_closed,
            window=int(params.get("macro_window_bars", len(daily_closed))),
            symbol=symbol,
        )

        trend_bias = self._determine_trend(
            daily_closed,
            swing_window=int(params.get("trend_swing_window", 5)),
            lookback_bars=int(params.get("trend_lookback_bars", 100)),
            min_swings=int(params.get("trend_min_swings", 3)),
            pip_size=pip,
            symbol=symbol,
        )

        breakout_rate, breakout_samples = self._simulate_breakouts(
            daily_closed,
            pip_size=pip,
            lookback=int(params.get("breakout_lookback_bars", 120)),
            lookahead=int(params.get("breakout_lookahead_bars", 5)),
            threshold_pips=float(params.get("breakout_threshold_pips", 5)),
            symbol=symbol,
        )
        min_samples = int(params.get("min_breakout_samples", 0) or 0)
        if breakout_samples and breakout_samples < min_samples:
            logger.debug(
                "%s: breakout sample size %s below minimum %s; suppressing rate",
                symbol,
                breakout_samples,
                min_samples,
            )
            breakout_rate = None

        snapshot = HistoricalSnapshot(
            symbol=symbol,
            generated_at=datetime.now(timezone.utc),
            params=params,
            adr_value=adr_value,
            adr_progress=adr_progress,
            adr_exhausted=adr_exhausted,
            donchian_upper=donchian_upper,
            donchian_lower=donchian_lower,
            donchian_mid=donchian_mid,
            trend_bias=trend_bias,
            macro_high=macro_high,
            macro_low=macro_low,
            breakout_success_rate=breakout_rate,
            breakout_samples=breakout_samples,
            metadata={
                "daily_timeframe": daily_tf,
                "pip_size": pip,
                "closed_bars": len(daily_closed),
            },
        )
        self._cache[symbol] = snapshot
        return snapshot

    def get_cached(self, symbol: str) -> Optional[HistoricalSnapshot]:
        """Return the last computed snapshot (if any)."""
        return self._cache.get(symbol)

    def resolve_params(self, symbol: str) -> Dict[str, Any]:
        """Public accessor for resolved symbol-specific parameters."""
        return self._resolve_symbol_params(symbol)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _extract_global_defaults(self, config: Any) -> Dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config.get("historical_analysis", {}))
        return dict(getattr(config, "historical_analysis", {}) or {})

    def _resolve_symbol_params(self, symbol: str) -> Dict[str, Any]:
        params = dict(self._global_defaults)
        if not symbol:
            return params

        symbol_entry = None
        symbols = None
        if isinstance(self.config, dict):
            symbols = self.config.get("symbols")
        else:
            symbols = getattr(self.config, "symbols", None)

        if symbols:
            for entry in symbols:
                name = entry.get("name") if isinstance(entry, dict) else None
                if name == symbol:
                    symbol_entry = entry
                    break

        if symbol_entry:
            # pip_unit already used elsewhere in the project; reuse it if present
            if "pip_unit" in symbol_entry:
                params.setdefault("pip_size", symbol_entry.get("pip_unit"))
            symbol_hist = symbol_entry.get("historical_analysis", {})
            if symbol_hist:
                params.update({k: v for k, v in symbol_hist.items() if v is not None})
        return params

    def _prepare_price_df(
        self,
        df: Optional[pd.DataFrame],
        *,
        symbol: str,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        if df is None or len(df) == 0:
            logger.warning("%s: no data returned for timeframe %s", symbol, timeframe)
            return None
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            logger.warning("%s: data for %s missing columns %s", symbol, timeframe, sorted(missing))
            return None
        cleaned = df.copy()
        if not cleaned.index.is_monotonic_increasing:
            cleaned = cleaned.sort_index()
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
        cleaned = cleaned.dropna(subset=list(required))
        if cleaned.empty:
            logger.warning("%s: dataframe empty after cleaning for %s", symbol, timeframe)
            return None
        return cleaned

    def _closed_bars(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or len(df) == 0:
            return None
        if len(df) == 1:
            return df.copy()
        return df.iloc[:-1].copy()

    def _compute_adr(
        self,
        df: pd.DataFrame,
        *,
        intraday_df: Optional[pd.DataFrame],
        window: int,
        exhaustion_pct: float,
    ) -> Tuple[Optional[float], Optional[float], bool]:
        if len(df) < window:
            logger.debug("ADR window %s larger than data length %s", window, len(df))
            return None, None, False
        ranges = (df["high"] - df["low"]).astype(float)
        adr = float(ranges.tail(window).mean())
        if adr <= 0:
            return adr, None, False

        progress = None
        if intraday_df is not None and len(intraday_df) > 0:
            day_range = float(
                intraday_df["high"].max() - intraday_df["low"].min()
            )
            progress = day_range / adr if adr else None

        exhausted = bool(progress is not None and progress >= exhaustion_pct)
        return adr, progress, exhausted

    def _compute_donchian(
        self, df: pd.DataFrame, *, window: int, symbol: str
    ) -> Tuple[Optional[float], Optional[float]]:
        if window <= 0 or len(df) < window:
            logger.warning("%s: insufficient data (%s bars) for Donchian window %s", symbol, len(df), window)
            return None, None
        highs = df["high"].rolling(window=window).max()
        lows = df["low"].rolling(window=window).min()
        upper = float(highs.iloc[-1]) if not pd.isna(highs.iloc[-1]) else None
        lower = float(lows.iloc[-1]) if not pd.isna(lows.iloc[-1]) else None
        return upper, lower

    def _macro_extremes(
        self, df: pd.DataFrame, *, window: int, symbol: str
    ) -> Tuple[Optional[float], Optional[float]]:
        if len(df) == 0 or window <= 0:
            return None, None
        subset = df.tail(window)
        if len(subset) == 0:
            logger.warning("%s: no data within macro window %s", symbol, window)
            return None, None
        highs = subset["high"].astype(float)
        lows = subset["low"].astype(float)
        return float(highs.max()), float(lows.min())

    def _find_swing_indices(self, series: pd.Series, window: int) -> list:
        """Find swing high/low indices in a series."""
        idxs = []
        values = series.values
        n = len(values)
        for i in range(window, n - window):
            left = values[i - window:i + 1]
            right = values[i:i + window + 1]

            if values[i] >= left.max() and values[i] >= right.max():
                idxs.append(i)
        return idxs

    def _determine_trend(
        self,
        df: pd.DataFrame,
        *,
        swing_window: int,
        lookback_bars: int,
        min_swings: int,
        pip_size: float,
        symbol: str,
    ) -> Optional[str]:
        """
        Determine trend from actual market structure (swing points).

        Uptrend: Series of higher highs AND higher lows
        Downtrend: Series of lower highs AND lower lows
        Neutral: Mixed structure, ranging, or insufficient data
        """
        if len(df) < swing_window * 2 + 1:
            return None

        # Find swing points using same logic as strategy
        highs = df['high']
        lows = df['low']

        swing_highs_idx = self._find_swing_indices(highs, swing_window)
        swing_lows_idx = self._find_swing_indices(-lows, swing_window)

        if not swing_highs_idx or not swing_lows_idx:
            logger.debug("%s: no swing points found for trend analysis", symbol)
            return None

        # Get recent swings only (last N bars)
        recent_highs = [i for i in swing_highs_idx if i >= len(df) - lookback_bars]
        recent_lows = [i for i in swing_lows_idx if i >= len(df) - lookback_bars]

        if len(recent_highs) < min_swings or len(recent_lows) < min_swings:
            logger.debug(
                "%s: insufficient swings for trend (highs=%s, lows=%s, need=%s)",
                symbol, len(recent_highs), len(recent_lows), min_swings
            )
            return None

        # Analyze structure of last N swings
        high_values = [float(df.iloc[i]['high']) for i in recent_highs[-min_swings:]]
        low_values = [float(df.iloc[i]['low']) for i in recent_lows[-min_swings:]]

        # Check if highs are ascending (higher highs)
        highs_ascending = all(high_values[i] > high_values[i-1] for i in range(1, len(high_values)))
        highs_descending = all(high_values[i] < high_values[i-1] for i in range(1, len(high_values)))

        # Check if lows are ascending (higher lows)
        lows_ascending = all(low_values[i] > low_values[i-1] for i in range(1, len(low_values)))
        lows_descending = all(low_values[i] < low_values[i-1] for i in range(1, len(low_values)))

        # Determine trend from market structure
        if highs_ascending and lows_ascending:
            logger.debug("%s: bullish trend (HH+HL)", symbol)
            return "bullish"  # Higher highs AND higher lows = uptrend
        elif highs_descending and lows_descending:
            logger.debug("%s: bearish trend (LH+LL)", symbol)
            return "bearish"  # Lower highs AND lower lows = downtrend
        else:
            logger.debug("%s: neutral/ranging (mixed structure)", symbol)
            return "neutral"  # Mixed structure = ranging/consolidating

    def _simulate_breakouts(
        self,
        df: pd.DataFrame,
        *,
        pip_size: float,
        lookback: int,
        lookahead: int,
        threshold_pips: float,
        symbol: str,
    ) -> Tuple[Optional[float], int]:
        if pip_size <= 0 or len(df) < lookback + lookahead + 5:
            logger.debug(
                "%s: insufficient data for breakout simulation (len=%s, lookback=%s, lookahead=%s)",
                symbol,
                len(df),
                lookback,
                lookahead,
            )
            return None, 0

        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        closes = df["close"].astype(float)
        thr_price = threshold_pips * pip_size

        successes = 0
        total = 0

        prev_high = highs.rolling(window=lookback).max().shift(1)
        prev_low = lows.rolling(window=lookback).min().shift(1)

        for idx in range(lookback + 1, len(df) - lookahead):
            level_high = prev_high.iloc[idx]
            level_low = prev_low.iloc[idx]
            price = closes.iloc[idx]
            if pd.isna(level_high) or pd.isna(level_low):
                continue

            future_slice = slice(idx + 1, idx + 1 + lookahead)
            future_high = highs.iloc[future_slice].max()
            future_low = lows.iloc[future_slice].min()

            if price > level_high + thr_price:
                total += 1
                if future_high >= price + thr_price:
                    successes += 1
                continue
            if price < level_low - thr_price:
                total += 1
                if future_low <= price - thr_price:
                    successes += 1

        rate = (successes / total) if total else None
        return rate, total
