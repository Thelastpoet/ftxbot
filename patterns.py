"""
TA-Lib Candlestick Pattern utilities for fast, robust confirmation.

We aggregate a curated set of TA-Lib candle patterns and convert them into a
simple directional score for the last few closed bars, suitable for intraday
breakout confirmation. Designed to be lightweight and fail-safe for live use.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd


# Map friendly names to TA-Lib function attribute names
_DEFAULT_PATTERNS: Dict[str, str] = {
    # Core reversals/continuations commonly effective intraday
    "engulfing": "CDLENGULFING",
    "hammer": "CDLHAMMER",
    "shooting_star": "CDLSHOOTINGSTAR",
    "harami": "CDLHARAMI",
    "three_inside": "CDL3INSIDE",
    "morning_star": "CDLMORNINGSTAR",
    "evening_star": "CDLEVENINGSTAR",
    "piercing": "CDLPIERCING",
    "dark_cloud": "CDLDARKCLOUDCOVER",
    "three_line_strike": "CDL3LINESTRIKE",
}


def _try_import_talib():
    try:
        import talib as ta  # type: ignore
        return ta
    except Exception:
        return None


def candlestick_confirm(
    df: pd.DataFrame,
    direction: str,
    window: int = 3,
    allowed_patterns: Optional[List[str]] = None,
) -> Tuple[bool, Dict[str, object]]:
    """
    Confirm breakout direction via TA-Lib candlestick patterns.

    Args:
      df: DataFrame with columns ['open','high','low','close'] indexed by time; should be closed bars.
      direction: 'bullish' or 'bearish' (from breakout logic)
      window: number of most-recent closed bars to evaluate (>=1)
      allowed_patterns: list of friendly names (keys of _DEFAULT_PATTERNS). If None, uses defaults.

    Returns:
      (ok, info) where ok indicates confirmation passed; info contains:
        {
          'score': float in [0,1],
          'dir': 'bullish'|'bearish'|None,
          'top': List[(name:str, value:int)],  # non-zero signals near the end
        }

    Fails open (returns ok=True) if TA-Lib unavailable to avoid blocking live trading.
    """
    info: Dict[str, object] = {"score": 0.0, "dir": None, "top": []}

    ta = _try_import_talib()
    if ta is None:
        # Fail-open in live: do not block trades if TA-Lib not importable
        return True, info

    if df is None or len(df) == 0:
        return False, info

    # Pass FULL DataFrame to TA-Lib for context (patterns need prior bars to validate trends)
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    patterns = allowed_patterns if allowed_patterns else list(_DEFAULT_PATTERNS.keys())

    # Clamp window to result array size (how many recent bars to check for patterns)
    w = max(1, min(int(window), len(df)))

    # Collect last non-zero signals across patterns for the evaluation window
    signals: List[Tuple[str, int]] = []
    for pname in patterns:
        fn_name = _DEFAULT_PATTERNS.get(pname)
        if not fn_name:
            continue
        fn = getattr(ta, fn_name, None)
        if fn is None:
            continue
        try:
            # TA-Lib gets full context to detect trends and validate pattern requirements
            arr = fn(o, h, l, c)
        except Exception:
            continue
        if arr is None or len(arr) == 0:
            continue
        # Look back within window for the most recent non-zero value
        # Only check last `window` bars for actual pattern occurrence
        val = 0
        for v in reversed(arr[-w:]):
            if int(v) != 0:
                val = int(v)
                break
        if val != 0:
            signals.append((pname, val))

    if not signals:
        # No explicit pattern hit; treat as soft fail (no confirmation)
        return False, info

    # Compute directional score: prefer matches to required direction
    # TA-Lib patterns typically return +100 or -100 (sometimes magnitude varies)
    best_name = None
    best_val = 0
    req_sign = 1 if direction == "bullish" else -1
    for name, v in signals:
        # favor signals that agree in sign; penalize opposite
        if v * req_sign > 0 and abs(v) >= abs(best_val):
            best_name, best_val = name, v

    # If nothing matches the required direction, take the strongest overall
    if best_name is None:
        best_name, best_val = max(signals, key=lambda t: abs(t[1]))

    # Normalize to [0,1] using 100 as a baseline unit (most patterns)
    magnitude = min(1.0, abs(best_val) / 100.0)
    dir_out = "bullish" if best_val > 0 else "bearish"

    info["score"] = float(magnitude)
    info["dir"] = dir_out
    # Sort top signals by magnitude desc
    top_sorted = [(n, int(v)) for n, v in sorted(signals, key=lambda t: abs(t[1]), reverse=True)]
    info["top"] = top_sorted
    info["primary"] = top_sorted[0][0] if top_sorted else None

    # Confirm if direction matches and score > 0
    ok = (dir_out == direction) and (magnitude > 0)
    return ok, info
