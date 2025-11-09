"""
AMD (Accumulation, Manipulation, Distribution) Context Analyzer.

Analyzes market structure to identify AMD phases and provide context
for filtering breakout trades.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Literal
import pandas as pd
import MetaTrader5 as mt5

from sessions import get_current_session, is_near_session_open, SessionType
from utils import resolve_pip_size, get_pip_size

logger = logging.getLogger(__name__)

PhaseType = Literal["ACCUMULATION", "MANIPULATION", "EXPANSION", "DISTRIBUTION", "UNKNOWN"]
AsiaType = Literal["ACCUMULATION", "EXPANSION", "UNKNOWN"]
DirectionType = Optional[Literal["BUY", "SELL"]]


class AMDAnalyzer:
    """
    Analyzes candle data to determine AMD phase and trading context.
    """

    def __init__(self, config):
        """
        Initialize AMD analyzer with config.

        Args:
            config: Bot config object with amd_settings
        """
        self.config = config

        # AMD feature flags
        self.enabled = getattr(config, 'amd_enabled', False)
        self.enable_asian_trading = getattr(config, 'amd_enable_asian_trading', False)
        self.enable_london_trading = getattr(config, 'amd_enable_london_trading', True)
        self.enable_ny_trading = getattr(config, 'amd_enable_ny_trading', True)

        # Session hours (UTC)
        self.asian_session_hours = getattr(config, 'amd_asian_session_hours', (0, 8))
        self.london_session_hours = getattr(config, 'amd_london_session_hours', (8, 16))
        self.ny_session_hours = getattr(config, 'amd_ny_session_hours', (13, 22))

        # Accumulation detection thresholds
        self.max_range_vs_adr_ratio = getattr(config, 'amd_max_range_vs_adr_ratio', 0.4)
        self.min_overlap_bars = getattr(config, 'amd_min_overlap_bars', 5)
        self.max_price_from_open_pips = getattr(config, 'amd_max_price_from_open_pips', 30)

        # Manipulation detection
        self.manipulation_enabled = getattr(config, 'amd_manipulation_enabled', True)
        self.sweep_threshold_pips = getattr(config, 'amd_sweep_threshold_pips', 5)
        self.reversal_confirmation_bars = getattr(config, 'amd_reversal_confirmation_bars', 3)
        self.reversal_threshold_pips = getattr(config, 'amd_reversal_threshold_pips', 15)

        # Directional bias
        self.directional_bias_enabled = getattr(config, 'amd_directional_bias_enabled', True)
        self.allow_countertrend_in_expansion = getattr(config, 'amd_allow_countertrend_in_expansion', False)
        self.disable_trades_in_distribution = getattr(config, 'amd_disable_trades_in_distribution', False)

    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Analyze candle data and return AMD context.

        Args:
            data: Candle DataFrame with UTC-aware DatetimeIndex
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., "M15")

        Returns:
            AMD context dict or None if analysis fails
        """
        try:
            if not self.enabled:
                return None

            if data is None or len(data) == 0:
                return None

            # Get current session
            now = datetime.now(timezone.utc)
            session = get_current_session(
                now,
                self.asian_session_hours,
                self.london_session_hours,
                self.ny_session_hours
            )

            # Initialize context
            context: Dict[str, Any] = {
                "session": session,
                "asia_type": "UNKNOWN",
                "asia_high": None,
                "asia_low": None,
                "asia_range_pips": None,
                "daily_open": None,
                "phase": "UNKNOWN",
                "has_swept_asia_high": False,
                "has_swept_asia_low": False,
                "sweep_confirmed": False,
                "is_first_sweep_high": False,
                "is_first_sweep_low": False,
                "sweep_high_at_london_open": False,
                "sweep_high_at_ny_open": False,
                "sweep_low_at_london_open": False,
                "sweep_low_at_ny_open": False,
                "is_late_session": False,
                "sweep_high_count": 0,
                "sweep_low_count": 0,
                "manipulation_type": None,  # "spring", "upthrust", or None
                "in_manipulation_window": False,
                "volatility_factor": 1.0,  # Multiplier for thresholds based on current vs average volatility
                "allowed_direction": None,
                "reason": "no_analysis"
            }

            # Check if we're in late session (late NY = 19:00+ UTC)
            if session == "NY" and now.hour >= 19:
                context["is_late_session"] = True

            # Check if we're in manipulation window (30 min after London/NY open)
            # This is when manipulation (liquidity grabs) typically occur
            manipulation_window_minutes = getattr(self.config, 'amd_manipulation_window_minutes', 30)

            if session == "LONDON":
                london_start_hour = self.london_session_hours[0]
                session_start = now.replace(hour=london_start_hour, minute=0, second=0, microsecond=0)
                minutes_since_open = (now - session_start).total_seconds() / 60
                if 0 <= minutes_since_open <= manipulation_window_minutes:
                    context["in_manipulation_window"] = True
                    logger.debug(f"In London manipulation window ({minutes_since_open:.0f} min since open)")

            elif session == "NY":
                ny_start_hour = self.ny_session_hours[0]
                session_start = now.replace(hour=ny_start_hour, minute=0, second=0, microsecond=0)
                minutes_since_open = (now - session_start).total_seconds() / 60
                if 0 <= minutes_since_open <= manipulation_window_minutes:
                    context["in_manipulation_window"] = True
                    logger.debug(f"In NY manipulation window ({minutes_since_open:.0f} min since open)")

            # Get symbol info for pip calculation
            sym_info = mt5.symbol_info(symbol)
            if not sym_info:
                logger.warning(f"{symbol}: No symbol info for AMD analysis")
                return context

            pip = resolve_pip_size(symbol, sym_info, self.config)
            if pip <= 0:
                logger.warning(f"{symbol}: Invalid pip size for AMD analysis")
                return context

            # Extract Asia session data
            asia_data = self._get_asia_session_data(data, now)
            if asia_data is not None and len(asia_data) > 0:
                context["asia_high"] = float(asia_data['high'].max())
                context["asia_low"] = float(asia_data['low'].min())
                context["asia_range_pips"] = (context["asia_high"] - context["asia_low"]) / pip

                # Daily open = first Asia candle open
                context["daily_open"] = float(asia_data.iloc[0]['open'])

                # Classify Asia session type
                context["asia_type"] = self._classify_asia_session(asia_data, pip, data)

                # Calculate volatility factor for dynamic threshold scaling
                # Compare current Asia range to recent average
                recent_avg_range = self._calculate_recent_asia_avg_range(data, pip, lookback_days=20)
                if recent_avg_range is not None and recent_avg_range > 0:
                    current_range = context["asia_range_pips"]
                    # Volatility factor = current / average (clamped to 0.5 - 2.0 range)
                    volatility_factor = max(0.5, min(2.0, current_range / recent_avg_range))
                    context["volatility_factor"] = volatility_factor
                    logger.debug(f"Volatility factor: {volatility_factor:.2f} "
                               f"(current={current_range:.1f}p avg={recent_avg_range:.1f}p)")

            # Detect manipulation (sweeps of Asia range)
            if session in ("LONDON", "NY") and context["asia_high"] is not None:
                self._detect_manipulation(data, context, pip, now)

            # Classify current phase
            context["phase"] = self._classify_phase(context, session)

            # Determine allowed direction
            if self.directional_bias_enabled:
                context["allowed_direction"] = self._determine_direction(context)

            # Generate reason string
            context["reason"] = self._generate_reason(context)

            logger.debug(f"{symbol} AMD: session={session} phase={context['phase']} "
                        f"asia_type={context['asia_type']} dir={context['allowed_direction']}")

            return context

        except Exception as e:
            logger.error(f"AMD analysis error for {symbol}: {e}", exc_info=True)
            return None

    def _get_asia_session_data(self, data: pd.DataFrame, now: datetime) -> Optional[pd.DataFrame]:
        """
        Extract Asia session data (00:00-08:00 UTC) for the most recent complete day in the dataset.

        Uses the last candle's date to determine which day's Asia session to extract.
        This allows the function to work in backtesting and with historical data.

        Args:
            data: Full candle DataFrame with UTC-aware DatetimeIndex
            now: Current UTC time (used as fallback if data is empty)

        Returns:
            DataFrame containing only Asia session candles, or None
        """
        try:
            if data is None or len(data) == 0:
                return None

            # Use last candle's date, not current date (for backtest compatibility)
            last_time = data.index[-1]

            # Ensure timezone-aware
            if last_time.tzinfo is None:
                logger.warning("Data index is not timezone-aware; assuming UTC")
                last_time = last_time.replace(tzinfo=timezone.utc)
            else:
                last_time = last_time.astimezone(timezone.utc)

            day = last_time.date()
            asia_start = datetime(day.year, day.month, day.day,
                                 self.asian_session_hours[0], 0, 0, tzinfo=timezone.utc)
            asia_end = datetime(day.year, day.month, day.day,
                               self.asian_session_hours[1], 0, 0, tzinfo=timezone.utc)

            # Filter candles within Asia session
            asia_candles = data[(data.index >= asia_start) & (data.index < asia_end)]

            if len(asia_candles) == 0:
                return None

            return asia_candles

        except Exception as e:
            logger.error(f"Error extracting Asia session data: {e}")
            return None

    def _calculate_recent_asia_avg_range(self, data: pd.DataFrame, pip: float, lookback_days: int = 20) -> Optional[float]:
        """
        Calculate average Asia session range over recent days.

        Args:
            data: Full candle DataFrame
            pip: Pip size
            lookback_days: Number of days to average

        Returns:
            Average Asia range in pips, or None if insufficient data
        """
        try:
            if data is None or len(data) == 0:
                return None

            # Get unique dates in the dataset
            dates = data.index.normalize().unique()

            if len(dates) < 2:
                return None

            asia_ranges = []

            # Calculate Asia range for each day
            for i in range(min(lookback_days, len(dates))):
                if i >= len(dates):
                    break

                day = dates[-(i+1)].to_pydatetime()
                asia_start = day.replace(hour=self.asian_session_hours[0], minute=0, second=0, microsecond=0)
                asia_end = day.replace(hour=self.asian_session_hours[1], minute=0, second=0, microsecond=0)

                day_asia = data[(data.index >= asia_start) & (data.index < asia_end)]

                if len(day_asia) > 0:
                    day_range = float(day_asia['high'].max() - day_asia['low'].min())
                    asia_ranges.append(day_range / pip)

            if len(asia_ranges) < 3:
                return None

            return float(sum(asia_ranges) / len(asia_ranges))

        except Exception as e:
            logger.error(f"Error calculating recent Asia avg range: {e}")
            return None

    def _classify_asia_session(self, asia_data: pd.DataFrame, pip: float, full_data: pd.DataFrame) -> AsiaType:
        """
        Classify Asia session as ACCUMULATION or EXPANSION.

        ACCUMULATION criteria:
        - Narrow range (< max_range_vs_adr_ratio * recent avg Asia range)
        - Multiple overlapping bars
        - Price stays near daily open

        Args:
            asia_data: Asia session candles
            pip: Pip size
            full_data: Full dataset for ADR calculation

        Returns:
            AsiaType: "ACCUMULATION" or "EXPANSION"
        """
        try:
            if len(asia_data) < 3:
                return "UNKNOWN"

            asia_high = float(asia_data['high'].max())
            asia_low = float(asia_data['low'].min())
            asia_range = asia_high - asia_low
            asia_range_pips = asia_range / pip

            daily_open = float(asia_data.iloc[0]['open'])

            # Calculate recent average Asia range
            recent_avg = self._calculate_recent_asia_avg_range(full_data, pip)

            # Criterion 1: Range vs historical average
            narrow_range = False
            if recent_avg is not None and recent_avg > 0:
                if asia_range_pips < (self.max_range_vs_adr_ratio * recent_avg):
                    narrow_range = True
                logger.debug(f"Asia range: {asia_range_pips:.1f}p vs avg: {recent_avg:.1f}p "
                           f"(threshold: {self.max_range_vs_adr_ratio * recent_avg:.1f}p)")
            else:
                # Fallback: use absolute threshold for major FX pairs (~50 pips)
                if asia_range_pips < 50:
                    narrow_range = True

            # Criterion 2: Price stays near daily open
            mid_price = (asia_high + asia_low) / 2.0
            distance_from_open_pips = abs(mid_price - daily_open) / pip
            near_open = distance_from_open_pips <= self.max_price_from_open_pips

            # Criterion 3: Overlapping bars (bodies overlap)
            overlapping_count = 0
            for i in range(1, len(asia_data)):
                prev_body_high = max(asia_data.iloc[i-1]['open'], asia_data.iloc[i-1]['close'])
                prev_body_low = min(asia_data.iloc[i-1]['open'], asia_data.iloc[i-1]['close'])
                curr_body_high = max(asia_data.iloc[i]['open'], asia_data.iloc[i]['close'])
                curr_body_low = min(asia_data.iloc[i]['open'], asia_data.iloc[i]['close'])

                # Check if bodies overlap
                if (curr_body_low <= prev_body_high) and (curr_body_high >= prev_body_low):
                    overlapping_count += 1

            has_overlap = overlapping_count >= self.min_overlap_bars

            logger.debug(f"Asia classification: narrow={narrow_range} near_open={near_open} "
                        f"overlap={overlapping_count}/{self.min_overlap_bars}")

            # Accumulation requires at least 2 of 3 criteria
            criteria_met = sum([narrow_range, near_open, has_overlap])

            if criteria_met >= 2:
                return "ACCUMULATION"
            else:
                return "EXPANSION"

        except Exception as e:
            logger.error(f"Error classifying Asia session: {e}")
            return "UNKNOWN"

    def _detect_manipulation(self, data: pd.DataFrame, context: Dict[str, Any],
                            pip: float, now: datetime) -> None:
        """
        Detect if Asia high/low has been swept (manipulation) with reversal confirmation.

        A sweep is confirmed when:
        1. Price breaks beyond Asia H/L by sweep_threshold_pips
        2. Price returns back inside the Asia range within reversal_confirmation_bars
        3. Return move is at least reversal_threshold_pips

        Updates context in-place with sweep detection results.

        Args:
            data: Full candle DataFrame
            context: AMD context dict to update
            pip: Pip size
            now: Current UTC time
        """
        try:
            if not self.manipulation_enabled:
                return

            asia_high = context.get("asia_high")
            asia_low = context.get("asia_low")

            if asia_high is None or asia_low is None:
                return

            # Use last candle's date for backtest compatibility
            last_time = data.index[-1].astimezone(timezone.utc)
            day = last_time.date()
            asia_end = datetime(day.year, day.month, day.day,
                               self.asian_session_hours[1], 0, 0, tzinfo=timezone.utc)

            post_asia = data[data.index >= asia_end]

            if len(post_asia) == 0:
                return

            sweep_threshold = self.sweep_threshold_pips * pip
            reversal_threshold = self.reversal_threshold_pips * pip

            # Detect sweep of Asia high (bearish sweep)
            swept_high = False
            high_confirmed = False
            sweep_high_count = 0

            if post_asia['high'].max() > (asia_high + sweep_threshold):
                # Count number of sweep attempts
                for idx, row in post_asia.iterrows():
                    if row['high'] > (asia_high + sweep_threshold):
                        sweep_high_count += 1
                swept_high = True
                context["has_swept_asia_high"] = True

                # Find the bar that swept the high
                sweep_idx = post_asia['high'].idxmax()
                sweep_bar_loc = post_asia.index.get_loc(sweep_idx)
                sweep_time = sweep_idx.to_pydatetime()

                # Check if this is the first sweep (no prior sweep above Asia high)
                prior_bars = post_asia[post_asia.index < sweep_idx]
                is_first_sweep = not any(prior_bars['high'] > (asia_high + sweep_threshold))
                context["is_first_sweep_high"] = is_first_sweep

                # Check if sweep occurred near London or NY open (within 30 minutes)
                context["sweep_high_at_london_open"] = is_near_session_open(
                    "LONDON", sweep_time, window_minutes=30,
                    asia_hours=self.asian_session_hours,
                    london_hours=self.london_session_hours,
                    ny_hours=self.ny_session_hours
                )
                context["sweep_high_at_ny_open"] = is_near_session_open(
                    "NY", sweep_time, window_minutes=30,
                    asia_hours=self.asian_session_hours,
                    london_hours=self.london_session_hours,
                    ny_hours=self.ny_session_hours
                )

                # Check for reversal confirmation

                # Look at next N bars for reversal
                bars_after_sweep = post_asia.iloc[sweep_bar_loc:sweep_bar_loc + self.reversal_confirmation_bars + 1]

                if len(bars_after_sweep) > 1:
                    sweep_high_price = float(bars_after_sweep.iloc[0]['high'])
                    subsequent_lows = bars_after_sweep.iloc[1:]['low']

                    if len(subsequent_lows) > 0:
                        lowest_after = float(subsequent_lows.min())

                        # Reversal confirmed if:
                        # 1. Price returned inside Asia range, AND
                        # 2. Moved at least reversal_threshold_pips down from sweep high
                        if (lowest_after < asia_high) and ((sweep_high_price - lowest_after) >= reversal_threshold):
                            high_confirmed = True
                            logger.debug(f"Asia high sweep confirmed: swept to {sweep_high_price:.5f}, "
                                       f"reversed to {lowest_after:.5f} ({(sweep_high_price - lowest_after)/pip:.1f}p)")

            # Detect sweep of Asia low (bullish sweep)
            swept_low = False
            low_confirmed = False
            sweep_low_count = 0

            if post_asia['low'].min() < (asia_low - sweep_threshold):
                # Count number of sweep attempts
                for idx, row in post_asia.iterrows():
                    if row['low'] < (asia_low - sweep_threshold):
                        sweep_low_count += 1
                swept_low = True
                context["has_swept_asia_low"] = True

                # Find the bar that swept the low
                sweep_idx = post_asia['low'].idxmin()
                sweep_bar_loc = post_asia.index.get_loc(sweep_idx)
                sweep_time = sweep_idx.to_pydatetime()

                # Check if this is the first sweep (no prior sweep below Asia low)
                prior_bars = post_asia[post_asia.index < sweep_idx]
                is_first_sweep = not any(prior_bars['low'] < (asia_low - sweep_threshold))
                context["is_first_sweep_low"] = is_first_sweep

                # Check if sweep occurred near London or NY open (within 30 minutes)
                context["sweep_low_at_london_open"] = is_near_session_open(
                    "LONDON", sweep_time, window_minutes=30,
                    asia_hours=self.asian_session_hours,
                    london_hours=self.london_session_hours,
                    ny_hours=self.ny_session_hours
                )
                context["sweep_low_at_ny_open"] = is_near_session_open(
                    "NY", sweep_time, window_minutes=30,
                    asia_hours=self.asian_session_hours,
                    london_hours=self.london_session_hours,
                    ny_hours=self.ny_session_hours
                )

                # Check for reversal confirmation

                bars_after_sweep = post_asia.iloc[sweep_bar_loc:sweep_bar_loc + self.reversal_confirmation_bars + 1]

                if len(bars_after_sweep) > 1:
                    sweep_low_price = float(bars_after_sweep.iloc[0]['low'])
                    subsequent_highs = bars_after_sweep.iloc[1:]['high']

                    if len(subsequent_highs) > 0:
                        highest_after = float(subsequent_highs.max())

                        # Reversal confirmed if:
                        # 1. Price returned inside Asia range, AND
                        # 2. Moved at least reversal_threshold_pips up from sweep low
                        if (highest_after > asia_low) and ((highest_after - sweep_low_price) >= reversal_threshold):
                            low_confirmed = True
                            logger.debug(f"Asia low sweep confirmed: swept to {sweep_low_price:.5f}, "
                                       f"reversed to {highest_after:.5f} ({(highest_after - sweep_low_price)/pip:.1f}p)")

            # Only mark sweep_confirmed if at least one sweep + reversal occurred
            if high_confirmed or low_confirmed:
                context["sweep_confirmed"] = True
            else:
                context["sweep_confirmed"] = False

            # Store which specific sweeps were confirmed
            context["sweep_high_confirmed"] = high_confirmed
            context["sweep_low_confirmed"] = low_confirmed

            # Store sweep counts
            context["sweep_high_count"] = sweep_high_count
            context["sweep_low_count"] = sweep_low_count

            # Label manipulation type using Wyckoff terminology
            # Spring = bullish manipulation (fake breakdown below support that fails)
            # Upthrust/UTAD = bearish manipulation (fake breakout above resistance that fails)
            if low_confirmed and not high_confirmed:
                context["manipulation_type"] = "spring"  # Bullish reversal expected
                logger.debug(f"Wyckoff: SPRING detected (bullish manipulation)")
            elif high_confirmed and not low_confirmed:
                context["manipulation_type"] = "upthrust"  # Bearish reversal expected
                logger.debug(f"Wyckoff: UPTHRUST detected (bearish manipulation)")
            elif high_confirmed and low_confirmed:
                context["manipulation_type"] = "whipsaw"  # Both sides swept = distribution
                logger.debug(f"Wyckoff: WHIPSAW detected (distribution behavior)")
            else:
                context["manipulation_type"] = None

        except Exception as e:
            logger.error(f"Error detecting manipulation: {e}")

    def _classify_phase(self, context: Dict[str, Any], session: SessionType) -> PhaseType:
        """
        Classify current AMD phase based on session and structure.

        Conservative approach:
        - Only label phases we have clear evidence for
        - Default to UNKNOWN when uncertain
        - Avoid false positives (e.g., labeling NY as DISTRIBUTION without evidence)

        Args:
            context: AMD context dict
            session: Current session

        Returns:
            PhaseType: Current AMD phase
        """
        try:
            asia_type = context.get("asia_type", "UNKNOWN")
            has_swept = context.get("has_swept_asia_high") or context.get("has_swept_asia_low")
            sweep_confirmed = context.get("sweep_confirmed", False)

            # During Asia
            if session == "ASIA":
                if asia_type == "ACCUMULATION":
                    return "ACCUMULATION"
                elif asia_type == "EXPANSION":
                    return "EXPANSION"
                else:
                    return "UNKNOWN"

            # During London
            elif session == "LONDON":
                if sweep_confirmed:
                    # Confirmed manipulation → expansion phase
                    return "EXPANSION"
                elif has_swept and not sweep_confirmed:
                    # Sweep happened but no reversal yet → manipulation in progress
                    return "MANIPULATION"
                elif asia_type == "ACCUMULATION":
                    # Asia accumulated, but no sweep detected yet
                    # Use ACCUMULATION phase (waiting for manipulation to occur)
                    # This allows slow breakouts during London if no manipulation happens
                    return "ACCUMULATION"
                else:
                    # Asia was expansion or unknown - can't determine London phase
                    return "UNKNOWN"

            # During NY
            elif session == "NY":
                # Check for distribution signals
                # 1. Both sweeps confirmed = whipsaw (distribution behavior)
                sweep_high_confirmed = context.get("sweep_high_confirmed", False)
                sweep_low_confirmed = context.get("sweep_low_confirmed", False)

                if sweep_high_confirmed and sweep_low_confirmed:
                    # Both Asia H/L swept and reversed = classic whipsaw/distribution
                    return "DISTRIBUTION"

                # 2. Late NY session (after 19:00 UTC) with confirmed sweep = potential distribution
                # Late NY is when institutions often close/distribute positions
                now_utc = datetime.now(timezone.utc)
                is_late_ny = now_utc.hour >= 19 and session == "NY"

                if is_late_ny and sweep_confirmed:
                    # Late NY after sweep = likely distribution phase
                    return "DISTRIBUTION"

                # 3. Otherwise, if sweep confirmed in NY, treat as expansion
                if sweep_confirmed:
                    return "EXPANSION"
                else:
                    # No confirmed sweep → can't determine phase
                    return "UNKNOWN"

            else:
                return "UNKNOWN"

        except Exception as e:
            logger.error(f"Error classifying phase: {e}")
            return "UNKNOWN"

    def _determine_direction(self, context: Dict[str, Any]) -> DirectionType:
        """
        Determine allowed trade direction based on AMD context.

        Uses confirmed sweep information to set directional bias:
        - If Asia low swept + confirmed reversal → BUY bias
        - If Asia high swept + confirmed reversal → SELL bias

        Args:
            context: AMD context dict

        Returns:
            DirectionType: "BUY", "SELL", or None (both directions allowed)
        """
        try:
            phase = context.get("phase", "UNKNOWN")
            sweep_high_confirmed = context.get("sweep_high_confirmed", False)
            sweep_low_confirmed = context.get("sweep_low_confirmed", False)

            # During expansion after confirmed sweep, bias toward reversal direction
            if phase == "EXPANSION":
                # Use specific confirmed sweep flags (more precise)
                if sweep_low_confirmed and not sweep_high_confirmed:
                    return "BUY"  # Swept low + reversed up → bullish expansion
                elif sweep_high_confirmed and not sweep_low_confirmed:
                    return "SELL"  # Swept high + reversed down → bearish expansion
                elif sweep_low_confirmed and sweep_high_confirmed:
                    # Both sweeps confirmed = whipsaw/distribution behavior
                    # Block trades to avoid getting chopped
                    return "BLOCK"
                else:
                    # Expansion phase but no confirmed sweeps - allow both
                    return None

            # During distribution, optionally disable trades
            if phase == "DISTRIBUTION" and self.disable_trades_in_distribution:
                # This would block all trades during distribution
                # Note: Currently we rarely label DISTRIBUTION, so this is mostly unused
                return None

            # Default: allow both directions
            return None

        except Exception as e:
            logger.error(f"Error determining direction: {e}")
            return None

    def _generate_reason(self, context: Dict[str, Any]) -> str:
        """
        Generate human-readable reason for AMD state.

        Args:
            context: AMD context dict

        Returns:
            str: Reason string
        """
        session = context.get("session", "UNKNOWN")
        phase = context.get("phase", "UNKNOWN")
        asia_type = context.get("asia_type", "UNKNOWN")
        direction = context.get("allowed_direction")

        parts = [f"{session.lower()}_session", f"{phase.lower()}_phase"]

        if asia_type != "UNKNOWN":
            parts.append(f"asia_{asia_type.lower()}")

        if direction:
            parts.append(f"bias_{direction.lower()}")

        return "_".join(parts)
