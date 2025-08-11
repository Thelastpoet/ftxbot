import pandas as pd
import pytz
from datetime import datetime, time, date
import logging
from dataclasses import dataclass
from typing import Dict

# logger for the engine.
engine_logger = logging.getLogger(__name__)

if not engine_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [ContextEngine] - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    engine_logger.addHandler(handler)
    engine_logger.setLevel(logging.INFO)
    engine_logger.propagate = False

@dataclass
class MarketNarrative:
    """
    A structured object encapsulating the engine's complete hypothesis on the
    market's current state and intent. This is the primary output of the ContextEngine.
    """
    session_profile: str = "AWAITING_CONSOLIDATION"
    daily_bias: str = "Neutral"
    consolidation_range_high: float = 0.0
    consolidation_range_low: float = 0.0
    liquidity_target: float = 0.0
    invalidation_level: float = 0.0
    narrative_summary: str = "Engine initialized. Awaiting market data."
    clarity_score: str = "Low"

class ContextEngine:
    """
    Analyzes intraday price action to build a session-based market hypothesis.
    It operates as a state machine, modeling the AMD/PO3 cycle.
    """
    def __init__(self, strategy_config: Dict):
        """
        Initializes the ContextEngine.

        Args:
            strategy_config (Dict): A dictionary containing strategy parameters from the config file.
        """
        self.strategy_config = strategy_config
        self.narrative = MarketNarrative()

        self.expansion_threshold = strategy_config['asian_range_max_atr_ratio']
        self.ltf_swing_length = strategy_config['ltf_swing_length']
        self.zones_to_check = strategy_config['zones_to_check']

        self.SESSION_WINDOWS = {
            'asia': {
                'start': time.fromisoformat(strategy_config['session_times_utc']['asia']['start']),
                'end': time.fromisoformat(strategy_config['session_times_utc']['asia']['end'])
            },
            'london': {
                'start': time.fromisoformat(strategy_config['session_times_utc']['london']['start']),
                'end': time.fromisoformat(strategy_config['session_times_utc']['london']['end'])
            },
            'ny': {
                'start': time.fromisoformat(strategy_config['session_times_utc']['ny']['start']),
                'end': time.fromisoformat(strategy_config['session_times_utc']['ny']['end'])
            }
        }

        self.asian_high = None
        self.asian_low = None
        
        self.current_day: date = date(1970, 1, 1) # Initialize to epoch
        engine_logger.info("ContextEngine initialized with configured parameters.")

    def _reset_for_new_day(self):
        """Resets the engine's state for a new trading day."""
        self.narrative = MarketNarrative()
        self.asian_high = None
        self.asian_low = None
        engine_logger.info(f"New trading day detected ({self.current_day}). Engine state has been reset.")

    def process_data(self, ohlc_df: pd.DataFrame, indicators_df: pd.DataFrame) -> MarketNarrative:
        if ohlc_df.index.tz is None or indicators_df.index.tz is None:
            raise ValueError("Input DataFrames must have a timezone-aware (UTC) index.")

        latest_time = ohlc_df.index[-1]

        if latest_time.date() != self.current_day:
            self.current_day = latest_time.date()
            self._reset_for_new_day()

        profile = self.narrative.session_profile
        
        current_session = self._get_current_session(latest_time)
        logging.info(f"[Market Data] Time: {latest_time.strftime('%H:%M')} UTC ({(latest_time + pd.Timedelta(hours=3)).strftime('%H:%M')} EAT), Session: {current_session}, Profile: {profile}")
        
        if profile in ["AWAITING_CONSOLIDATION", "ANALYZING_CONSOLIDATION"]:
            self._handle_asian_session(ohlc_df, latest_time)
            if self.narrative.session_profile == "CONSOLIDATION_DEFINED":
                self._evaluate_asian_range(indicators_df)
            profile = self.narrative.session_profile

        if profile == "CONSOLIDATION_DEFINED":
            self._handle_london_session(ohlc_df, latest_time)
            profile = self.narrative.session_profile

        if "MANIPULATION_CONFIRMED" in profile:
            self._validate_hypothesis(ohlc_df)

        return self.narrative

    def _handle_asian_session(self, ohlc_df: pd.DataFrame, latest_time: datetime):
        asia_start = datetime.combine(self.current_day, self.SESSION_WINDOWS['asia']['start']).replace(tzinfo=pytz.UTC)
        asia_end = datetime.combine(self.current_day, self.SESSION_WINDOWS['asia']['end']).replace(tzinfo=pytz.UTC)

        session_df = ohlc_df[(ohlc_df.index >= asia_start) & (ohlc_df.index <= asia_end)]

        if session_df.empty:
            return

        self.asian_high = session_df['high'].max()
        self.asian_low = session_df['low'].min()
        self.narrative.session_profile = "ANALYZING_CONSOLIDATION"
        self.narrative.narrative_summary = "Analyzing Asian session consolidation..."
        
        if latest_time > asia_end:
            self.narrative.session_profile = "CONSOLIDATION_DEFINED"
            
    def _get_current_session(self, current_time: datetime) -> str:
        """
        Determines the precise current trading session, accounting for overlaps.
        """
        # We only need the time component for comparison
        t = current_time.time()

        # Get session start/end times from our config
        asia_start = self.SESSION_WINDOWS['asia']['start']
        asia_end = self.SESSION_WINDOWS['asia']['end']
        london_start = self.SESSION_WINDOWS['london']['start']
        london_end = self.SESSION_WINDOWS['london']['end']
        ny_start = self.SESSION_WINDOWS['ny']['start']
        ny_end = self.SESSION_WINDOWS['ny']['end']

        # Check for the specific overlap period first
        if t >= ny_start and t < london_end:
            return "London/NY Overlap"
        
        # Check for solo sessions
        if t >= london_start and t < london_end:
            return "London"
        
        if t >= ny_start and t < ny_end:
            return "New York"
            
        if t >= asia_start and t < asia_end:
            return "Asia"
        
        # Check for the gaps between sessions
        if t >= asia_end and t < london_start:
            return "Post-Asia / Pre-London"
            
        return "Out of Session"

    def _evaluate_asian_range(self, indicators_df: pd.DataFrame):
        if self.asian_high is None or self.asian_low is None:
            engine_logger.warning("Asian range evaluation called but no data was recorded.")
            return

        daily_atr = indicators_df['atr'].iloc[-1]
        
        if daily_atr <= 0:
            engine_logger.warning("Daily ATR is zero or invalid. Cannot evaluate Asian range profile.")
            self.narrative.session_profile = "CONSOLIDATION_DEFINED"
            self.narrative.clarity_score = "Low"
            self.narrative.narrative_summary = "Could not determine range profile due to invalid Daily ATR."
            return

        final_high = self.asian_high
        final_low = self.asian_low

        self.narrative.consolidation_range_high = final_high
        self.narrative.consolidation_range_low = final_low
        
        current_range_size = final_high - final_low
        range_to_atr_ratio = current_range_size / daily_atr

        engine_logger.info(
            f"Asian Range Evaluation Data: "
            f"Size={current_range_size:.5f}, "
            f"Daily ATR={daily_atr:.5f}, "
            f"Ratio={range_to_atr_ratio:.2%}"
        )

        if range_to_atr_ratio >= self.expansion_threshold:
            self.narrative.session_profile = "TREND_EXPANSION"
            self.narrative.clarity_score = "Low"
            self.narrative.narrative_summary = (
                f"Asian session was expansive (consumed {range_to_atr_ratio:.0%} of Daily ATR). "
                "Standard AMD template is void."
            )
        else:
            self.narrative.session_profile = "CONSOLIDATION_DEFINED"
            self.narrative.clarity_score = "High"
            self.narrative.narrative_summary = (
                f"Classic accumulation formed. Asian Range: {final_low:.5f} - {final_high:.5f}. "
                "Awaiting London manipulation."
            )
        
        engine_logger.info(f"Conclusion: {self.narrative.narrative_summary}")

    def _handle_london_session(self, ohlc_df: pd.DataFrame, latest_time: datetime):
        london_start = datetime.combine(self.current_day, self.SESSION_WINDOWS['london']['start']).replace(tzinfo=pytz.UTC)

        if latest_time < london_start:
            return

        london_df = ohlc_df[ohlc_df.index >= london_start]
        if london_df.empty:
            return

        swept_high = london_df['high'].max() > self.narrative.consolidation_range_high
        swept_low = london_df['low'].min() < self.narrative.consolidation_range_low

        if swept_low and not swept_high:
            self._formulate_hypothesis(direction="down", sweep_level=london_df['low'].min(), bias="Bullish", target=self.narrative.consolidation_range_high)
        elif swept_high and not swept_low:
            self._formulate_hypothesis(direction="up", sweep_level=london_df['high'].max(), bias="Bearish", target=self.narrative.consolidation_range_low)
        elif swept_high and swept_low:
            self.narrative.session_profile = "HYPOTHESIS_INVALIDATED"
            self.narrative.narrative_summary = "London session swept both sides. Market is choppy, no clear institutional intent."
            self.narrative.clarity_score = "Low"
            engine_logger.warning(self.narrative.narrative_summary)
        else:
            london_timeout = london_start + pd.Timedelta(hours=4)
            if latest_time > london_timeout:
                self.narrative.session_profile = "NO_MANIPULATION_DETECTED"
                self.narrative.narrative_summary = "London session timeout. No clear manipulation detected."
                self.narrative.clarity_score = "low"
            
    def _formulate_hypothesis(self, direction: str, sweep_level: float, bias: str, target: float):
        self.narrative.session_profile = f"MANIPULATION_CONFIRMED_{direction.upper()}"
        self.narrative.daily_bias = bias
        self.narrative.liquidity_target = target
        self.narrative.invalidation_level = sweep_level
        self.narrative.clarity_score = "High"
        self.narrative.narrative_summary = (
            f"London confirmed manipulation sweeping the Asian {('low' if direction == 'down' else 'high')}. "
            f"Bias is now {bias}. Target: {target:.5f}. Invalidation: {sweep_level:.5f}."
        )
        engine_logger.info(self.narrative.narrative_summary)

    def _validate_hypothesis(self, ohlc_df: pd.DataFrame):
        latest_price = ohlc_df['close'].iloc[-1]
        is_invalidated = False

        if self.narrative.daily_bias == "Bullish" and latest_price < self.narrative.invalidation_level:
            is_invalidated = True
        elif self.narrative.daily_bias == "Bearish" and latest_price > self.narrative.invalidation_level:
            is_invalidated = True
        
        if is_invalidated:
            original_bias = self.narrative.daily_bias
            self.narrative.session_profile = "HYPOTHESIS_INVALIDATED"
            self.narrative.daily_bias = "Neutral"
            self.narrative.clarity_score = "Medium"
            self.narrative.narrative_summary = (
                f"Original {original_bias} hypothesis FAILED. Price broke invalidation. "
                f"The move was likely a genuine trend. Bias has flipped to {self.narrative.daily_bias}."
            )
            engine_logger.warning(self.narrative.narrative_summary)