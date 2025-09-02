import traceback
from talib import (
    CDLENGULFING, CDLDOJI, CDLSHOOTINGSTAR, CDLHAMMER,
    CDLMORNINGSTAR, CDLPIERCING, CDLHARAMI, CDLHARAMICROSS,
    CDL3WHITESOLDIERS, CDLMORNINGDOJISTAR,
    CDLEVENINGSTAR, CDLDARKCLOUDCOVER, CDL3BLACKCROWS, CDLEVENINGDOJISTAR,
    MFI
)
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IndicatorCalculator:
    def __init__(self, config=None):
        self.config = config or {}

        # Load MFI parameters
        mfi_params = self.config.get('mfi', {})
        self.mfi_period = mfi_params.get('period', 14)

        # Load Price Action parameters
        pa_settings = self.config.get('price_action_settings', {})
        self.proximity_atr_multiplier = pa_settings.get('proximity_atr_multiplier', 1.5)
        self.divergence_lookback = pa_settings.get('divergence_lookback', 20)

        # Thresholds for MFI momentum confirmation
        self.mfi_oversold_threshold = pa_settings.get('mfi_oversold_threshold', 40)
        self.mfi_overbought_threshold = pa_settings.get('mfi_overbought_threshold', 60)

    def calculate_candlestick_patterns(self, data):
        try:
            if data is None or data.empty:
                logging.error("No data provided for candlestick pattern calculation")
                return None
            df = data.copy()

            if 'tick_volume' in df.columns:
                df['mfi'] = MFI(
                    df['high'], df['low'], df['close'],
                    df['tick_volume'].astype(float),
                    timeperiod=self.mfi_period
                )
            else:
                logging.warning("Tick volume not found in data, MFI will not be calculated")

            # Calculate patterns
            df['cdl_engulfing'] = CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            df['cdl_doji'] = CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['cdl_shooting_star'] = CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_hammer'] = CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['cdl_morning_star'] = CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_piercing'] = CDLPIERCING(df['open'], df['high'], df['low'], df['close'])
            df['cdl_harami'] = CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
            df['cdl_harami_cross'] = CDLHARAMICROSS(df['open'], df['high'], df['low'], df['close'])
            df['cdl_3_white_soldiers'] = CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
            df['cdl_morning_doji_star'] = CDLMORNINGDOJISTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_evening_star'] = CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_dark_cloud'] = CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'])
            df['cdl_3_black_crows'] = CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
            df['cdl_evening_doji_star'] = CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['close'])

            return df
        except Exception as e:
            logging.error(f"Error in calculate_candlestick_patterns: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def _check_mfi_divergence(self, data, signal_index, direction):
        """
        Check for MFI divergence relative to price over a lookback window.
        - Bullish divergence: price makes a lower low, MFI makes a higher low.
        - Bearish divergence: price makes a higher high, MFI makes a lower high.
        """
        if signal_index < self.divergence_lookback:
            return False

        lookback_data = data.iloc[signal_index - self.divergence_lookback: signal_index]
        signal_candle = data.iloc[signal_index]

        if 'mfi' not in signal_candle or pd.isna(signal_candle['mfi']):
            return False

        if direction == 'buy':
            prev_low_idx = lookback_data['low'].idxmin()
            prev_low_price = lookback_data.at[prev_low_idx, 'low']
            prev_low_mfi = lookback_data.at[prev_low_idx, 'mfi']

            if signal_candle['low'] < prev_low_price and signal_candle['mfi'] > prev_low_mfi:
                logging.info(
                    f"Bullish MFI Divergence: Price LL {signal_candle['low']:.5f} < {prev_low_price:.5f}, "
                    f"MFI HL {signal_candle['mfi']:.2f} > {prev_low_mfi:.2f}"
                )
                return True

        elif direction == 'sell':
            prev_high_idx = lookback_data['high'].idxmax()
            prev_high_price = lookback_data.at[prev_high_idx, 'high']
            prev_high_mfi = lookback_data.at[prev_high_idx, 'mfi']

            if signal_candle['high'] > prev_high_price and signal_candle['mfi'] < prev_high_mfi:
                logging.info(
                    f"Bearish MFI Divergence: Price HH {signal_candle['high']:.5f} > {prev_high_price:.5f}, "
                    f"MFI LH {signal_candle['mfi']:.2f} < {prev_high_mfi:.2f}"
                )
                return True

        return False

    def check_entry_patterns(self, data_with_indicators, direction, pullback_level=None, atr=None):
        if data_with_indicators is None or len(data_with_indicators) < self.divergence_lookback + 2:
            return False

        last_closed = data_with_indicators.iloc[-2]
        signal_index = len(data_with_indicators) - 2

        # --- Filter 1: Proximity to pullback level ---
        if pullback_level is not None and atr is not None and atr > 0:
            proximity_zone = atr * self.proximity_atr_multiplier
            price_to_check = last_closed['low'] if direction == 'buy' else last_closed['high']
            if abs(price_to_check - pullback_level) > proximity_zone:
                return False

        # --- Filter 2: Valid candlestick pattern ---
        if direction == 'buy':
            is_pattern_valid = (
                last_closed['cdl_engulfing'] == 100 or
                last_closed['cdl_hammer'] == 100 or
                last_closed['cdl_morning_star'] == 100 or
                last_closed['cdl_piercing'] == 100 or
                last_closed['cdl_harami'] == 100 or
                last_closed['cdl_3_white_soldiers'] == 100
            )
        else:  # sell
            is_pattern_valid = (
                last_closed['cdl_engulfing'] == -100 or
                last_closed['cdl_shooting_star'] == -100 or
                last_closed['cdl_evening_star'] == -100 or
                last_closed['cdl_dark_cloud'] == -100 or
                last_closed['cdl_harami'] == -100 or
                last_closed['cdl_3_black_crows'] == -100
            )

        if not is_pattern_valid:
            return False

        # --- Filter 3: Divergence OR MFI Momentum ---
        if self._check_mfi_divergence(data_with_indicators, signal_index, direction):
            logging.info("Entry confirmed by MFI Divergence.")
            return True

        if 'mfi' in last_closed and not pd.isna(last_closed['mfi']):
            if direction == 'buy' and last_closed['mfi'] < self.mfi_oversold_threshold:
                logging.info(
                    f"Entry confirmed by MFI oversold: {last_closed['mfi']:.2f} < {self.mfi_oversold_threshold}"
                )
                return True
            elif direction == 'sell' and last_closed['mfi'] > self.mfi_overbought_threshold:
                logging.info(
                    f"Entry confirmed by MFI overbought: {last_closed['mfi']:.2f} > {self.mfi_overbought_threshold}"
                )
                return True

        return False

    def identify_support_resistance_levels(self, swing_points):
        if swing_points:
            highs = swing_points['highs']['high'].unique()
            lows = swing_points['lows']['low'].unique()
            levels = np.concatenate([highs, lows])
            return np.unique(levels)
        return np.array([])
