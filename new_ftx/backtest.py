#!/usr/bin/env python3
"""
Robust Backtest for the Forex Trading Bot
- Reuses live components where practical
- Avoids dependency on live ticks by simulating ticks from historical bars
- Disables M1 confirmation during backtest unless M1 data is provided
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional

import asyncio
import pandas as pd
import MetaTrader5 as mt5

from main import Config
from strategy import PurePriceActionStrategy
from market_data import MarketData
from mt5_client import MetaTrader5Client
from risk_manager import RiskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _MockAccountInfo:
    def __init__(self, balance: float):
        self.balance = balance


class _MockSymbolInfo:
    """Copy of MT5 symbol_info with essential fields for our logic."""
    def __init__(self, src):
        # Copy selected attributes if present, else fallback sensibly
        defaults = {
            'digits': 5,
            'point': 10 ** -5,
            'volume_min': 0.01,
            'volume_step': 0.01,
            'volume_max': 100.0,
            'trade_tick_size': 10 ** -5,
            'trade_tick_value': 1.0,
            'trade_stops_level': 0,
        }
        for k, v in defaults.items():
            setattr(self, k, getattr(src, k, v))


class _BacktestMT5Shim:
    """Shim that provides the subset of MT5 client APIs RiskManager needs."""

    def __init__(self, account_balance: float, symbol: str):
        self._account = _MockAccountInfo(balance=account_balance)
        self._symbol = symbol
        self._symbol_info = _MockSymbolInfo(symbol)

    def get_account_info(self):
        return self._account

    def get_symbol_info(self, symbol: str):
        # Single-symbol backtest; return consistent info
        return self._symbol_info

    # Unused by this backtest flow, but kept for compatibility
    def get_all_positions(self):
        return []


class Backtest:
    """Backtest that simulates ticks from historical candles and reuses strategy logic."""

    def __init__(self, config_path: str = 'config.json', starting_balance: float = 10000.0):
        # Make config path robust across CWDs
        from pathlib import Path as _Path
        cfg_path = _Path(config_path)
        if not cfg_path.exists():
            # try relative to script directory
            alt = _Path(__file__).parent / cfg_path.name
            if alt.exists():
                cfg_path = alt
        self.config = Config(str(cfg_path))
        self.mt5_client = MetaTrader5Client()
        self.market_data = MarketData(self.mt5_client, self.config)
        self.strategy = PurePriceActionStrategy(self.config)

        # Disable M1 confirmation for now; keep backtest mode for time filters
        self.strategy.m1_confirmation_enabled = False
        self.strategy.backtest_mode = True

        # Risk manager uses shimmed account/symbol info from the backtest balance
        self.balance = starting_balance
        self.initial_balance = starting_balance
        self.risk_manager = RiskManager(self.config, _BacktestMT5Shim(self.balance, ''))

        # State
        self.trades: list[Dict] = []
        self.open_positions: Dict[str, Dict] = {}
        self._pip_value_per_lot: float = 10.0  # default, refined at run()

    @staticmethod
    def _tf_const(tf: str) -> Optional[int]:
        mapping = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        return mapping.get(tf)

    def _load_history(self, symbol: str, start_date: datetime, end_date: datetime, timeframes: list[str]) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            tfc = self._tf_const(tf)
            if not tfc:
                continue
            rates = mt5.copy_rates_range(symbol, tfc, start_date, end_date)
            if rates is None or len(rates) == 0:
                logger.warning(f"No historical data for {symbol} {tf} in range {start_date} -> {end_date}")
                continue
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            data[tf] = df
        # Diagnostics: log loaded sizes
        for tf, df in data.items():
            logger.info(f"Loaded {len(df)} bars for {symbol} {tf}")
        return data

    def _pip_size(self, symbol: str) -> float:
        try:
            from utils import get_pip_size as _gps
            si = mt5.symbol_info(symbol)
            if si:
                return float(_gps(si))
        except Exception:
            pass
        return 0.01 if symbol.endswith('JPY') else 0.0001

    def _simulate_tick(self, symbol: str, bar: pd.Series) -> Dict[str, float]:
        # MT5 bars are bid-based; reconstruct ask using spread (points)
        digits = 3 if symbol.endswith('JPY') else 5
        point = 10 ** -digits
        spread_points = float(bar['spread']) if 'spread' in bar else 20.0
        bid = float(bar['close'])
        ask = bid + spread_points * point
        return {'bid': bid, 'ask': ask}

    def _check_exits(self, symbol: str, bar: pd.Series):
        if symbol not in self.open_positions:
            return
        pos = self.open_positions[symbol]
        if pos['type'] == 'BUY':
            if bar['low'] <= pos['sl']:
                self._close_position(symbol, pos['sl'], 'SL')
            elif bar['high'] >= pos['tp']:
                self._close_position(symbol, pos['tp'], 'TP')
        else:
            if bar['high'] >= pos['sl']:
                self._close_position(symbol, pos['sl'], 'SL')
            elif bar['low'] <= pos['tp']:
                self._close_position(symbol, pos['tp'], 'TP')

    def _open_position(self, symbol: str, signal, size: float, time):
        self.open_positions[symbol] = {
            'type': 'BUY' if signal.type == 0 else 'SELL',
            'entry': signal.entry_price,
            'sl': signal.stop_loss,
            'tp': signal.take_profit,
            'size': size,
            'time': time,
            'confidence': signal.confidence,
        }
        logger.info(f"OPEN: {'BUY' if signal.type == 0 else 'SELL'} {symbol} @ {signal.entry_price:.5f}")

    def _close_position(self, symbol: str, exit_price: float, reason: str):
        if symbol not in self.open_positions:
            return
        pos = self.open_positions[symbol]
        pip = self._pip_size(symbol)
        pips = (exit_price - pos['entry']) / pip if pos['type'] == 'BUY' else (pos['entry'] - exit_price) / pip
        profit = pips * pos['size'] * self._pip_value_per_lot
        self.balance += profit
        self.trades.append({
            'symbol': symbol,
            'type': pos['type'],
            'entry': pos['entry'],
            'exit': exit_price,
            'pips': pips,
            'profit': profit,
            'reason': reason,
            'confidence': pos['confidence'],
        })
        del self.open_positions[symbol]
        logger.info(f"CLOSE: {symbol} @ {exit_price:.5f} ({reason}), Profit: {profit:.2f}")

    def _save_results(self) -> Dict:
        wins = [t for t in self.trades if t['profit'] > 0]
        losses = [t for t in self.trades if t['profit'] <= 0]
        results = {
            'summary': {
                'initial_balance': self.initial_balance,
                'final_balance': self.balance,
                'total_profit': self.balance - self.initial_balance,
                'total_trades': len(self.trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(self.trades) if self.trades else 0,
                'avg_win': sum(t['profit'] for t in wins) / len(wins) if wins else 0,
                'avg_loss': sum(t['profit'] for t in losses) / len(losses) if losses else 0,
            },
            'trades': self.trades,
        }
        filename = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {filename}")
        print("\n=== BACKTEST COMPLETE ===")
        print(f"Total trades: {results['summary']['total_trades']}")
        print(f"Win rate: {results['summary']['win_rate']:.2%}")
        print(f"Total profit: ${results['summary']['total_profit']:.2f}")
        print(f"Final balance: ${results['summary']['final_balance']:.2f}")
        return results

    async def run(self, symbol: str, start_date: datetime, end_date: datetime):
        logger.info(f"Backtesting {symbol} from {start_date} to {end_date}")

        symbol_cfg = next((s for s in self.config.symbols if s['name'] == symbol), None)
        if not symbol_cfg:
            logger.error(f"Symbol {symbol} not in config")
            return None

        tfs = symbol_cfg.get('timeframes', ['M15'])
        # Ensure M1 is available for confirmation
        tf_set = list(dict.fromkeys(tfs + ['M1']))
        data = self._load_history(symbol, start_date, end_date, tf_set)
        if not data:
            logger.error("No data loaded for backtest")
            return None

        # Primary timeframe = first listed timeframe
        primary_tf = tfs[0]
        primary = data[primary_tf]
        logger.info(f"Primary timeframe: {primary_tf} with {len(primary)} bars")

        m1_df = data.get('M1')

        # Compute pip value per lot from real symbol info if available
        try:
            si = mt5.symbol_info(symbol)
            if si and getattr(si, 'trade_tick_size', 0) > 0:
                from utils import get_pip_size as _gps
                pip_size = float(_gps(si))
                per_price_unit = float(si.trade_tick_value) / float(si.trade_tick_size)
                self._pip_value_per_lot = per_price_unit * pip_size
                logger.info(f"Pip value per lot for {symbol}: {self._pip_value_per_lot:.4f}")
        except Exception as _e:
            logger.debug(f"Using default pip value per lot: {_e}")

        # Prepare RiskManager shim with correct symbol
        real_si = mt5.symbol_info(symbol)
        self.risk_manager.mt5_client = _BacktestMT5Shim(self.balance, symbol)

        for i in range(self.config.lookback_period, len(primary)):
            bar_time = primary.index[i]
            bar = primary.iloc[i]

            # Simulate tick so strategy can use spread and current price
            tick = self._simulate_tick(symbol, bar)

            # Monkeypatch MetaTrader5 symbol_info_tick/symbol_info for strategy only during this iteration
            real_symbol_info_tick = mt5.symbol_info_tick
            real_symbol_info = mt5.symbol_info
            real_copy_rates_from_pos = getattr(mt5, 'copy_rates_from_pos', None)
            try:
                def _mock_symbol_info_tick(_):
                    class T: pass
                    t = T()
                    t.bid = tick['bid']
                    t.ask = tick['ask']
                    return t

                def _mock_symbol_info(_):
                    # Use real symbol info if available; fall back to mock defaults
                    return real_si if real_si is not None else _MockSymbolInfo(object())

                mt5.symbol_info_tick = _mock_symbol_info_tick  # type: ignore
                mt5.symbol_info = _mock_symbol_info            # type: ignore

                # Provide M1 bars aligned to bar_time for confirmation logic
                def _mock_copy_rates_from_pos(sym, timeframe, start_pos, count):
                    if timeframe == mt5.TIMEFRAME_M1 and m1_df is not None and len(m1_df) > 0:
                        # Slice up to current simulated time
                        df = m1_df[m1_df.index <= bar_time]
                        if len(df) == 0:
                            return None
                        tail = df.tail(max(1, int(count)))
                        # Build list of dicts compatible with DataFrame([...])
                        out = []
                        for ts, row in tail.iterrows():
                            out.append({
                                'time': int(pd.Timestamp(ts).timestamp()),
                                'open': float(row['open']),
                                'high': float(row['high']),
                                'low': float(row['low']),
                                'close': float(row['close']),
                                'spread': float(row['spread']) if 'spread' in row else 0.0,
                                'tick_volume': int(row['tick_volume']) if 'tick_volume' in row else 0,
                                'real_volume': int(row['real_volume']) if 'real_volume' in row else 0,
                            })
                        return out
                    # Fallback to real function for other timeframes
                    if real_copy_rates_from_pos is not None:
                        return real_copy_rates_from_pos(sym, timeframe, start_pos, count)
                    return None

                mt5.copy_rates_from_pos = _mock_copy_rates_from_pos  # type: ignore

                # Manage exits on the opening of this bar
                self._check_exits(symbol, bar)

                # Skip if position open
                if symbol in self.open_positions:
                    continue

                # Window of data up to current bar (inclusive)
                window = primary.iloc[: i + 1].tail(max(self.config.max_period, 50))

                # Determine trend from higher timeframe if provided
                trend = 'ranging'
                ht = next((tf for tf in ['H1', 'H4', 'D1'] if tf in data), None)
                if ht is not None:
                    htf = data[ht]
                    hwin = htf[htf.index <= bar_time].tail(50)
                    if len(hwin) >= 20:
                        trend = self.market_data.identify_trend(hwin)

                # Generate signal from strategy
                signal = self.strategy.generate_signal(window, symbol, trend)
                if not signal:
                    continue

                # Position sizing from risk manager using backtest balance
                self.risk_manager.mt5_client._account.balance = self.balance  # update live balance
                size = self.risk_manager.calculate_position_size(symbol, stop_loss_pips=signal.stop_loss_pips)
                if size <= 0:
                    continue

                self._open_position(symbol, signal, size, bar_time)

            finally:
                # Restore original functions to avoid side effects
                mt5.symbol_info_tick = real_symbol_info_tick  # type: ignore
                mt5.symbol_info = real_symbol_info            # type: ignore
                if real_copy_rates_from_pos is not None:
                    mt5.copy_rates_from_pos = real_copy_rates_from_pos  # type: ignore

        # Close any remaining positions at last close
        last_close = float(primary.iloc[-1]['close'])
        for sym in list(self.open_positions.keys()):
            self._close_position(sym, last_close, 'END')

        return self._save_results()


async def main():
    import argparse

    parser = argparse.ArgumentParser(description='Backtest the forex bot using historical data from MT5')
    parser.add_argument('--symbol', default='EURUSD')
    parser.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--balance', type=float, default=10000.0)

    args = parser.parse_args()
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    bt = Backtest(args.config, starting_balance=args.balance)
    await bt.run(args.symbol, start_date, end_date)

    # MT5 shutdown is safe even if not connected
    try:
        mt5.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
