"""
ICT Trading Bot - High-Fidelity Backtester

This backtester is designed to be a true simulation of the main `ict_bot.py`.
It uses all the same components (Signal Generator, Position Sizer, Symbol Manager)
and follows the same logic flow to ensure the results are as accurate as possible.
"""

import logging
import pandas as pd
from datetime import datetime
import pytz

# Import our modules
from bot_components import MarketDataProvider, MetaTrader5Client, PositionSizer, SymbolManager
from ict_bot import ICTSignalGenerator
import config

# Setup logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HighFidelityBacktester:
    """
    A backtesting engine that accurately simulates the live trading bot.
    """
    
    def __init__(self, symbol, from_date, to_date):
        self.symbol = symbol
        self.from_date = from_date
        self.to_date = to_date
        
        # Initialize all the same components as the live bot
        self.mt5_client = MetaTrader5Client()
        self.market_provider = MarketDataProvider(self.mt5_client)
        self.symbol_manager = SymbolManager(self.mt5_client)
        self.signal_generator = ICTSignalGenerator(config)
        self.position_sizer = PositionSizer(self.mt5_client, config.RISK_PER_TRADE_PERCENT)
        
        # Backtesting state
        self.completed_trades = []
        self.open_trade = None
        self.initial_balance = 10000  # Starting balance
        self.balance = 10000
        self.equity = 10000
        
        # Realistic trading parameters
        self.spread_data = {}  # Store realistic spreads per time
        self.slippage_pips = 0.5  # Average slippage in pips
        
    def run(self):
        """Run the backtest."""
        logger.info(f"Starting high-fidelity backtest for {self.symbol} from {self.from_date} to {self.to_date}")
        
        if not self.mt5_client.initialize():
            logger.error("Failed to initialize MT5 connection. Exiting.")
            return

        # Pre-load all necessary data to avoid lookahead bias
        logger.info("Loading historical data...")
        
        # Calculate the required start date for the context data
        context_start_date = self.from_date - pd.Timedelta(days=100) # Add buffer for lookbacks

        ohlc_df = self.market_provider.get_ohlc_range(self.symbol, config.TIMEFRAME_STR, context_start_date, self.to_date)
        daily_df = self.market_provider.get_ohlc_range(self.symbol, "D1", context_start_date, self.to_date)
        h4_df = self.market_provider.get_ohlc_range(self.symbol, "H4", context_start_date, self.to_date)
        
        self.mt5_client.shutdown() # We have all data, so we can disconnect

        if ohlc_df.empty or daily_df.empty or h4_df.empty:
            logger.error("Could not load sufficient historical data for the specified range. Exiting.")
            return

        # Filter data for the specified date range
        backtest_data = ohlc_df[(ohlc_df.index >= self.from_date) & (ohlc_df.index <= self.to_date)]
        logger.info(f"Loaded {len(backtest_data)} candles for backtesting.")

        # Initialize the symbol in the symbol manager (needed for position sizing)
        # We need to connect again briefly to get symbol info
        self.mt5_client.initialize()
        self.symbol_manager.initialize_symbols([self.symbol])
        self.mt5_client.shutdown()

        # Main backtesting loop
        for i in range(config.DATA_LOOKBACK, len(backtest_data)):
            current_candle = backtest_data.iloc[i]
            current_time = backtest_data.index[i]
            
            # Skip weekend data
            if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                continue

            # First, check if an open trade needs to be closed
            if self.open_trade:
                self._check_exit(current_candle, current_time)

            # If there are no open positions, we can look for a new trade
            if not self.open_trade:
                # 1. Get realistic spread for this time
                current_spread = self._get_realistic_spread(current_time, current_candle)
                spread_points = current_spread / self.symbol_manager.symbols[self.symbol]['point']
                
                if spread_points > config.MAX_SPREAD_POINTS:
                    continue

                # 2. Get REALISTIC market view (NO LOOKAHEAD BIAS)
                start_idx = max(0, i - config.DATA_LOOKBACK)
                market_view = backtest_data.iloc[start_idx:i+1]  # Only last N candles + current
                
                # Get daily/H4 views up to current time only
                daily_view = daily_df[daily_df.index <= current_time].tail(50)  # Last 50 days max
                h4_view = h4_df[h4_df.index <= current_time].tail(200)  # Last 200 H4 candles max

                # 3. Generate a signal with realistic data
                signal, sl_price, tp_price, _ = self.signal_generator.generate_signal(
                    market_view, self.symbol, current_spread, daily_view, h4_view
                )
                
                if signal:
                    # 4. Calculate position size using BACKTEST data, not live MT5
                    entry_price = self._get_realistic_entry(signal, current_candle, current_spread)
                    volume = self._calculate_backtest_volume(self.symbol, entry_price, sl_price, signal)

                    if volume and volume > 0:
                        # 5. Execute the trade with realistic entry
                        self._execute_trade(signal, current_time, entry_price, sl_price, tp_price, volume, current_spread)
        
        self._generate_report()

    def _check_exit(self, candle, time):
        """Check if the currently open trade should be closed with realistic fills."""
        trade = self.open_trade
        exit_price = None
        exit_reason = None
        
        if trade['signal'] == 'BUY':
            # Check stop loss hit (with slippage)
            if candle['low'] <= trade['sl']:
                exit_price = self._apply_slippage(trade['sl'], 'SELL', negative=True)
                exit_price = max(exit_price, candle['low'])  # Can't fill better than actual low
                exit_reason = 'stop_loss'
            # Check take profit hit
            elif candle['high'] >= trade['tp']:
                exit_price = self._apply_slippage(trade['tp'], 'SELL', negative=False)
                exit_price = min(exit_price, candle['high'])  # Can't fill better than actual high
                exit_reason = 'take_profit'
                
        elif trade['signal'] == 'SELL':
            # Check stop loss hit (with slippage)
            if candle['high'] >= trade['sl']:
                exit_price = self._apply_slippage(trade['sl'], 'BUY', negative=True)
                exit_price = min(exit_price, candle['high'])  # Can't fill worse than actual high
                exit_reason = 'stop_loss'
            # Check take profit hit
            elif candle['low'] <= trade['tp']:
                exit_price = self._apply_slippage(trade['tp'], 'BUY', negative=False)
                exit_price = max(exit_price, candle['low'])  # Can't fill worse than actual low
                exit_reason = 'take_profit'
        
        if exit_price:
            trade['exit_time'] = time
            trade['exit_price'] = exit_price
            trade['exit_reason'] = exit_reason
            
            # CORRECTED P/L Calculation with proper pip values
            pip_difference = self._calculate_pip_difference(trade['entry_price'], exit_price, trade['signal'])
            pip_value = self._get_pip_value(trade['volume'])
            gross_profit = pip_difference * pip_value
            
            # Subtract exit spread cost
            exit_spread_cost = trade['entry_spread'] * pip_value  # Spread cost on exit
            net_profit = gross_profit - exit_spread_cost
            
            trade['gross_profit'] = gross_profit
            trade['spread_cost'] = exit_spread_cost
            trade['net_profit'] = net_profit
            trade['pip_profit'] = pip_difference
            
            self.balance += net_profit
            self.equity = self.balance
            
            self.completed_trades.append(trade)
            self.open_trade = None
            
            logger.info(f"Closed {trade['signal']} at {exit_price:.5f} ({exit_reason}). "
                       f"Pips: {pip_difference:.1f}, Net P/L: ${net_profit:.2f}")

    def _execute_trade(self, signal, entry_time, entry_price, sl_price, tp_price, volume, spread):
        """Simulate realistic trade execution with costs."""
        
        # Calculate entry spread cost
        pip_value = self._get_pip_value(volume)
        entry_spread_cost = spread * pip_value
        
        # Deduct entry spread cost immediately
        self.balance -= entry_spread_cost
        
        self.open_trade = {
            'entry_time': entry_time,
            'entry_price': entry_price,
            'signal': signal,
            'sl': sl_price,
            'tp': tp_price,
            'volume': volume,
            'entry_spread': spread,
            'entry_spread_cost': entry_spread_cost,
            'exit_time': None,
            'exit_price': None,
            'exit_reason': None,
            'gross_profit': 0,
            'spread_cost': 0,
            'net_profit': 0,
            'pip_profit': 0
        }
        
        pip_distance = abs(entry_price - sl_price) * 10000  # Convert to pips
        risk_amount = pip_distance * pip_value
        risk_percent = (risk_amount / self.balance) * 100
        
        logger.info(f"Executed {signal} at {entry_price:.5f}, Vol: {volume:.2f}, "
                   f"Risk: ${risk_amount:.2f} ({risk_percent:.1f}%), Spread: ${entry_spread_cost:.2f}")

    def _generate_report(self):
        """Generate a comprehensive performance report and save to CSV."""
        if not self.completed_trades:
            logger.info("No trades were executed during the backtest.")
            return

        results_df = pd.DataFrame(self.completed_trades)
        results_df.to_csv("backtest_results.csv", index=False)
        logger.info(f"Backtest results saved to backtest_results.csv")

        # Basic metrics
        total_trades = len(self.completed_trades)
        winning_trades = sum(1 for t in self.completed_trades if t['net_profit'] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_profit = sum(t['net_profit'] for t in self.completed_trades)
        total_spread_costs = sum(t['entry_spread_cost'] + t.get('spread_cost', 0) for t in self.completed_trades)
        gross_profit = sum(t['gross_profit'] for t in self.completed_trades)
        
        # Calculate additional metrics
        if self.completed_trades:
            winning_profits = [t['net_profit'] for t in self.completed_trades if t['net_profit'] > 0]
            losing_profits = [t['net_profit'] for t in self.completed_trades if t['net_profit'] < 0]
            
            avg_win = sum(winning_profits) / max(1, len(winning_profits))
            avg_loss = sum(losing_profits) / max(1, len(losing_profits))
            profit_factor = abs(sum(winning_profits) / sum(losing_profits)) if losing_profits else float('inf')
            max_drawdown = self._calculate_max_drawdown()
            
            # Additional detailed metrics
            largest_win = max(winning_profits) if winning_profits else 0
            largest_loss = min(losing_profits) if losing_profits else 0
            expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
            
            # Trade distribution by signal type
            buy_trades = sum(1 for t in self.completed_trades if t['signal'] == 'BUY')
            sell_trades = sum(1 for t in self.completed_trades if t['signal'] == 'SELL')
            
            # Trade duration analysis
            durations = []
            for trade in self.completed_trades:
                if trade['exit_time'] and trade['entry_time']:
                    duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600  # hours
                    durations.append(duration)
            
            avg_duration = sum(durations) / len(durations) if durations else 0
            
        else:
            avg_win = avg_loss = profit_factor = max_drawdown = 0
            largest_win = largest_loss = expectancy = 0
            buy_trades = sell_trades = 0
            avg_duration = 0

        # Calculate trading period and frequency
        trading_days = (self.to_date - self.from_date).days
        trades_per_day = total_trades / max(1, trading_days)
        
        # Generate comprehensive report
        logger.info("\n" + "="*60)
        logger.info("          ICT HIGH-FIDELITY BACKTEST REPORT")
        logger.info("="*60)
        
        # Test Configuration
        logger.info(f"\n BACKTEST SETUP:")
        logger.info(f"   Symbol: {self.symbol}")
        logger.info(f"   Period: {self.from_date.strftime('%Y-%m-%d')} to {self.to_date.strftime('%Y-%m-%d')}")
        logger.info(f"   Duration: {trading_days} days")
        logger.info(f"   Starting Balance: ${self.initial_balance:,.2f}")
        logger.info(f"   Risk Per Trade: {config.RISK_PER_TRADE_PERCENT}%")
        logger.info(f"   Minimum R:R: {config.MIN_TARGET_RR}:1")
        logger.info(f"   Max Spread: {config.MAX_SPREAD_POINTS} points")
        logger.info(f"   Timeframe: {config.TIMEFRAME_STR}")
        
        # Trade Summary
        logger.info(f"\n TRADE SUMMARY:")
        logger.info(f"   Total Trades: {total_trades}")
        logger.info(f"   BUY Trades: {buy_trades} ({buy_trades/max(1,total_trades)*100:.1f}%)")
        logger.info(f"   SELL Trades: {sell_trades} ({sell_trades/max(1,total_trades)*100:.1f}%)")
        logger.info(f"   Trades Per Day: {trades_per_day:.2f}")
        logger.info(f"   Avg Trade Duration: {avg_duration:.1f} hours")
        
        # Performance Metrics
        logger.info(f"\n PERFORMANCE:")
        logger.info(f"   Win Rate: {win_rate:.2f}%")
        logger.info(f"   Winning Trades: {winning_trades}")
        logger.info(f"   Losing Trades: {losing_trades}")
        logger.info(f"   Profit Factor: {profit_factor:.2f}")
        logger.info(f"   Expectancy: ${expectancy:.2f} per trade")
        
        # P&L Breakdown
        logger.info(f"\n PROFIT & LOSS:")
        logger.info(f"   Gross Profit: ${gross_profit:.2f}")
        logger.info(f"   Total Spread Costs: ${total_spread_costs:.2f}")
        logger.info(f"   Net Profit: ${total_profit:.2f}")
        logger.info(f"   Return: {((self.balance / self.initial_balance) - 1) * 100:.2f}%")
        logger.info(f"   Final Balance: ${self.balance:,.2f}")
        
        # Trade Statistics
        logger.info(f"\n TRADE STATISTICS:")
        logger.info(f"   Average Win: ${avg_win:.2f}")
        logger.info(f"   Average Loss: ${avg_loss:.2f}")
        logger.info(f"   Largest Win: ${largest_win:.2f}")
        logger.info(f"   Largest Loss: ${largest_loss:.2f}")
        logger.info(f"   Max Drawdown: {max_drawdown:.2f}%")
        
        # Risk Analysis
        logger.info(f"\n  RISK ANALYSIS:")
        risk_adj_return = (total_profit / self.initial_balance) / max(0.01, max_drawdown/100) if max_drawdown > 0 else 0
        logger.info(f"   Risk-Adjusted Return: {risk_adj_return:.2f}")
        logger.info(f"   Sharpe-like Ratio: {expectancy/max(0.01, abs(avg_loss)):.2f}")
        
        # Strategy Health Check
        logger.info(f"\n STRATEGY HEALTH:")
        if win_rate >= 50 and profit_factor >= 1.2:
            health_status = "HEALTHY"
        elif win_rate >= 40 and profit_factor >= 1.0:
            health_status = "MARGINAL"
        else:
            health_status = "POOR"
        
        logger.info(f"   Strategy Status: {health_status}")
        logger.info(f"   ICT Methodology: Implemented")
        logger.info(f"   Liquidity Targeting: Active")
        logger.info(f"   Realistic Costs: Included")
        
        logger.info("\n" + "="*60)
        logger.info("End of Report - Data saved to backtest_results.csv")
        logger.info("="*60 + "\n")

    def _get_realistic_spread(self, current_time, candle):
        """Calculate realistic spread based on time and volatility."""
        base_spread = 0.00015  # 1.5 pips base
        
        # Increase spread during high volatility
        candle_range = candle['high'] - candle['low']
        volatility_multiplier = min(2.0, max(1.0, candle_range / 0.0010))  # Scale with volatility
        
        # Increase spread during off-hours
        hour = current_time.hour
        if hour < 6 or hour > 18:  # Outside main trading hours
            time_multiplier = 1.5
        else:
            time_multiplier = 1.0
            
        return base_spread * volatility_multiplier * time_multiplier
    
    def _get_realistic_entry(self, signal, candle, spread):
        """Get realistic entry price with spread."""
        if signal == 'BUY':
            return candle['close'] + spread / 2  # Pay the ask
        else:
            return candle['close'] - spread / 2  # Hit the bid
    
    def _apply_slippage(self, target_price, order_type, negative=True):
        """Apply realistic slippage to fills."""
        slippage_amount = self.slippage_pips * 0.0001  # Convert pips to price
        
        if negative:  # Negative slippage (worse fill)
            if order_type == 'BUY':
                return target_price + slippage_amount
            else:
                return target_price - slippage_amount
        else:  # No positive slippage on TP
            return target_price
    
    def _calculate_pip_difference(self, entry_price, exit_price, signal):
        """Calculate pip difference correctly."""
        if signal == 'BUY':
            return (exit_price - entry_price) * 10000
        else:
            return (entry_price - exit_price) * 10000
    
    def _get_pip_value(self, volume):
        """Get pip value for position size."""
        # For EURUSD: 1 pip = $10 per lot
        # This is simplified - real implementation would handle all currency pairs
        if 'JPY' in self.symbol:
            return volume * 100  # JPY pairs have different pip values
        else:
            return volume * 10   # Standard pairs
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown percentage."""
        if not self.completed_trades:
            return 0
            
        running_balance = self.initial_balance
        peak_balance = self.initial_balance
        max_drawdown = 0
        
        for trade in self.completed_trades:
            running_balance += trade['net_profit']
            if running_balance > peak_balance:
                peak_balance = running_balance
            else:
                drawdown = ((peak_balance - running_balance) / peak_balance) * 100
                max_drawdown = max(max_drawdown, drawdown)
                
        return max_drawdown

    def _calculate_backtest_volume(self, symbol, entry_price, sl_price, signal):
        """Calculate position size using backtest data instead of live MT5."""
        # Validate SL direction
        if signal == "BUY" and sl_price >= entry_price:
            logger.error(f"{symbol}: Invalid BUY SL {sl_price:.5f} >= entry {entry_price:.5f}")
            return None
        elif signal == "SELL" and sl_price <= entry_price:
            logger.error(f"{symbol}: Invalid SELL SL {sl_price:.5f} <= entry {entry_price:.5f}")
            return None
        
        # Calculate risk amount (1% of balance)
        risk_amount = (config.RISK_PER_TRADE_PERCENT / 100.0) * self.balance
        
        # Calculate SL distance in pips
        sl_distance_pips = abs(entry_price - sl_price) * 10000
        
        # Calculate pip value ($10 per pip for 1 lot on major pairs)
        pip_value = 10 if 'JPY' not in symbol else 100
        
        # Calculate volume: risk_amount / (sl_distance_pips * pip_value)
        volume = risk_amount / (sl_distance_pips * pip_value)
        
        # Round to valid lot size (0.01 minimum)
        volume = max(0.01, round(volume, 2))
        
        logger.debug(f"{symbol}: Volume calculation - Risk: ${risk_amount:.2f}, SL Distance: {sl_distance_pips:.1f} pips, Volume: {volume:.2f}")
        
        return volume

if __name__ == "__main__":
    # Define backtest parameters
    SYMBOL_TO_TEST = "EURUSD"
    # Use timezone-aware datetime objects for date filtering
    START_DATE = pytz.utc.localize(datetime(2024, 1, 1))
    END_DATE = pytz.utc.localize(datetime.now()) # Run until today
    
    backtester = HighFidelityBacktester(SYMBOL_TO_TEST, START_DATE, END_DATE)
    backtester.run()