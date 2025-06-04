import time
import logging
import math
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import traceback

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration constants
MAGIC_NUMBER = 234000
EA_COMMENT = "MTF EA"
DEFAULT_SYMBOLS = ['AUDUSD', 'CHFJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'EURCAD', 'GBPJPY', 'AUDCHF', 'AUDCAD', 'AUDJPY',
    'EURAUD', 'EURJPY', 'EURCHF', 'EURNZD', 'AUDNZD', 'GBPCHF', 'CADCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'NZDUSD']
DEFAULT_TIMEFRAMES = (mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1)
CHECK_INTERVAL = 300 

class MetaTrader5Client:
    def __init__(self):
        self.initialized = mt5.initialize()
        if not self.initialized:
            logging.error("Failed to initialize MT5")

    def is_initialized(self):
        return self.initialized

    def get_account_info(self):
        return mt5.account_info()

class MarketData:
    def __init__(self, symbol, timeframes):
        self.symbol = symbol
        self.timeframes = timeframes
        
        # Select symbol
        if not mt5.symbol_select(self.symbol, True):
            raise ValueError(f"Failed to select symbol {self.symbol}")
        
    def fetch_data(self, timeframe):
        """Fetch historical data for the given timeframe"""            
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, 1000)
        if rates is None or len(rates) == 0:
            logging.error(f"No rates available for {self.symbol} on timeframe {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        df['tick_volume'] = df['tick_volume'].replace(0, 1)
                        
        return df    
                                          
class TradeManager:
    def __init__(self, client, market_data):
        self.client = client
        self.market_data = market_data
        self.timeframes = market_data.timeframes
        self.order_manager = OrderManager(client, market_data)
        
        # Initialize timeframe hierarchy
        self.tf_higher = max(self.timeframes)
        self.tf_medium = sorted(self.timeframes)[1] 
        self.tf_lower = min(self.timeframes)
        
        # Position limits
        self.max_positions = 10
        self.max_per_symbol = 2
    
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range manually"""
        df = data.copy().reset_index(drop=True)
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['prev_close']).abs()
        df['tr3'] = (df['low'] - df['prev_close']).abs()
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        if len(df) < period + 1:
            return np.nan

        first_tr = df['true_range'].iloc[1 : period + 1]
        initial_atr = first_tr.mean()

        atr_values = [np.nan] * len(df)
        atr_values[period] = initial_atr

        for i in range(period + 1, len(df)):
            prev_atr = atr_values[i - 1]
            curr_tr = df['true_range'].iloc[i]
            atr_values[i] = (prev_atr * (period - 1) + curr_tr) / period

        return atr_values[-1]

    def analyze_higher_timeframe(self, data):
        """Higher Timeframe - Clear trend identification"""
        try:
            # Structure-based trend detection
            df = data.copy()
            
            # Calculate swing highs/lows
            df['swing_high'] = df['high'] == df['high'].rolling(6).max().shift(1)
            df['swing_low'] = df['low'] == df['low'].rolling(6).min().shift(1)
            
            # Get last 3 swing points
            recent_highs = df[df['swing_high']]['high'].tail(3)
            recent_lows = df[df['swing_low']]['low'].tail(3)
            
            # Predictive structure analysis
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-2]
                higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-2]
                
                if higher_highs and higher_lows:
                    trend = 'uptrend'
                elif not higher_highs and not higher_lows:
                    trend = 'downtrend'  
                else:
                    trend = 'unclear'
            else:
                trend = 'unclear'
                
            return {'trend': trend}
        
        except Exception as e:
            logging.error(f"Error in analyze_higher_timeframe: {str(e)}")
            return {'trend': 'unclear'}

    def analyze_medium_timeframe(self, data, htf_context):
        try:
            df = data.copy()
        
            # Get ATR and normalize to percentage
            atr_value = self.calculate_atr(df)
            atr_threshold = max(atr_value / df['close'].iloc[-1], 0.0007)
            
            df['volume_momentum'] = ((df['close'] - df['open']) / df['open']) * df['tick_volume']
            df['vm_ema'] = df['volume_momentum'].ewm(span=5).mean()
            df['price_momentum'] = df['close'].pct_change(5)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            trend = htf_context['trend']
            
            if trend == 'uptrend':
                setup_valid = (
                    latest['vm_ema'] > prev['vm_ema'] and  
                    latest['price_momentum'] > atr_threshold
                )
            elif trend == 'downtrend':
                setup_valid = (
                    latest['vm_ema'] < prev['vm_ema'] and  
                    latest['price_momentum'] < -atr_threshold
                )
            else:
                setup_valid = False
                
            return {'valid_setup': setup_valid}

        except Exception as e:
            logging.error(f"Error in analyze_medium_timeframe: {str(e)}")
            return {'valid_setup': False}

    def analyze_lower_timeframe(self, data, trade_direction):
        try:
            df = data.copy()
    
            # Calculate immediate breakout levels
            df['resistance'] = df['high'].rolling(10).max().shift(1)
            df['support'] = df['low'].rolling(10).min().shift(1)
            
            # Volume surge detection 
            df['avg_volume'] = df['tick_volume'].rolling(10).mean()
            volume_std = df['tick_volume'].rolling(10).std()
            volume_threshold = df['avg_volume'] + (2 * volume_std)
            df['volume_surge'] = df['tick_volume'] > volume_threshold
            
            latest = df.iloc[-1]
            
            if trade_direction == 'buy':
                # Only 2 conditions: breakout + volume
                entry_condition = (
                    latest['close'] > latest['resistance'] and 
                    latest['volume_surge']
                )
            else:
                entry_condition = (
                    latest['close'] < latest['support'] and
                    latest['volume_surge'] 
                )
                
            return {'valid_entry': entry_condition}

        except Exception as e:
            logging.error(f"Error in analyze_lower_timeframe: {str(e)}")
            return {'valid_entry': False}

    def check_for_signals(self, symbol):
        """Main signal checking method - ALWAYS returns tuple"""
        try:                            
            # Check position limits first
            positions = mt5.positions_get()
            if positions:
                total_positions = len(positions)
                symbol_positions = len([p for p in positions if p.symbol == symbol])
                
                if total_positions >= self.max_positions:
                    return False, f"Max total positions ({self.max_positions}) reached", None
                    
                if symbol_positions >= self.max_per_symbol:
                    return False, f"Max positions for {symbol} ({self.max_per_symbol}) reached", None

            # Fetch data for all timeframes
            data = {}
            for tf in self.timeframes:
                data[tf] = self.market_data.fetch_data(tf)
                if data[tf] is None:
                    logging.warning(f"[{symbol}] Failed to fetch data for timeframe {tf}")
                    return False, f"[{symbol}] No data for timeframe {tf}", None

            # 1. Higher timeframe trend
            htf_analysis = self.analyze_higher_timeframe(data[self.tf_higher])
            if htf_analysis['trend'] == 'unclear':
                return False, f"[{symbol}] No clear trend on higher timeframe", None
            
            logging.info(f"[{symbol}] HTF trend: {htf_analysis['trend']}, checking MTF setup...")
                        
            # 2. Medium timeframe setup
            mtf_analysis = self.analyze_medium_timeframe(data[self.tf_medium], htf_analysis)
            if not mtf_analysis['valid_setup']:
                return False, f"[{symbol}] No valid setup on medium timeframe", None

            logging.info(f"[{symbol}] MTF setup valid, checking LTF entry...")

            # 3. Lower timeframe entry
            trade_direction = 'buy' if htf_analysis['trend'] == 'uptrend' else 'sell'
            ltf_analysis = self.analyze_lower_timeframe(data[self.tf_lower], trade_direction)
            if not ltf_analysis['valid_entry']:
                return False, f"[{symbol}] No valid entry on lower timeframe", None

            logging.info(f"[{symbol}] All conditions met! Preparing {trade_direction} trade...")

            # All conditions met - prepare trade
            account_info = self.client.get_account_info()
            if account_info is None:
                logging.error(f"[{symbol}] Failed to get account info")
                return False, f"[{symbol}] Could not get account info", None
                
            atr_value = self.calculate_atr(data[self.tf_lower])
            
            # Calculate trade parameters with FIXED R:R ratio
            stop_loss, take_profit, stop_loss_pips = self.order_manager.calculate_sl_tp(
                symbol,
                trade_direction,
                atr_value,
                sl_multiplier=1.5,  # Risk 1.5x ATR
                tp_multiplier=3.0   # Reward 3x ATR = 2:1 R:R
            )
            
            # Simple position sizing - 1% risk
            lots_size = self.order_manager.calculate_lot_size(
                symbol, 
                account_info, 
                stop_loss_pips,
                risk_percent=0.01  # 1% risk per trade
            )

            logging.info(f"[{symbol}] Calculated lot size: {lots_size}")

            # Execute trade
            logging.info(f"[{symbol}] Executing {trade_direction} order...")
            result = self.order_manager.place_order(
                symbol,
                lots_size,
                trade_direction,
                stop_loss,
                take_profit
            )
                    
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                trade_info = {
                    'direction': trade_direction,
                    'entry_price': data[self.tf_lower]['close'].iloc[-1],
                    'lot_size': lots_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'stop_loss_pips': stop_loss_pips,
                    'atr': atr_value
                }
                logging.info(f"[{symbol}] Trade executed successfully: {trade_direction} {lots_size} lots")
                return True, f"[{symbol}] Trade executed successfully", trade_info
            else:
                error_msg = result.comment if result else "Unknown error"
                logging.error(f"[{symbol}] ❌ Order execution failed: {error_msg}")
                return False, f"[{symbol}] Order failed: {error_msg}", None

        except Exception as e:
            logging.error(f"[{symbol}] Error in check_for_signals: {str(e)}")
            logging.error(traceback.format_exc())
            return False, f"[{symbol}] Error: {str(e)}", None
        
class OrderManager:
    def __init__(self, client, market_data):
        self.client = client
        self.market_data = market_data
        
    def get_symbol_info(self, symbol):
        """Get symbol info with pip calculation utilities"""
        symbol_tick = mt5.symbol_info_tick(symbol)
        symbol_info = mt5.symbol_info(symbol)
        
        if not symbol_tick or not symbol_info:
            raise ValueError(f"Could not get info for {symbol}")
        
        pip_size = symbol_info.point * 10 if symbol_info.digits == 5 or symbol_info.digits == 3 else symbol_info.point
        pip_value_per_lot = symbol_info.trade_tick_value * (pip_size / symbol_info.trade_tick_size)
        
        return {
            'tick': symbol_tick,
            'info': symbol_info,
            'pip_size': pip_size,
            'pip_value_per_lot': pip_value_per_lot
        }
        
    def calculate_sl_tp(self, symbol, order_type, atr_value, sl_multiplier=1.5, tp_multiplier=3.0):
        """Calculate SL and TP with proper R:R ratio"""
        symbol_data = self.get_symbol_info(symbol)
        symbol_tick = symbol_data['tick']
        pip_size = symbol_data['pip_size']
        
        # Get current price and spread
        if order_type == 'buy':
            current_price = symbol_tick.ask
            stop_distance = atr_value * sl_multiplier
            profit_distance = atr_value * tp_multiplier
            
            stop_loss = current_price - stop_distance
            take_profit = current_price + profit_distance
        else:  # sell
            current_price = symbol_tick.bid
            stop_distance = atr_value * sl_multiplier
            profit_distance = atr_value * tp_multiplier
            
            stop_loss = current_price + stop_distance
            take_profit = current_price - profit_distance
            
        # Calculate stop loss in pips
        stop_loss_pips = stop_distance / pip_size
                    
        return stop_loss, take_profit, stop_loss_pips
    
    def calculate_lot_size(self, symbol, account_info, stop_loss_pips, risk_percent=0.01):
        """Simple lot size calculation based on fixed risk percentage"""
        risk_amount = account_info.balance * risk_percent
        symbol_data = self.get_symbol_info(symbol)
        symbol_info = symbol_data['info']
        pip_value_per_lot = symbol_data['pip_value_per_lot']
                
        # Calculate lot size
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        
        # Round to allowed lot step
        lot_step = symbol_info.volume_step
        lot_size = math.floor(lot_size / lot_step) * lot_step
        
        # Apply min/max constraints
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        
        # Additional safety check - never risk more than 2% even if calculation is wrong
        max_allowed_lots = (account_info.balance * 0.02) / (stop_loss_pips * pip_value_per_lot)
        lot_size = min(lot_size, max_allowed_lots)
        
        return lot_size
        
    def place_order(self, symbol, lots_size, order_type, stop_loss, take_profit):
        """Place market order with proper error handling"""
        symbol_data = self.get_symbol_info(symbol)
        symbol_tick = symbol_data['tick']
        symbol_info = symbol_data['info']
            
        # Verify symbol is tradeable
        if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            logging.error(f"Trading disabled for {symbol}")
            return None
                        
        if order_type == "buy":
            mt5_order_type = mt5.ORDER_TYPE_BUY
            entry_price = symbol_tick.ask
        else:  # sell
            mt5_order_type = mt5.ORDER_TYPE_SELL
            entry_price = symbol_tick.bid
                             
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots_size,
            "type": mt5_order_type,
            "price": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "magic": MAGIC_NUMBER,
            "comment": EA_COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Check the request
        check_result = mt5.order_check(request)
        if check_result is None:
            logging.error(f"Order check failed for {symbol}")
            return None
            
        if check_result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order check failed: {check_result.comment}")
            return check_result
                    
        # Send the order
        result = mt5.order_send(request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Order placed successfully for {symbol}: {order_type} {lots_size} lots")
        else:
            logging.error(f"Order failed for {symbol}: {result.comment}")

        return result

def main():
    client = MetaTrader5Client()
    if not client.is_initialized():
        logging.error("Failed to connect to MetaTrader 5")
        return
        
    symbols = DEFAULT_SYMBOLS
    timeframes = DEFAULT_TIMEFRAMES

    try:
        # Initialize market data and trade managers
        market_data_dict = {symbol: MarketData(symbol, timeframes) for symbol in symbols}
        trade_managers = {symbol: TradeManager(client, market_data_dict[symbol]) for symbol in symbols}

        while True:
            start_time = time.time()

            # Check each symbol for signals
            for symbol in symbols:
                success, message, trade_info = trade_managers[symbol].check_for_signals(symbol)
                
                if success and trade_info:
                    logging.info(f"{message} - {trade_info}")
                elif success is False and "Max" in message:
                    logging.info(message)

            end_time = time.time()
            cycle_duration = end_time - start_time
            
            # Sleep until next check
            sleep_time = max(0, CHECK_INTERVAL - cycle_duration)
            if sleep_time > 0:
                logging.info(f"Next check in {sleep_time:.0f} seconds...")
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("Shutdown requested by user")
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        mt5.shutdown()
        logging.info("Trading script terminated")

if __name__ == "__main__":
    main()