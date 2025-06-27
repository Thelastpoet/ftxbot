"""
Generic, reusable components for the ICT Trading Bot.
These components are not ICT-specific and can be used by any MT5 trading system.
"""
import MetaTrader5 as mt5
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MetaTrader5Client:
    """Handles connection and basic interactions with MetaTrader 5."""
    
    def __init__(self):
        self.initialized = False
        self.connection_start_time = None

    def initialize(self):
        """Initialize MT5 connection."""
        try:
            self.initialized = mt5.initialize()
        except Exception as e:
            logger.error(f"Exception during mt5.initialize(): {e}")
            self.initialized = False
            return False

        if not self.initialized:
            logger.error(f"MT5 initialize() failed. Code: {mt5.last_error()}")
            return False

        self.connection_start_time = datetime.now()

        # Verify terminal and account
        terminal_info = self.get_terminal_info()
        if not terminal_info:
            logger.error("Failed to get terminal_info.")
            self.shutdown()
            return False

        if not terminal_info.trade_allowed:
            logger.error("Algorithmic trading NOT allowed in MT5.")
            self.shutdown()
            return False

        account_info = self.get_account_info()
        if not account_info:
            logger.error("Failed to get account_info.")
            self.shutdown()
            return False

        logger.info(f"Connected to account: {account_info.login}, "
                   f"Server: {account_info.server}, "
                   f"Balance: {account_info.balance:.2f} {account_info.currency}")
        return True

    def shutdown(self):
        """Shutdown MT5 connection."""
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            logger.info("MT5 connection closed.")

    def is_connected(self):
        """Check if MT5 is connected."""
        return self.initialized

    def get_account_info(self):
        """Get account information."""
        if not self.initialized:
            return None
        return mt5.account_info()

    def get_terminal_info(self):
        """Get terminal information."""
        if not self.initialized:
            return None
        return mt5.terminal_info()

    def get_symbol_info(self, symbol):
        """Get symbol information."""
        if not self.initialized:
            return None
        return mt5.symbol_info(symbol)

    def get_symbol_ticker(self, symbol):
        """Get current symbol ticker."""
        if not self.initialized:
            return None
        return mt5.symbol_info_tick(symbol)

    def get_current_positions(self, symbol=None):
        """Get current open positions."""
        if not self.initialized:
            return []
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        return positions if positions is not None else []

    def timeframe_to_mql(self, tf_string):
        """Convert timeframe string to MT5 constant."""
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1
        }
        mql_tf = tf_map.get(tf_string.upper())
        if mql_tf is None:
            raise ValueError(f"Unsupported timeframe string: {tf_string}")
        return mql_tf

class MarketDataProvider:
    """Fetches and prepares market data."""
    
    def __init__(self, mt5_client: MetaTrader5Client):
        self.client = mt5_client

    def get_ohlc(self, symbol, timeframe_str, count):
        """Fetch OHLC data for a symbol."""
        if not self.client.is_connected():
            logger.error(f"MT5 not connected for {symbol}.")
            return None
            
        try:
            timeframe_mql = self.client.timeframe_to_mql(timeframe_str)
        except ValueError as e:
            logger.error(f"{symbol}: {e}")
            return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe_mql, 0, count)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No rates returned for {symbol} on {timeframe_str}")
            return pd.DataFrame()

        # Convert to DataFrame
        ohlc_df = pd.DataFrame(rates)
        ohlc_df['time'] = pd.to_datetime(ohlc_df['time'], unit='s')
        ohlc_df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        # Set time as index
        ohlc_df.set_index('time', inplace=True)
        
        # Define the broker's timezone. For UTC+3, the string is 'Etc/GMT-3'.
        broker_tz = 'Etc/GMT-3'
        
        try:
            # Localize the naive broker time to its correct timezone (UTC+3)
            ohlc_df.index = ohlc_df.index.tz_localize(broker_tz)
            
            # Convert the localized time to the universal UTC standard
            ohlc_df = ohlc_df.tz_convert('UTC')
            
        except Exception as e:
            logger.error(f"Failed to convert timezone for {symbol}: {e}")
            # Return an empty dataframe or handle error as appropriate
            return pd.DataFrame()
        
        logger.debug(f"{symbol}: Fetched {len(ohlc_df)} candles, "
                    f"Range: {ohlc_df.index.min()} to {ohlc_df.index.max()}")
        
        return ohlc_df[['open', 'high', 'low', 'close', 'volume']]
    
    def get_h4_data(self, symbol, count=100):
        """Fetch actual H4 timeframe data from broker."""
        if not self.client.is_connected():
            logger.error(f"MT5 not connected for {symbol}.")
            return None
            
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No H4 rates returned for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            h4_df = pd.DataFrame(rates)
            h4_df['time'] = pd.to_datetime(h4_df['time'], unit='s')
            h4_df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            # Set time as index
            h4_df.set_index('time', inplace=True)
            
            # Define the broker's timezone
            broker_tz = 'Etc/GMT-3'
            
            try:
                # Localize and convert timezone
                h4_df.index = h4_df.index.tz_localize(broker_tz)
                h4_df = h4_df.tz_convert('UTC')
                
            except Exception as e:
                logger.error(f"Failed to convert timezone for {symbol} H4: {e}")
                return pd.DataFrame()
            
            logger.debug(f"{symbol}: Fetched {len(h4_df)} H4 candles, "
                        f"Range: {h4_df.index.min()} to {h4_df.index.max()}")
            
            return h4_df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error fetching H4 data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def get_timeframe_data(self, symbol, timeframe_str, count):
        """Fetch data for any timeframe directly from broker."""
        if not self.client.is_connected():
            logger.error(f"MT5 not connected for {symbol}.")
            return None
            
        try:
            timeframe_mql = self.client.timeframe_to_mql(timeframe_str)
        except ValueError as e:
            logger.error(f"{symbol}: {e}")
            return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe_mql, 0, count)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No {timeframe_str} rates returned for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        # Set time as index
        df.set_index('time', inplace=True)
        
        # Define the broker's timezone
        broker_tz = 'Etc/GMT-3'
        
        try:
            # Localize and convert timezone
            df.index = df.index.tz_localize(broker_tz)
            df = df.tz_convert('UTC')
            
        except Exception as e:
            logger.error(f"Failed to convert timezone for {symbol} {timeframe_str}: {e}")
            return pd.DataFrame()
        
        logger.debug(f"{symbol}: Fetched {len(df)} {timeframe_str} candles, "
                    f"Range: {df.index.min()} to {df.index.max()}")
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def get_daily_data(self, symbol, count=50):
        """Fetch daily timeframe data for proper ICT bias analysis."""
        if not self.client.is_connected():
            logger.error(f"MT5 not connected for {symbol}.")
            return None
            
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No daily rates returned for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            daily_df = pd.DataFrame(rates)
            daily_df['time'] = pd.to_datetime(daily_df['time'], unit='s')
            daily_df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            # Set time as index
            daily_df.set_index('time', inplace=True)
            
            # Define the broker's timezone
            broker_tz = 'Etc/GMT-3'
            
            try:
                # Localize and convert timezone
                daily_df.index = daily_df.index.tz_localize(broker_tz)
                daily_df = daily_df.tz_convert('UTC')
                
            except Exception as e:
                logger.error(f"Failed to convert timezone for {symbol}: {e}")
                return pd.DataFrame()
            
            logger.debug(f"{symbol}: Fetched {len(daily_df)} daily candles, "
                        f"Range: {daily_df.index.min()} to {daily_df.index.max()}")
            
            return daily_df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

class PositionSizer:
    """Calculates position size based on risk parameters."""
    
    def __init__(self, mt5_client: MetaTrader5Client, risk_percent):
        self.client = mt5_client
        self.risk_percent = risk_percent

    def calculate_volume(self, symbol, sl_price, order_type):
        """Calculate position volume based on risk."""
        account_info = self.client.get_account_info()
        symbol_info = self.client.get_symbol_info(symbol)
        ticker = self.client.get_symbol_ticker(symbol)

        if not all([account_info, symbol_info, ticker]):
            logger.error(f"{symbol}: Missing data for position sizing")
            return None
            
        balance = account_info.balance
        risk_amount = (self.risk_percent / 100.0) * balance
        
        current_price = ticker.ask if order_type == "BUY" else ticker.bid
        
        if (order_type == "BUY" and sl_price >= current_price) or \
           (order_type == "SELL" and sl_price <= current_price):
            logger.error(f"{symbol}: Invalid SL {sl_price} vs price {current_price}")
            return None
            
        sl_distance_price = abs(current_price - sl_price)
        if sl_distance_price == 0:
            logger.error(f"{symbol}: SL distance is zero")
            return None
        
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        
        # Calculate loss for a 1.0 lot trade
        loss_per_lot = (sl_distance_price / tick_size) * tick_value
        
        if loss_per_lot <= 0:
            logger.error(f"{symbol}: Calculated loss per lot is {loss_per_lot}. Cannot size position.")
            return None

        volume = risk_amount / loss_per_lot

        lot_step = symbol_info.volume_step
        volume = round(volume / lot_step) * lot_step
        
        volume = max(symbol_info.volume_min, min(symbol_info.volume_max, volume))
        
        if volume < symbol_info.volume_min:
            logger.warning(f"{symbol}: Calculated volume {volume} is below minimum {symbol_info.volume_min}. No trade.")
            return None
            
        logger.info(f"{symbol}: Volume={volume:.2f}, Risk=${risk_amount:.2f} for SL distance of {sl_distance_price:.5f}")
        return volume

class TradeExecutor:
    """Executes trades on MT5."""
    
    def __init__(self, mt5_client: MetaTrader5Client, magic_prefix):
        self.client = mt5_client
        self.magic_prefix = magic_prefix

    def place_market_order(self, symbol, order_type, volume, sl_price, tp_price, comment=""):
        """Place a market order."""
        if not self.client.is_connected():
            return None
            
        ticker = self.client.get_symbol_ticker(symbol)
        if not ticker:
            return None
            
        # Determine order type and price
        mt5_order_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
        price = ticker.ask if order_type == "BUY" else ticker.bid
        
        # Validate SL/TP
        if (order_type == "BUY" and (sl_price >= price or tp_price <= price)) or \
           (order_type == "SELL" and (sl_price <= price or tp_price >= price)):
            logger.error(f"{symbol}: Invalid SL/TP levels")
            return None
            
        # Generate unique magic number
        magic = int(f"{self.magic_prefix}{int(datetime.now().timestamp()) % 100000}")
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5_order_type,
            "price": price,
            "sl": float(sl_price),
            "tp": float(tp_price),
            "deviation": 20,
            "magic": magic,
            "comment": comment or f"ICT_{symbol}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"{symbol}: Order send returned None")
            return None
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"{symbol}: Order failed - {result.retcode}: {result.comment}")
            return None
            
        logger.info(f"{symbol}: Order executed - Ticket: {result.order}")
        return result

class SymbolManager:
    """Manages symbol-specific data and state."""
    
    def __init__(self, mt5_client: MetaTrader5Client):
        self.client = mt5_client
        self.symbols = {}
        
    def initialize_symbols(self, symbol_list):
        """Initialize symbol data."""
        valid_symbols = []
        
        for symbol in symbol_list:
            info = self.client.get_symbol_info(symbol)
            if not info:
                logger.error(f"Failed to get info for {symbol}")
                continue
                
            self.symbols[symbol] = {
                'point': info.point,
                'digits': info.digits,
                'spread': info.spread,
                'description': info.description,
                'min_volume': info.volume_min,
                'max_volume': info.volume_max,
                'volume_step': info.volume_step,
                'has_position': False,
                'position_type': None,
                'last_signal_time': None
            }
            valid_symbols.append(symbol)
            
        logger.info(f"Initialized {len(valid_symbols)} symbols: {valid_symbols}")
        return valid_symbols
    
    def update_position_status(self, symbol):
        """Update position status for a symbol."""
        if symbol not in self.symbols:
            return
            
        positions = self.client.get_current_positions(symbol)
        
        if positions:
            self.symbols[symbol]['has_position'] = True
            self.symbols[symbol]['position_type'] = "BUY" if positions[0].type == mt5.ORDER_TYPE_BUY else "SELL"
        else:
            self.symbols[symbol]['has_position'] = False
            self.symbols[symbol]['position_type'] = None
    
    def get_symbol_data(self, symbol):
        """Get symbol data."""
        return self.symbols.get(symbol)
    
    def check_spread(self, symbol, max_spread_points):
        """Check if spread is acceptable. Returns raw spread value if OK, else None."""
        ticker = self.client.get_symbol_ticker(symbol)
        if not ticker:
            return None
            
        symbol_data = self.get_symbol_data(symbol)
        if not symbol_data:
            return None
            
        raw_spread = ticker.ask - ticker.bid
        spread_points = round(raw_spread / symbol_data['point'])
        
        if spread_points > max_spread_points:
            logger.info(f"{symbol}: Spread {spread_points} > max {max_spread_points}. Skipping.")
            return None
            
        return raw_spread