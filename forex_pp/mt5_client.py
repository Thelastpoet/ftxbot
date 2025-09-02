"""
MetaTrader5 Client Module
Handles all interactions with the MT5 terminal
"""

import MetaTrader5 as mt5
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class MetaTrader5Client:
    """Encapsulates all MetaTrader 5 API interactions"""
    
    def __init__(self):
        self.mt5 = mt5
        self.initialized = False
        self.reconnect_attempts = 3
        self.reconnect_delay = 5  # seconds
        self._initialize()
    
    def _initialize(self):
        """Initialize MT5 connection"""
        for attempt in range(self.reconnect_attempts):
            try:
                if mt5.initialize():
                    self.initialized = True
                    terminal_info = mt5.terminal_info()
                    if terminal_info:
                        logger.info(f"Connected to MT5 terminal: {terminal_info.name}")
                        logger.info(f"MT5 version: {mt5.version()}")
                    return True
                else:
                    logger.error(f"MT5 initialization failed, attempt {attempt + 1}/{self.reconnect_attempts}")
                    if attempt < self.reconnect_attempts - 1:
                        time.sleep(self.reconnect_delay)
            except Exception as e:
                logger.error(f"Error initializing MT5: {e}")
                if attempt < self.reconnect_attempts - 1:
                    time.sleep(self.reconnect_delay)
        
        self.initialized = False
        return False
    
    def __del__(self):
        """Cleanup on destruction"""
        if self.initialized:
            mt5.shutdown()
            logger.info("MT5 connection shut down")
    
    def is_connected(self) -> bool:
        """Check if MT5 is connected"""
        if not self.initialized:
            return False
        
        terminal_info = mt5.terminal_info()
        return terminal_info is not None and terminal_info.connected
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to MT5"""
        logger.info("Attempting to reconnect to MT5")
        mt5.shutdown()
        self.initialized = False
        return self._initialize()
    
    def get_account_info(self) -> Optional[Any]:
        """Get account information"""
        if not self.is_connected():
            logger.error("MT5 not connected")
            return None
        
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return None
        
        return account_info
    
    def get_symbol_info(self, symbol: str) -> Optional[Any]:
        """Get symbol information"""
        if not self.is_connected():
            return None
        
        # Ensure symbol is selected in Market Watch
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            return None
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return None
        
        return symbol_info
    
    def get_symbol_info_tick(self, symbol: str) -> Optional[Any]:
        """Get latest tick for symbol"""
        if not self.is_connected():
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None
        
        return tick
    
    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int) -> Optional[Any]:
        """Copy rates from position"""
        if not self.is_connected():
            return None
        
        # Convert timeframe string to MT5 constant
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        timeframe_mt5 = timeframe_map.get(timeframe)
        if timeframe_mt5 is None:
            logger.error(f"Unknown timeframe: {timeframe}")
            return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, start_pos, count)

        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get rates for {symbol}")
            return None
        
        return rates
    
    def place_order(self, symbol: str, order_type: int, volume: float, 
                   price: Optional[float] = None, sl: Optional[float] = None, 
                   tp: Optional[float] = None, comment: str = "") -> Optional[Any]:
        """Place a market order"""
        if not self.is_connected():
            return None
        
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return None
        
        # Get current price if not specified
        if price is None:
            tick = self.get_symbol_info_tick(symbol)
            if tick is None:
                return None
            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        # Normalize volume to symbol's lot step
        lot_step = symbol_info.volume_step
        volume = round(volume / lot_step) * lot_step
        volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,  # Maximum price deviation
            "magic": 234000,  # Magic number for identification
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add stop loss if specified
        if sl is not None:
            request["sl"] = sl
        
        # Add take profit if specified
        if tp is not None:
            request["tp"] = tp
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error("Order send failed: No result returned")
            return None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None
        
        logger.info(f"Order placed successfully: {result.order}")
        return result
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Any]:
        """Get open positions"""
        if not self.is_connected():
            return []
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        return list(positions)
    
    def get_all_positions(self) -> List[Any]:
        """Get all open positions"""
        return self.get_positions()
    
    def close_position(self, ticket: int, volume: Optional[float] = None) -> Optional[Any]:
        """Close an open position"""
        if not self.is_connected():
            return None
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position or len(position) == 0:
            logger.error(f"Position {ticket} not found")
            return None
        
        position = position[0]
        
        # Determine close order type
        close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
        
        # Use position volume if not specified
        if volume is None:
            volume = position.volume
        
        # Get current price
        tick = self.get_symbol_info_tick(position.symbol)
        if tick is None:
            return None
        
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": position.symbol,
            "volume": volume,
            "type": close_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": f"Close position {ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send close order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error("Close order failed: No result returned")
            return None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close order failed: {result.retcode} - {result.comment}")
            return None
        
        logger.info(f"Position {ticket} closed successfully")
        return result
    
    def modify_position(self, ticket: int, sl: Optional[float] = None, 
                       tp: Optional[float] = None) -> Optional[Any]:
        """Modify stop loss and take profit of an open position"""
        if not self.is_connected():
            return None
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position or len(position) == 0:
            logger.error(f"Position {ticket} not found")
            return None
        
        position = position[0]
        
        # Prepare modification request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "sl": sl if sl is not None else position.sl,
            "tp": tp if tp is not None else position.tp,
            "magic": 234000,
            "comment": f"Modify position {ticket}",
        }
        
        # Send modification order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error("Modify order failed: No result returned")
            return None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Modify order failed: {result.retcode} - {result.comment}")
            return None
        
        logger.info(f"Position {ticket} modified successfully")
        return result
    
    def get_history_orders(self, from_date: datetime, to_date: datetime) -> List[Any]:
        """Get historical orders"""
        if not self.is_connected():
            return []
        
        orders = mt5.history_orders_get(from_date, to_date)
        
        if orders is None:
            return []
        
        return list(orders)
    
    def get_history_deals(self, from_date: datetime, to_date: datetime) -> List[Any]:
        """Get historical deals"""
        if not self.is_connected():
            return []
        
        deals = mt5.history_deals_get(from_date, to_date)
        
        if deals is None:
            return []
        
        return list(deals)