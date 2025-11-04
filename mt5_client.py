"""
MetaTrader5 Client Module
Handles all interactions with the MT5 terminal
"""

import MetaTrader5 as mt5
import logging
from typing import Optional, List, Any
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
        self._filling_mode_cache = {}
        # Automatically try to recover on connection loss
        self.auto_reconnect = True
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
    
    def _ensure_connected(self) -> bool:
        """Ensure connection is alive; try to reconnect if allowed."""
        if self.is_connected():
            return True
        if not self.auto_reconnect:
            return False
        for attempt in range(self.reconnect_attempts):
            try:
                logger.warning(f"MT5 not connected; reconnect attempt {attempt+1}/{self.reconnect_attempts}")
                if self.reconnect() and self.is_connected():
                    return True
            except Exception as e:
                logger.error(f"Reconnect attempt {attempt+1} failed: {e}")
            time.sleep(self.reconnect_delay)
        return self.is_connected()
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to MT5"""
        logger.info("Attempting to reconnect to MT5")
        mt5.shutdown()
        self.initialized = False
        return self._initialize()
    
    def get_account_info(self) -> Optional[Any]:
        """Get account information with auto-reconnect."""
        if not self._ensure_connected():
            logger.error("MT5 not connected")
            return None
        account_info = mt5.account_info()
        if account_info is None and self.auto_reconnect and self.reconnect():
            account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return None
        return account_info
    
    def get_symbol_info(self, symbol: str) -> Optional[Any]:
        """Get symbol information with auto-reconnect."""
        if not self._ensure_connected():
            return None
        # Ensure symbol is selected in Market Watch (selection may be lost after reconnect)
        if not mt5.symbol_select(symbol, True):
            if self.auto_reconnect and self.reconnect():
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to select symbol {symbol}")
                    return None
            else:
                logger.error(f"Failed to select symbol {symbol}")
                return None
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return None
        return symbol_info
    
    def get_symbol_info_tick(self, symbol: str) -> Optional[Any]:
        """Get latest tick for symbol with auto-reconnect."""
        if not self._ensure_connected():
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None and self.auto_reconnect and self.reconnect():
            tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None
        return tick

    def get_supported_filling_modes(self, symbol: str) -> list:
        """Return a list of supported filling modes for a symbol, best-effort."""
        info = self.get_symbol_info(symbol)
        modes = []
        try:
            if info is None:
                return []
            if hasattr(info, 'filling_modes') and info.filling_modes:
                try:
                    modes.extend(list(info.filling_modes))
                except Exception:
                    pass
            if hasattr(info, 'filling_mode') and info.filling_mode is not None:
                modes.append(info.filling_mode)
        except Exception:
            return []
        # Deduplicate while preserving order
        seen = set()
        out = []
        for m in modes:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def preferred_filling_mode(self, symbol: str, preference: Optional[list] = None) -> int:
        """Choose a preferred filling mode for a symbol using allowed modes and preference order."""
        if symbol in self._filling_mode_cache:
            return self._filling_mode_cache[symbol]
        modes = self.get_supported_filling_modes(symbol)
        # Default preference: IOC > FOK > RETURN
        IOC = getattr(self.mt5, 'ORDER_FILLING_IOC', None)
        FOK = getattr(self.mt5, 'ORDER_FILLING_FOK', None)
        RET = getattr(self.mt5, 'ORDER_FILLING_RETURN', None)
        default_pref = [m for m in [IOC, FOK, RET] if m is not None]
        pref = preference if preference else default_pref
        for p in pref:
            if p in modes:
                self._filling_mode_cache[symbol] = p
                return p
        # Fallback
        fallback = FOK if FOK is not None else (modes[0] if modes else self.mt5.ORDER_FILLING_FOK)
        self._filling_mode_cache[symbol] = fallback
        return fallback
    
    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int) -> Optional[Any]:
        """Copy rates from position with auto-reconnect and retry."""
        if not self._ensure_connected():
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

        # Ensure symbol is selected (can be dropped after reconnect)
        mt5.symbol_select(symbol, True)

        rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, start_pos, count)

        if rates is None or len(rates) == 0:
            logger.warning(f"Failed to get rates for {symbol}; attempting reconnect and retry")
            if self.auto_reconnect and self.reconnect():
                mt5.symbol_select(symbol, True)
                rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, start_pos, count)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get rates for {symbol}")
            return None
        return rates
    
    def place_order(self, symbol: str, order_type: int, volume: float, 
                   price: Optional[float] = None, sl: Optional[float] = None, 
                   tp: Optional[float] = None, comment: str = "",
                   deviation_points: Optional[int] = None,
                   type_filling_override: Optional[int] = None) -> Optional[Any]:
        """Place a market order (ensures connection)."""
        if not self._ensure_connected():
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
        
        # --- Build filling mode candidates (override -> symbol -> defaults) ---
        candidates = []
        # cached working fill first if available
        try:
            if position.symbol in self._filling_mode_cache:
                candidates.append(int(self._filling_mode_cache[position.symbol]))
        except Exception:
            pass
        # 1) cached working mode first if available
        try:
            if symbol in self._filling_mode_cache:
                candidates.append(int(self._filling_mode_cache[symbol]))
        except Exception:
            pass
        # 2) external override next
        try:
            if type_filling_override is not None and int(type_filling_override) not in candidates:
                candidates.append(int(type_filling_override))
        except Exception:
            pass
        try:
            # use symbol's reported modes next
            symbol_modes = []
            if hasattr(symbol_info, 'filling_mode') and getattr(symbol_info, 'filling_mode') is not None:
                symbol_modes.append(int(symbol_info.filling_mode))
            if hasattr(symbol_info, 'filling_modes') and getattr(symbol_info, 'filling_modes'):
                try:
                    symbol_modes.extend([int(m) for m in list(symbol_info.filling_modes)])
                except Exception:
                    pass
            for m in symbol_modes:
                if m not in candidates:
                    candidates.append(m)
        except Exception:
            pass
        # default preference at the end
        for m in [getattr(mt5, 'ORDER_FILLING_IOC', None), getattr(mt5, 'ORDER_FILLING_FOK', None), getattr(mt5, 'ORDER_FILLING_RETURN', None)]:
            if m is not None and m not in candidates:
                candidates.append(int(m))
        # final guard
        if not candidates:
            candidates = [int(getattr(mt5, 'ORDER_FILLING_FOK'))]
        
        # Derive deviation in points if not provided
        if deviation_points is None:
            try:
                tick = self.get_symbol_info_tick(symbol)
                spread_points = int(round(abs(float(tick.ask) - float(tick.bid)) / float(symbol_info.point))) if tick else 20
                # dynamic clamp: 1x spread, between 10 and 100 points
                deviation_points = max(10, min(100, spread_points))
            except Exception:
                deviation_points = 20

        # Try order with candidates sequentially, cache the one that works
        last_result = None
        unsupported_code = getattr(mt5, 'TRADE_RETCODE_INVALID_FILL', None)  # Some builds expose this
        for idx, type_filling in enumerate(candidates):
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": int(deviation_points),
                "magic": 234000,  # Magic number for identification
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": int(type_filling),
            }
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp

            result = mt5.order_send(request)
            last_result = result
            if result is None:
                logger.error("Order send failed: No result returned")
                break
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Cache working filling for this symbol
                try:
                    self._filling_mode_cache[symbol] = int(type_filling)
                except Exception:
                    pass
                logger.info(f"Order placed successfully: {getattr(result, 'order', None)}")
                return result
            # If unsupported filling mode, try next candidate; else, stop and return
            if int(getattr(result, 'retcode', -1)) in (10030, unsupported_code if unsupported_code is not None else -9999):
                if idx < len(candidates) - 1:
                    logger.warning(f"{symbol}: Filling mode {type_filling} unsupported, retrying with next option")
                    continue
            # Any other error: log and stop trying
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None

        return None
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Any]:
        """Get open positions"""
        if not self._ensure_connected():
            return []
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            if self.auto_reconnect and self.reconnect():
                positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            if positions is None:
                return []
        
        return list(positions)
    
    def get_all_positions(self) -> List[Any]:
        """Get all open positions"""
        return self.get_positions()
    
    def close_position(self, ticket: int, volume: Optional[float] = None) -> Optional[Any]:
        """Close an open position"""
        if not self._ensure_connected():
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
        
        # --- Build filling mode candidates for closing ---
        symbol_info = self.get_symbol_info(position.symbol)
        candidates = []
        try:
            if hasattr(symbol_info, "filling_mode") and getattr(symbol_info, 'filling_mode') is not None:
                candidates.append(int(symbol_info.filling_mode))
            if hasattr(symbol_info, "filling_modes") and getattr(symbol_info, 'filling_modes'):
                try:
                    for m in list(symbol_info.filling_modes):
                        if int(m) not in candidates:
                            candidates.append(int(m))
                except Exception:
                    pass
        except Exception:
            pass
        for m in [getattr(mt5, 'ORDER_FILLING_IOC', None), getattr(mt5, 'ORDER_FILLING_FOK', None), getattr(mt5, 'ORDER_FILLING_RETURN', None)]:
            if m is not None and int(m) not in candidates:
                candidates.append(int(m))
        if not candidates:
            candidates = [int(getattr(mt5, 'ORDER_FILLING_FOK'))]

        # Try close with candidates
        unsupported_code = getattr(mt5, 'TRADE_RETCODE_INVALID_FILL', None)
        last_result = None
        for idx, type_filling in enumerate(candidates):
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
                "type_filling": int(type_filling),
            }
            result = mt5.order_send(request)
            last_result = result
            if result is None:
                logger.error("Close order failed: No result returned")
                break
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Position {ticket} closed successfully")
                return result
            if int(getattr(result, 'retcode', -1)) in (10030, unsupported_code if unsupported_code is not None else -9999):
                if idx < len(candidates) - 1:
                    logger.warning(f"{position.symbol}: Filling mode {type_filling} unsupported on close, retrying")
                    continue
            logger.error(f"Close order failed: {result.retcode} - {result.comment}")
            return None

        return None
    
    def modify_position(self, ticket: int, sl: Optional[float] = None, 
                       tp: Optional[float] = None) -> Optional[Any]:
        """Modify stop loss and take profit of an open position"""
        if not self._ensure_connected():
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
        if not self._ensure_connected():
            return []
        
        orders = mt5.history_orders_get(from_date, to_date)
        
        if orders is None:
            return []
        
        return list(orders)
    
    def get_history_deals_by_position(self, position_id: int) -> List[Any]:
        """Get historical deals by position ID"""
        if not self._ensure_connected():
            return []

        try:
            # Query deals within a wide enough date window directly
            # Some MT5 Python builds do not provide `history_select`,
            # so pass the range to `history_deals_get` and filter by position.
            now = datetime.now()
            from_date = datetime(now.year - 2, 1, 1)
            try:
                deals = mt5.history_deals_get(from_date, now, position=position_id)
            except TypeError:
                # Older builds may not support the `position` kwarg; fetch all and filter client-side
                all_deals = mt5.history_deals_get(from_date, now)
                deals = [d for d in (all_deals or []) if getattr(d, 'position', None) == position_id]
            if deals:
                deals = sorted(deals, key=lambda d: d.time)
            return list(deals) if deals else []
        except Exception as e:
            logger.error(f"Error getting deals for position {position_id}: {e}")
            return []
    
    def get_history_deals(self, from_date: datetime, to_date: datetime) -> List[Any]:
        """Get historical deals"""
        if not self.is_connected():
            return []
        
        deals = mt5.history_deals_get(from_date, to_date)
        
        if deals is None:
            return []
        
        return list(deals)
