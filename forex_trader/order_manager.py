# order_manager.py
import MetaTrader5 as mt5
import math
import logging

class OrderManager:
    """
    Handles all trade execution and risk management logic.
    """
    def __init__(self, client, config):
        self.client = client
        self.config = config

    def calculate_position_size(self, symbol, stop_loss_pips):
        """Calculates lot size based on account balance, risk %, and SL in pips."""
        account_info = self.client.get_account_info()
        symbol_info = mt5.symbol_info(symbol)
        if not account_info or not symbol_info: return None

        risk_amount = account_info.balance * (self.config.RISK_PER_TRADE_PERCENT / 100)
        
        pip_value_per_lot = symbol_info.trade_tick_value
        if symbol_info.trade_tick_size != symbol_info.point:
             pip_value_per_lot = (pip_value_per_lot / symbol_info.trade_tick_size) * symbol_info.point
        
        risk_in_currency = stop_loss_pips * pip_value_per_lot
        if risk_in_currency <= 0: return None
        
        lot_size = risk_amount / risk_in_currency
        
        lot_step = symbol_info.volume_step
        lot_size = math.floor(lot_size / lot_step) * lot_step
        return max(min(lot_size, symbol_info.volume_max), symbol_info.volume_min)

    def place_order(self, symbol, direction, stop_loss, take_profit):
        """Sends an order request to the broker."""
        tick = mt5.symbol_info_tick(symbol)
        if not tick: return None
        
        order_type = mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == 'buy' else tick.bid
        
        stop_loss_pips = abs(price - stop_loss) / tick.point
        lot_size = self.calculate_position_size(symbol, stop_loss_pips)
        if not lot_size or lot_size <= 0:
            logging.error(f"[{symbol}] Invalid lot size calculated: {lot_size}")
            return None

        request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot_size, "type": order_type,
                   "price": price, "sl": stop_loss, "tp": take_profit, "magic": 234002, "comment": "PA_Trader_v3.0",
                   "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
                   
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order send failed, retcode={result.retcode}, comment={result.comment}")
        return result