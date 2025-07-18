
"""
ICT Trade Management Engine

This module is responsible for managing all open positions according to the rules
defined in `trade_management_rules.md`.
"""

from turtle import position
import MetaTrader5 as mt5
from collections import defaultdict

# Import project modules
from components import MetaTrader5Client, MarketDataProvider
from ict_bot import ICTAnalyzer
import config

# Use the centralized logger
from logger import operations_logger as logger, log_trade_event

class TradeManager:
    """
    Manages open trades based on dynamic market conditions and ICT principles.
    """

    def __init__(self, mt5_client: MetaTrader5Client, market_provider: MarketDataProvider, analyzer: ICTAnalyzer):
        self.mt5_client = mt5_client
        self.market_provider = market_provider
        self.analyzer = analyzer
        # Flags to ensure management actions are only performed once per trade.
        self.managed_flags = defaultdict(lambda: defaultdict(bool))

    def manage_open_trades(self):
        if not self.mt5_client.is_connected():
            logger.warning("MT5 connection lost. Attempting to reconnect...")
            if not self.mt5_client.initialize():
                logger.error("MT5 reconnection failed. Cannot manage trades.")
                return

        positions = self.mt5_client.get_current_positions()
        if not positions:
            return

        bot_positions = [p for p in positions if str(p.magic).startswith(str(config.MAGIC_NUMBER_PREFIX))]
        if not bot_positions:
            logger.debug("No open positions managed by this bot.")
            return

        logger.info(f"Managing {len(bot_positions)} open position(s)...")
        for position in bot_positions:
            self.manage_single_trade(position)

    def manage_single_trade(self, position):
        """
        Applies all relevant management rules to a single open trade.
        """
        # 1. Check for Invalidation (highest priority)
        if self._handle_invalidation(position):
            return  # Trade was closed due to invalidation

        # 2. Check for Breakeven (if not already done)
        if not self.managed_flags[position.ticket]['breakeven']:
            self._handle_breakeven(position)

        # 3. Check for Partial Profits (if not already done)
        if not self.managed_flags[position.ticket]['partials']:
            self._handle_partials(position)

        # 4. Handle Stop Loss Trailing
        self._handle_trailing_sl(position)

    def _handle_invalidation(self, position) -> bool:
        # Check if the feature is enabled in config
        if not config.INVALIDATE_ON_CHOCH:
            return False

        logger.debug(f"[{position.symbol}] Checking for invalidation...")
        ohlc_df = self.market_provider.get_data(position.symbol, config.TIMEFRAME_STR, config.DATA_LOOKBACK)
        if ohlc_df is None:
            return False

        analysis = self.analyzer.analyze(ohlc_df, position.symbol)
        if not analysis or 'structure' not in analysis:
            return False

        structure = analysis['structure']
        choch = structure[structure['CHOCH'].notna()].tail(1)
        if choch.empty:
            return False

        trade_direction = 1 if position.type == mt5.ORDER_TYPE_BUY else -1
        if choch.iloc[0]['CHOCH'] == -trade_direction:
            # Narrative is invalidated. Instead of closing, move SL to breakeven to protect capital.
            is_buy_trade = position.type == mt5.ORDER_TYPE_BUY
            sl_is_in_loss = (
                (is_buy_trade and position.sl < position.price_open) or
                (not is_buy_trade and position.sl > position.price_open)
            )

            if sl_is_in_loss:
                logger.info(f"!!! NARRATIVE INVALIDATED for {position.symbol} trade {position.ticket}. CHoCH against position. Moving SL to Breakeven.")
                self._modify_position(position.ticket, position.price_open, position.tp)
                self.managed_flags[position.ticket]['breakeven'] = True # Mark as managed
            else:
                logger.debug(f"[{position.symbol}] CHoCH detected, but SL is already at or beyond breakeven. No action needed.")
            
            # Even if no action is taken, we return True to signify the invalidation event was handled.
            return True
        return False

    def _handle_breakeven(self, position):
        logger.debug(f"[{position.symbol}] Checking for breakeven...")
        risk_amount = abs(position.price_open - position.sl)
        if risk_amount == 0: return

        breakeven_target_price = 0
        if position.type == mt5.ORDER_TYPE_BUY:
            breakeven_target_price = position.price_open + (risk_amount * config.BREAKEVEN_R_MULTIPLIER)
            if position.price_current >= breakeven_target_price:
                if position.sl < position.price_open:
                    logger.info(f"[{position.symbol}] Trade {position.ticket} hit {config.BREAKEVEN_R_MULTIPLIER}R. Moving SL to Breakeven.")
                    self._modify_position(position.ticket, position.price_open, position.tp)
                    self.managed_flags[position.ticket]['breakeven'] = True
        else:  # SELL
            breakeven_target_price = position.price_open - (risk_amount * config.BREAKEVEN_R_MULTIPLIER)
            if position.price_current <= breakeven_target_price:
                if position.sl > position.price_open:
                    logger.info(f"[{position.symbol}] Trade {position.ticket} hit {config.BREAKEVEN_R_MULTIPLIER}R. Moving SL to Breakeven.")
                    self._modify_position(position.ticket, position.price_open, position.tp)
                    self.managed_flags[position.ticket]['breakeven'] = True

    def _handle_partials(self, position):
        logger.debug(f"[{position.symbol}] Checking for partials...")
        risk_amount = abs(position.price_open - position.sl)
        if risk_amount == 0: return

        target_2r = 0
        if position.type == mt5.ORDER_TYPE_BUY:
            target_2r = position.price_open + (2 * risk_amount)
            if position.price_current >= target_2r:
                logger.info(f"[{position.symbol}] Trade {position.ticket} hit 2R. Taking 50% partials.")
                self._close_partial_position(position, 0.5, "2R Partial Profit")
                self.managed_flags[position.ticket]['partials'] = True
        else:  # SELL
            target_2r = position.price_open - (2 * risk_amount)
            if position.price_current <= target_2r:
                logger.info(f"[{position.symbol}] Trade {position.ticket} hit 2R. Taking 50% partials.")
                self._close_partial_position(position, 0.5, "2R Partial Profit")
                self.managed_flags[position.ticket]['partials'] = True

    def _handle_trailing_sl(self, position):
        logger.debug(f"[{position.symbol}] Checking for trailing SL...")
        ohlc_df = self.market_provider.get_data(position.symbol, config.TIMEFRAME_STR, config.DATA_LOOKBACK)
        if ohlc_df is None: return

        swings = self.analyzer._get_swings(ohlc_df)
        if swings.empty: return

        if position.type == mt5.ORDER_TYPE_BUY:
            swing_lows = swings[swings['HighLow'] == -1].dropna()
            if len(swing_lows) < 2: return

            new_sl = swing_lows.iloc[-2]['Level']
            if new_sl > position.sl:
                logger.info(f"[{position.symbol}] Trailing SL for BUY {position.ticket} to {new_sl:.5f}")
                self._modify_position(position.ticket, new_sl, position.tp)

        else:  # SELL
            swing_highs = swings[swings['HighLow'] == 1].dropna()
            if len(swing_highs) < 2: return

            new_sl = swing_highs.iloc[-2]['Level']
            if new_sl < position.sl:
                logger.info(f"[{position.symbol}] Trailing SL for SELL {position.ticket} to {new_sl:.5f}")
                self._modify_position(position.ticket, new_sl, position.tp)

    def _modify_position(self, ticket, sl, tp):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": float(sl),
            "tp": float(tp),
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to modify position {ticket}: {result.comment}")
        else:
            logger.info(f"Position {ticket} modified successfully.")

    def _close_position(self, position, comment):
        # Ensure the ticker is valid before attempting to close
        ticker = self.mt5_client.get_symbol_ticker(position.symbol)
        if not ticker:
            logger.error(f"Could not get ticker for {position.symbol}. Cannot close position {position.ticket}.")
            return

        price = ticker.bid if position.type == mt5.ORDER_TYPE_BUY else ticker.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": price,
            "deviation": 20,
            "magic": position.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        
        if result is None:
            error_code, error_message = mt5.last_error()
            logger.error(f"Failed to send close order for {position.symbol} ({position.ticket}). MT5 Error: {error_code} - {error_message}")
            logger.error(f"Request details: {request}")
            return

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position {position.ticket}: {result.comment} (retcode: {result.retcode})")
        else:
            logger.info(f"Position {position.ticket} closed successfully.")
            # Assuming _log_close_event is defined elsewhere to log the trade details
            # self._log_close_event(position.ticket, position.symbol, position.volume, result.price, result.retcode, comment)

    def _close_partial_position(self, position, percentage, comment):
        volume_to_close = round(position.volume * percentage, 2)
        if volume_to_close < self.mt5_client.get_symbol_info(position.symbol).volume_min:
            logger.warning(f"Partial close volume for {position.ticket} is too small. Skipping.")
            return

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": volume_to_close,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": self.mt5_client.get_symbol_ticker(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else self.mt5_client.get_symbol_ticker(position.symbol).ask,
            "deviation": 20,
            "magic": position.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close partial on position {position.ticket}: {result.comment}")
        else:
            logger.info(f"Partial close on position {position.ticket} for {volume_to_close} lots successful.")
            self._log_management_event(position.ticket, "PARTIAL_CLOSE", comment, position.price_current)
