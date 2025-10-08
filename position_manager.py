"""
Position Manager for live trade management (specific to this bot).

This module provides intelligent, configurable management of open positions:
- Break-even moves based on R-multiple or pips
- Trailing stops (ATR or step-based) with safety guards
- Partial take-profit tiers with optional stop adjustments
- Session-based protective adjustments near boundaries/blackouts (no auto-close)

It is NOT wired into main yet; integrate by calling
`await PositionManager.manage_open_positions()` inside the bot's loop.

Design notes
- Uses mt5_client.modify_position for SL/TP changes and mt5_client.close_position for partials
- Pulls ATR via MarketData; pip math via utils.get_pip_size
- Reads initial risk (R) from TradeLogger if available (ticket-matched),
  otherwise falls back to position.price_open vs position.sl
- Debounces modifications via per-ticket state: last action time, last trail anchor, partials done
- All prices are rounded to symbol precision and validated against stops level

PEP8, type hints, and dataclasses are used per repo conventions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5

from utils import get_pip_size

logger = logging.getLogger(__name__)


# -------------------------------
# Dataclasses for configuration
# -------------------------------

@dataclass
class BreakEvenConfig:
    enabled: bool = True
    trigger_r_multiple: float = 1.0  # move SL to BE at >= 1R
    buffer_pips: float = 0.3         # keep a small positive buffer


@dataclass
class PartialTPLevel:
    at_r_multiple: float            # e.g., 1.0 means at 1R
    close_fraction: float           # e.g., 0.5 closes half
    move_stop_to: str = "breakeven" # options: 'none', 'breakeven'


@dataclass
class TrailingConfig:
    mode: str = "disabled"          # 'disabled' | 'atr' | 'step'
    # ATR trailing
    atr_period: int = 14
    atr_multiplier: float = 1.25
    start_after_r_multiple: float = 1.0  # only trail after reaching this R
    # Step trailing (ratchet)
    step_trigger_pips: float = 10.0      # advance anchor each time price moves this much
    step_distance_pips: float = 8.0      # stop distance behind current price


@dataclass
class SessionExitConfig:
    """Session-aware protective adjustments near boundaries/blackouts.

    - No auto-closing. We only tighten stops (e.g., move to BE or ATR-based guard).
    - Uses MarketSession minutes_to_boundary() and get_phase().
    """
    enabled: bool = True
    boundary_minutes: int = 10           # apply protective action if boundary within N minutes
    respect_blackouts: bool = True       # e.g., rollover -> apply immediately
    min_r_to_keep: float = 0.5           # if >= this R, keep runner but tighten; else move to BE at least
    tighten_atr_multiplier: float = 1.0  # ATR-based guard distance when tightening


@dataclass
class GuardrailsConfig:
    min_seconds_between_mods: int = 15
    min_improvement_pips: float = 0.2
    max_spread_pips_for_mods: float = 4.0


@dataclass
class PositionManagerConfig:
    breakeven: BreakEvenConfig = field(default_factory=BreakEvenConfig)
    trailing: TrailingConfig = field(default_factory=TrailingConfig)
    session_exit: SessionExitConfig = field(default_factory=SessionExitConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    # Default timeframe for ATR and structure context
    manage_timeframe: str = "M15"


# -------------------------------
# Internal state
# -------------------------------

@dataclass
class _TicketState:
    last_action_ts: datetime = field(default_factory=lambda: datetime.min)
    last_trail_anchor: Optional[float] = None  # last price used to advance trailing
    partials_done: List[int] = field(default_factory=list)  # indices into configured partial levels


@dataclass
class PositionSnapshot:
    ticket: int
    symbol: str
    order_type: int  # mt5.ORDER_TYPE_BUY (0) or SELL (1)
    volume: float
    price_open: float
    price_current: float
    sl: Optional[float]
    tp: Optional[float]
    profit: float
    time_open: Optional[datetime]
    pip_size: float
    digits: int
    point: float
    initial_risk_pips: Optional[float]
    r_multiple: Optional[float]


class PositionManager:
    """Position management engine for the bot.

    Dependencies
    - mt5_client: MetaTrader5Client (used for positions, ticks, symbol info, modify/close)
    - market_data: MarketData (for ATR)
    - trade_logger: TradeLogger (optional; used to read initial SL/entry for R)

    Typical usage (not integrated yet):
        pm = PositionManager(mt5_client, market_data, trade_logger)
        await pm.manage_open_positions()  # call in main loop
    """

    def __init__(
        self,
        mt5_client: Any,
        market_data: Any,
        trade_logger: Optional[Any] = None,
        config: Optional[PositionManagerConfig] = None,
        symbol_overrides: Optional[Dict[str, PositionManagerConfig]] = None,
        session_manager: Optional[Any] = None,
    ) -> None:
        self.mt5_client = mt5_client
        self.market_data = market_data
        self.trade_logger = trade_logger
        self.config = config or PositionManagerConfig()
        self.symbol_overrides = symbol_overrides or {}
        self.session_manager = session_manager
        # ephemeral per-ticket state
        self._state: Dict[int, _TicketState] = {}
        # define default partial TP ladder (no partials by default)
        self._partials: Dict[str, List[PartialTPLevel]] = {}

    # -------------------------------
    # Public API
    # -------------------------------
    async def manage_open_positions(self, symbol: Optional[str] = None) -> None:
        """
        Manage all open positions (or those for a specific symbol).
        Safe to call each main loop iteration.
        """
        try:
            positions = self.mt5_client.get_positions(symbol) if symbol else self.mt5_client.get_all_positions()
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return

        if not positions:
            return

        # Pre-fetch data contexts per symbol (ATR, pip size, spread)
        per_symbol_ctx: Dict[str, Dict[str, Any]] = {}

        for pos in positions:
            try:
                snapshot, ctx = await self._build_snapshot_with_ctx(pos, per_symbol_ctx)
                if not snapshot:
                    continue
                sym_cfg = self._cfg_for_symbol(snapshot.symbol)

                # Guard: spread too large? skip management to avoid rejections
                spread_ok = self._spread_allows_mods(ctx, sym_cfg)
                if not spread_ok:
                    continue

                # 1) Break-even management
                if sym_cfg.breakeven.enabled:
                    await self._maybe_move_to_breakeven(snapshot, ctx, sym_cfg)

                # 2) Partial take-profits
                await self._maybe_take_partials(snapshot, ctx, sym_cfg)

                # 3) Trailing stop logic
                await self._maybe_trail_stop(snapshot, ctx, sym_cfg)

                # 4) Session-based protective adjustment (no auto-close)
                await self._maybe_session_exit(snapshot, ctx, sym_cfg)

            except Exception as e:
                logger.error(f"Error managing position: {e}", exc_info=True)

    # -------------------------------
    # Snapshot/Context helpers
    # -------------------------------
    async def _build_snapshot_with_ctx(
        self,
        pos: Any,
        per_symbol_ctx: Dict[str, Dict[str, Any]],
    ) -> Tuple[Optional[PositionSnapshot], Dict[str, Any]]:
        symbol = getattr(pos, "symbol", None)
        if not symbol:
            return None, {}

        ctx = per_symbol_ctx.get(symbol)
        if ctx is None:
            sym_info = self.mt5_client.get_symbol_info(symbol)
            if not sym_info:
                return None, {}
            pip_size = get_pip_size(sym_info)
            digits = getattr(sym_info, "digits", 5)
            point = getattr(sym_info, "point", 0.00001)
            tick = self.mt5_client.get_symbol_info_tick(symbol)
            spread = 0.0
            if tick:
                spread = max(0.0, float(getattr(tick, "ask", 0.0) - getattr(tick, "bid", 0.0)))

            # ATR context
            timeframe = self._cfg_for_symbol(symbol).manage_timeframe
            atr_series = None
            atr_value = None
            try:
                # fetch 2*period candles to stabilize ATR reading
                period = max(14, self._cfg_for_symbol(symbol).trailing.atr_period)
                df = await self.market_data.fetch_data(symbol, timeframe, num_candles=period * 2)
                if df is not None and len(df) >= period:
                    atr_series = self.market_data.calculate_atr(df, period=period)
                    atr_value = float(atr_series.iloc[-1]) if len(atr_series) > 0 else None
            except Exception as e:
                logger.debug(f"ATR fetch failed for {symbol}: {e}")

            ctx = {
                "sym_info": sym_info,
                "pip_size": pip_size,
                "digits": int(digits),
                "point": float(point),
                "tick": tick,
                "spread": float(spread),
                "atr": atr_value,
            }
            per_symbol_ctx[symbol] = ctx

        # Build snapshot
        ticket = int(getattr(pos, "ticket", 0))
        order_type = int(getattr(pos, "type", 0))
        price_open = float(getattr(pos, "price_open", 0.0))
        price_current = float(getattr(ctx.get("tick"), "ask", 0.0)) if order_type == mt5.ORDER_TYPE_BUY else float(getattr(ctx.get("tick"), "bid", 0.0))
        price_current = price_current or float(getattr(pos, "price_current", 0.0))
        sl = getattr(pos, "sl", None)
        tp = getattr(pos, "tp", None)
        profit = float(getattr(pos, "profit", 0.0) or 0.0)
        volume = float(getattr(pos, "volume", 0.0) or 0.0)
        time_open = self._coerce_dt(getattr(pos, "time", None))

        # derive initial risk in pips for R multiple
        initial_risk_pips = self._initial_risk_pips(ticket, symbol, order_type, price_open, sl, ctx["pip_size"]) 
        r_multiple = None
        if initial_risk_pips and initial_risk_pips > 0:
            move = price_current - price_open if order_type == mt5.ORDER_TYPE_BUY else price_open - price_current
            r_multiple = (move / ctx["pip_size"]) / initial_risk_pips if ctx["pip_size"] > 0 else None

        snap = PositionSnapshot(
            ticket=ticket,
            symbol=symbol,
            order_type=order_type,
            volume=volume,
            price_open=price_open,
            price_current=price_current,
            sl=float(sl) if sl is not None else None,
            tp=float(tp) if tp is not None else None,
            profit=profit,
            time_open=time_open,
            pip_size=ctx["pip_size"],
            digits=ctx["digits"],
            point=ctx["point"],
            initial_risk_pips=initial_risk_pips,
            r_multiple=r_multiple,
        )
        return snap, ctx

    # -------------------------------
    # Rules: Break-even
    # -------------------------------
    async def _maybe_move_to_breakeven(self, snap: PositionSnapshot, ctx: Dict[str, Any], cfg: PositionManagerConfig) -> None:
        r = snap.r_multiple
        if r is None or r < cfg.breakeven.trigger_r_multiple:
            return

        # compute BE price with small buffer in the favorable direction
        be_price = self._breakeven_price(snap, cfg)
        if be_price is None:
            return

        # new SL must improve and remain valid vs current price and stops level
        new_sl = self._clamp_stop_to_rules(snap, ctx, be_price)
        if new_sl is None:
            return

        # Only if this is a genuine improvement over existing SL
        if snap.sl is not None:
            if snap.order_type == mt5.ORDER_TYPE_BUY and new_sl <= snap.sl:
                return
            if snap.order_type == mt5.ORDER_TYPE_SELL and new_sl >= snap.sl:
                return

        await self._throttled_modify(snap.ticket, new_sl, snap.tp, snap.symbol, reason="breakeven")

    def _breakeven_price(self, snap: PositionSnapshot, cfg: PositionManagerConfig) -> Optional[float]:
        buf = cfg.breakeven.buffer_pips * snap.pip_size
        if snap.order_type == mt5.ORDER_TYPE_BUY:
            return snap.price_open + buf
        else:
            return snap.price_open - buf

    # -------------------------------
    # Rules: Partial take-profits
    # -------------------------------
    async def _maybe_take_partials(self, snap: PositionSnapshot, ctx: Dict[str, Any], cfg: PositionManagerConfig) -> None:
        levels = self._partials.get(snap.symbol, [])
        if not levels or snap.r_multiple is None:
            return

        st = self._state.setdefault(snap.ticket, _TicketState())
        for idx, lvl in enumerate(sorted(levels, key=lambda x: x.at_r_multiple)):
            if idx in st.partials_done:
                continue
            if snap.r_multiple < lvl.at_r_multiple:
                break

            # compute volume to close
            vol_to_close = max(0.0, min(snap.volume * float(lvl.close_fraction), snap.volume))
            if vol_to_close <= 0.0:
                continue

            # Attempt partial close
            if not self._debounce_ok(snap.ticket, cfg.guardrails):
                continue
            try:
                result = self.mt5_client.close_position(snap.ticket, volume=vol_to_close)
                if result and getattr(result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                    st.partials_done.append(idx)
                    logger.info(f"Partial close done: ticket={snap.ticket}, volume={vol_to_close:.2f}")
                    # Optional stop move after partial
                    if lvl.move_stop_to == "breakeven":
                        be_price = self._breakeven_price(snap, cfg)
                        if be_price is not None:
                            new_sl = self._clamp_stop_to_rules(snap, ctx, be_price)
                            if new_sl is not None:
                                self.mt5_client.modify_position(snap.ticket, sl=new_sl, tp=snap.tp)
                else:
                    logger.warning(f"Partial close failed for ticket={snap.ticket}")
            except Exception as e:
                logger.error(f"Partial close error ticket={snap.ticket}: {e}")

    # -------------------------------
    # Rules: Trailing stop
    # -------------------------------
    async def _maybe_trail_stop(self, snap: PositionSnapshot, ctx: Dict[str, Any], cfg: PositionManagerConfig) -> None:
        tcfg = cfg.trailing
        if tcfg.mode == "disabled":
            return

        # don't trail before the required R is reached
        if snap.r_multiple is None or snap.r_multiple < tcfg.start_after_r_multiple:
            return

        candidate: Optional[float] = None

        if tcfg.mode == "atr":
            atr = ctx.get("atr")
            if atr is None or atr <= 0:
                return
            distance = float(tcfg.atr_multiplier) * float(atr)
            if snap.order_type == mt5.ORDER_TYPE_BUY:
                candidate = snap.price_current - distance
            else:
                candidate = snap.price_current + distance

        elif tcfg.mode == "step":
            st = self._state.setdefault(snap.ticket, _TicketState())
            anchor = st.last_trail_anchor or snap.price_open
            move_needed = tcfg.step_trigger_pips * snap.pip_size
            advanced = False
            if snap.order_type == mt5.ORDER_TYPE_BUY:
                if snap.price_current >= anchor + move_needed:
                    st.last_trail_anchor = snap.price_current
                    advanced = True
                    candidate = snap.price_current - (tcfg.step_distance_pips * snap.pip_size)
            else:
                if snap.price_current <= anchor - move_needed:
                    st.last_trail_anchor = snap.price_current
                    advanced = True
                    candidate = snap.price_current + (tcfg.step_distance_pips * snap.pip_size)
            if not advanced:
                return  # no new trail step

        else:
            return

        if candidate is None:
            return

        # Never loosen the stop, only improve
        new_sl = self._clamp_stop_to_rules(snap, ctx, candidate)
        if new_sl is None:
            return

        if snap.sl is not None:
            if snap.order_type == mt5.ORDER_TYPE_BUY and new_sl <= snap.sl:
                return
            if snap.order_type == mt5.ORDER_TYPE_SELL and new_sl >= snap.sl:
                return

        await self._throttled_modify(snap.ticket, new_sl, snap.tp, snap.symbol, reason=f"trail_{tcfg.mode}")

    # -------------------------------
    # Rules: Session-based protective adjustments (no auto-close)
    # -------------------------------
    async def _maybe_session_exit(self, snap: PositionSnapshot, ctx: Dict[str, Any], cfg: PositionManagerConfig) -> None:
        scfg = cfg.session_exit
        if not scfg.enabled or not self.session_manager:
            return

        try:
            phase = self.session_manager.get_phase()
            mins_to_boundary = self.session_manager.minutes_to_boundary()
        except Exception:
            return

        # Apply if entering blackout or approaching boundary
        apply_now = False
        if phase.is_blackout and scfg.respect_blackouts:
            apply_now = True
        elif mins_to_boundary <= max(0, int(scfg.boundary_minutes)):
            apply_now = True
        if not apply_now:
            return

        r = snap.r_multiple or 0.0
        atr = ctx.get("atr") or 0.0

        # Compute candidate protective stops
        be_price = self._breakeven_price(snap, cfg)
        tighten_price = None
        if atr and atr > 0:
            dist = scfg.tighten_atr_multiplier * float(atr)
            if snap.order_type == mt5.ORDER_TYPE_BUY:
                tighten_price = snap.price_current - dist
            else:
                tighten_price = snap.price_current + dist

        # Decision: if R below threshold, ensure at least BE; else tighten via ATR if it improves
        desired = None
        if r < scfg.min_r_to_keep and be_price is not None:
            desired = be_price
        elif tighten_price is not None:
            desired = tighten_price
        elif be_price is not None:
            desired = be_price

        if desired is None:
            return

        new_sl = self._clamp_stop_to_rules(snap, ctx, desired)
        if new_sl is None:
            return

        if snap.sl is not None:
            if snap.order_type == mt5.ORDER_TYPE_BUY and new_sl <= snap.sl:
                return
            if snap.order_type == mt5.ORDER_TYPE_SELL and new_sl >= snap.sl:
                return

        await self._throttled_modify(snap.ticket, new_sl, snap.tp, snap.symbol, reason="session_guard")

    # -------------------------------
    # Utilities
    # -------------------------------
    def _cfg_for_symbol(self, symbol: str) -> PositionManagerConfig:
        return self.symbol_overrides.get(symbol, self.config)

    def _spread_allows_mods(self, ctx: Dict[str, Any], cfg: PositionManagerConfig) -> bool:
        spread = float(ctx.get("spread", 0.0))
        pip = float(ctx.get("pip_size", 0.0001))
        spread_pips = spread / pip if pip > 0 else 0.0
        if spread_pips > cfg.guardrails.max_spread_pips_for_mods:
            logger.debug(f"Skip management due to wide spread: {spread_pips:.1f}p")
            return False
        return True

    def _debounce_ok(self, ticket: int, guard: GuardrailsConfig) -> bool:
        st = self._state.setdefault(ticket, _TicketState())
        now = datetime.now()
        if (now - st.last_action_ts).total_seconds() < guard.min_seconds_between_mods:
            return False
        st.last_action_ts = now
        return True

    async def _throttled_modify(self, ticket: int, sl: Optional[float], tp: Optional[float], symbol: str, reason: str) -> None:
        try:
            if sl is None and tp is None:
                return
            # debounce to avoid frequent modify spam
            if not self._debounce_ok(ticket, self.config.guardrails):
                return
            result = self.mt5_client.modify_position(ticket, sl=sl, tp=tp)
            if result and getattr(result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Modify SL/TP ok ticket={ticket} ({reason})")
            else:
                logger.warning(f"Modify SL/TP failed ticket={ticket} ({reason})")
        except Exception as e:
            logger.error(f"Modify SL/TP error ticket={ticket} ({reason}): {e}")

    def _clamp_stop_to_rules(self, snap: PositionSnapshot, ctx: Dict[str, Any], desired: float) -> Optional[float]:
        """Validate and clamp a desired stop to broker constraints and digits.

        Ensures:
        - Correct side of price (protective)
        - Min distance to current price (trade_stops_level)
        - Rounds to symbol precision
        """
        sym_info = ctx.get("sym_info")
        if not sym_info:
            return None

        # respect min stop distance from current price
        min_stops_points = getattr(sym_info, "trade_stops_level", 0) or 0
        min_dist_price = float(min_stops_points) * float(ctx.get("point", 0.00001))
        price_now = snap.price_current

        target = float(desired)
        if snap.order_type == mt5.ORDER_TYPE_BUY:
            # stop must be strictly below current price
            max_allowed = price_now - min_dist_price
            target = min(target, max_allowed)
            # never above entry for BE/Trailing safety unless already above
        else:
            # for SELL, stop must be strictly above current price
            min_allowed = price_now + min_dist_price
            target = max(target, min_allowed)

        # round to digits
        point = snap.point
        digits = snap.digits
        if point and digits is not None:
            target = round(round(target / point) * point, int(digits))
        elif digits is not None:
            target = round(target, int(digits))

        # ensure improvement vs current SL by minimum threshold to avoid noise
        min_impr = self.config.guardrails.min_improvement_pips * snap.pip_size
        if snap.sl is not None:
            if snap.order_type == mt5.ORDER_TYPE_BUY and (target - snap.sl) < min_impr:
                return None
            if snap.order_type == mt5.ORDER_TYPE_SELL and (snap.sl - target) < min_impr:
                return None

        return float(target)

    def _initial_risk_pips(
        self,
        ticket: int,
        symbol: str,
        order_type: int,
        price_open: float,
        sl: Optional[float],
        pip_size: float,
    ) -> Optional[float]:
        """Derive initial SL distance in pips used to measure R.

        Priority:
        1) TradeLogger entry for this ticket (entry_price vs stop_loss)
        2) Position's current SL vs price_open
        3) None if not derivable
        """
        # From trade logger if available
        if self.trade_logger and hasattr(self.trade_logger, "trades"):
            try:
                for t in self.trade_logger.trades:
                    if int(t.get("ticket", -1)) == int(ticket):
                        entry = float(t.get("entry_price", price_open) or price_open)
                        stop = float(t.get("stop_loss", sl) or sl)
                        if entry is not None and stop is not None and pip_size > 0:
                            return abs(entry - stop) / pip_size
            except Exception:
                pass

        # Fallback: from current SL
        if sl is not None and pip_size > 0:
            try:
                return abs(price_open - float(sl)) / pip_size
            except Exception:
                return None
        return None

    def _coerce_dt(self, v: Any) -> Optional[datetime]:
        try:
            if v is None:
                return None
            if isinstance(v, datetime):
                return v
            # Some mt5 structs expose `time` as seconds since epoch
            return datetime.fromtimestamp(float(v))
        except Exception:
            return None

    # -------------------------------
    # Optional configuration helpers
    # -------------------------------
    def set_partials_for_symbol(self, symbol: str, levels: List[PartialTPLevel]) -> None:
        """Define partial take-profit ladder for a symbol.

        Example:
            pm.set_partials_for_symbol(
                "EURUSD",
                [PartialTPLevel(1.0, 0.5, "breakeven"), PartialTPLevel(2.0, 0.5, "none")]
            )
        """
        self._partials[symbol] = list(levels)
