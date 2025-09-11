# Strategy Tuning Proposals (Review Only)

Purpose
- Improve trade quality and expectancy by addressing logic that can cause late/false entries, overtrading, and miscalibrated risk.
- Keep changes surgical and configurable via `config.json`.

Quick Recommendations
- Use closed-candle momentum for direction; avoid forming-candle bias.
- Convert hard counter‑trend blocks into strong penalties (soft filter).
- Tighten confidence model: lower base/bonuses, raise penalties; increase `min_confidence`.
- Calibrate breakout threshold and “room to structure” adaptively with ATR.
- Align time filters with the actual timeframe being evaluated; fix timezone handling.
- Reduce SL buffers; ensure SL sits just beyond structure with volatility-aware padding.
- Strengthen S/R detection using longer lookback + multi-window clustering.
- Respect sessions in backtests; avoid Asia chop.

File References
- `new_ftx/strategy.py:95` (breakout detection)
- `new_ftx/strategy.py:157` (M1 confirmation)
- `new_ftx/strategy.py:227` (confidence model)
- `new_ftx/strategy.py:320` (stop-loss construction/buffer)
- `new_ftx/strategy.py:365` (generate_signal entry)
- `new_ftx/strategy.py:428` (time-based filters)
- `new_ftx/strategy.py:451` (momentum filter – forming candle)
- `new_ftx/strategy.py:530` (room-to-next-structure)

---

**1) Candle Use And M1 Confirmation**
- Problem: Direction/momentum come from the forming M15 candle unless M1 confirmation overrides it (risk of false momentum).
- Proposal:
  - Primary: Use the last closed M15 candle for momentum/direction when operating on M15 data.
  - If `m1_confirmation_enabled`, use last closed M1 candle(s) for direction only after a level break is detected.
  - Add `min_m1_bars` and `m1_max_latency_ms` for confirmation robustness; degrade gracefully to closed M15 if M1 insufficient.
- Implementation Sketch:
  - After breakout detection, fetch last closed M1 (or N) to set `is_bullish_candle` and `candle_body/range` for confidence. If unavailable, use last closed M15; never the forming M15.
- Config Knobs:
  - `m1_confirmation_enabled`, `m1_confirmation_candles`, `m1_confirmation_buffer_pips`, `use_closed_m15_momentum` (default true).

**2) Time Filters And Timeframe Alignment**
- Problem: `time_remaining` uses the first configured timeframe, which may not match the data passed to `generate_signal` (e.g., evaluating H1 while config lists M15 first). Timezone handling uses `replace(tzinfo=UTC)` which can misinterpret naive timestamps.
- Proposal:
  - Detect timeframe from data frequency (index deltas) or pass `current_timeframe` explicitly into `generate_signal`.
  - Convert candle time with `pd.to_datetime(..., utc=True)` or localize properly before converting to UTC.
  - In backtests, bypass time gating entirely (already done via `backtest_mode`).
- Implementation Sketch:
  - Determine seconds per bar from data index; compute `time_remaining` accordingly.
- Config Knobs:
  - `time_filters_enabled` (global) and per‑timeframe overrides if needed.

**3) Confidence Model Calibration** (`new_ftx/strategy.py:227`)
- Problem: Base + bonuses can exceed `min_confidence=0.6` too easily; spread penalty is small relative to bonuses; overtrades in chop.
- Proposal:
  - Reduce base from 0.3 → 0.2; remove unconditional +0.1 direction bonus (direction already a hard check) or make it +0.05.
  - Momentum weight: cap using ATR-normalized body and include wick/upper-lower shadow ratio to avoid exhaustive candles.
  - Increase spread penalty magnitude; add extension penalty if distance beyond level is large relative to ATR.
  - Increase `min_confidence` to 0.70–0.75 by default.
- Implementation Sketch:
  - `confidence = 0.2 + 0.2*strength + 0.25*momentum - 0.15*spreadImpact - 0.1*extension`.
  - Define `extension = breakout.distance / max(atr, eps)`; penalize when `extension > 0.6–0.8` ATR.
- Config Knobs:
  - `min_confidence`, `confidence_weights.{strength,momentum,spread,extension,trend}`.

**4) Breakout Threshold & Entry Price** (`new_ftx/strategy.py:95`)
- Problem: Requires price to exceed level + threshold and then enters at current tick, often after the move is extended.
- Proposal:
  - Adaptive threshold: `threshold = max(fixed_pips, k1*ATR)`; set `k1` per symbol.
  - Entry improvement option: allow limit-on-break pullback entry (enter on first small retracement toward the broken level within X pips/time).
  - Reject over-extended breaks: if `distance > k2*ATR`, skip (or heavily penalize confidence).
- Config Knobs:
  - `breakout_threshold_pips`, `breakout_threshold_atr_mult`, `max_break_extension_atr`.

**5) Room-To-Structure Check** (`new_ftx/strategy.py:530`)
- Problem: `room_req = max(1.5*minSL, 0.8*ATR)` may be too strict/lenient depending on regime.
- Proposal:
  - Use `room_req = max(k_sl*SL, k_atr*ATR)` with symbol/session‑specific `k_sl`, `k_atr` (e.g., `k_sl=1.8`, `k_atr=1.0` for majors; smaller for JPY pairs).
  - Optionally compute “nearest opposing structure” using more distant swing clusters to avoid micro-level blockage.
- Config Knobs:
  - `min_room_after_breakout_pips` OR `{room_req_sl_mult, room_req_atr_mult}` with per‑symbol overrides.

**6) Trend Alignment** (`new_ftx/strategy.py:273`)
- Problem: Hard-stopping counter-trend trades discards potential reversal setups; also risky if trend detection lags.
- Proposal:
  - Convert hard stop into strong penalty in confidence (e.g., -0.25) unless trend is “strong” and extension is high (then hard stop).
  - Define “strong” via LR angle + normalized slope and ATR.
- Config Knobs:
  - `trend_block_mode` = `hard|soft`, `trend_penalty`, `trend_strong_thresholds`.

**7) Stop Loss Construction** (`new_ftx/strategy.py:320`)
- Problem: `safety_buffer = max(spread + 1 pip, configured_buffer)` with default 10 pips can bloat SL and hurt R:R.
- Proposal:
  - Make buffer = `max(spread + 0.5 pip, min(stop_loss_buffer_pips, k*ATR))`; reduce default `stop_loss_buffer_pips` (e.g., 5 for majors, 8 for XAUUSD).
  - Re‑check SL is behind structure by >= 0.5–1.0 pip after rounding.
  - Allow `min_sl_atr_mult` tuning per symbol and session.
- Config Knobs:
  - `stop_loss_buffer_pips`, `stop_loss_buffer_atr_mult`, `min_sl_atr_mult` (per symbol override).

**8) Support/Resistance Extraction Quality**
- Problem: Using only the last `lookback_period` (default 20) bars for swing detection can produce weak levels; cluster proximity fixed at 20 pips may mis-scale across symbols.
- Proposal:
  - Use a longer historical window for S/R (e.g., 150–300 bars) while still evaluating signals on the recent subset.
  - Scale clustering proximity by ATR or pip_size dynamically (e.g., `max(10 pips, 0.3*ATR)`), with min/max bounds.
  - Weight level strength by recurrence count and recency (decay factor), select top-N by score.
- Config Knobs:
  - `sr_lookback_bars`, `sr_cluster_proximity_pips`, `sr_cluster_proximity_atr_mult`, `sr_max_levels`.

**9) Spread Handling**
- Problem: Penalty is mild; threshold fixed in pips when ATR unavailable; across assets (e.g., XAUUSD), sensitivity differs.
- Proposal:
  - Compute `max_spread` as function of ATR and time of day (session). Raise penalty to -0.2 when `spread/ATR > 0.2`.
  - Optional dynamic skip outside broker’s typical session spreads.
- Config Knobs:
  - `max_spread_atr_ratio` per session/symbol; `spread_penalty_weight`.

**10) M1 Data Availability (Live/Backtest)** (`new_ftx/strategy.py:157`)
- Problem: Early bar checks can lack enough M1 bars and skip good trades.
- Proposal:
  - Keep retries (already added), but add fallback: if M1 insufficient, use last closed M15 for direction/momentum and continue.
  - Pre‑warm M1 history on init (already added) and expose warmup size in config.
- Config Knobs:
  - `m1_warmup_bars`, `m1_retry_attempts`, `m1_retry_wait_ms`.

**11) Rounding And R:R Integrity**
- Problem: Rounding SL/TP to symbol precision can drift R:R materially.
- Proposal:
  - After rounding, recompute achieved R:R and optionally adjust TP slightly within a tolerance to maintain target R:R.
- Config Knobs:
  - `rr_preserve_after_rounding` (bool), `rr_adjustment_tolerance_pips`.

**12) Risk Management Interplay**
- Observation: Pip value calculation is fixed; ensure cross-asset correctness (XAUUSD, indices) in both live and backtests.
- Proposal:
- Add per-symbol pip value sanity checks; cap lot sizes during extreme volatility; daily max loss cap and max consecutive losses filter.
- Config Knobs:
  - `risk_per_trade`, `daily_loss_cap_r`, `max_consecutive_losses`, `max_positions_per_symbol`.

**13) Sessions In Backtests**
- Problem: Backtests ignore sessions and include Asia, which often degrades breakout performance.
- Proposal:
  - Respect `trading_sessions` during backtests; only consider bars whose timestamps fall inside configured local-time windows.
- Config Knobs:
  - Reuse existing `trading_sessions` (EAT); add `backtest_respect_sessions` flag.

**14) Logging & Diagnostics**
- Proposal:
  - Add counters for each rejection reason: spread, momentum, trend, room, M1, time filters. Emit a summary per run (and write to JSON).
  - Log per-signal computed components (strength, momentum, spreadImpact, extension) when a debug flag is on.
- Config Knobs:
  - `debug_signal_components` (bool), `log_rejection_summary` (bool).

**15) Parameterization & Per‑Symbol Overrides**
- Proposal:
  - Allow per‑symbol overrides for: `min_confidence`, `min_body_ratio`, `breakout_threshold_pips`, `stop_loss_buffer_pips`, `min_sl_atr_mult`, room multipliers, and trend mode.
  - Example: tighter thresholds on GBPJPY, wider buffers on XAUUSD.

Validation Plan
- Phase 1 (Safety):
  - Enable closed-candle momentum; keep everything else unchanged. Backtest 6–12 months per symbol with session gating. Metric: trade count, win rate, PF, MAE/MFE.
- Phase 2 (Selectivity):
  - Tighten confidence model (+min_confidence=0.7, spread/extension penalties). Expect fewer trades, higher WR and PF.
- Phase 3 (S/R & Room):
  - Increase S/R lookback; adaptive clustering proximity; adjust room multipliers. Validate reduced false breakouts.
- Phase 4 (Trend):
  - Switch to soft trend penalty; evaluate reversals at strong levels.
- Phase 5 (Per‑symbol tuning):
  - Apply symbol-specific overrides; test majors vs JPY vs XAUUSD separately.

Rollout Plan
- Guard all changes behind config flags with defaults matching current behavior.
- Sequentially enable features in live shadow mode (signals logged, not traded) before full activation.

Risks & Edge Cases
- Too strict filters may starve trades; use per‑symbol overrides to prevent overfitting.
- ATR unavailable: provide robust fixed‑pip fallbacks with sensible symbol defaults.
- History gaps: ensure S/R and M1 confirmation degrade gracefully.

Next Steps (if approved)
- I can prepare a minimal PR implementing sections 1, 2, and 3 behind flags, plus session gating in backtests, so you can A/B test quickly.

