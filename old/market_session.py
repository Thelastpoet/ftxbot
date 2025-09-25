"""
Market Session module
Defines session phases (London/NY microstructure), rollover gating, and utilities
for session-aware breakout trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Optional

import logging

logger = logging.getLogger(__name__)

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover
    ZoneInfo = None  # Fallback handled below


@dataclass
class SessionPhase:
    name: str               # e.g. "LO_expansion", "NY_data_0830", "Rollover"
    session: str            # "Asia", "London", "NewYork", "Other"
    weight: float           # additive confidence weight (typical 0.0..0.2)
    ttl_minutes: int        # suggested time-to-follow-through horizon
    runner_bias: bool       # whether to allow runners more readily
    is_blackout: bool = False  # only for hard avoid windows like rollover


class MarketSession:
    """Session/phase awareness for FX breakout trading.

    - Provides lightweight, deterministic session phases using London/NY time.
    - Designed to be additive (weights), not restrictive.
    - Only hard blackout is rollover around 17:00 ET.
    """

    def __init__(
        self,
        config=None,
        tz_london: str = "Europe/London",
        tz_new_york: str = "America/New_York",
    ) -> None:
        self.config = config
        self._tz_london = self._safe_zoneinfo(tz_london)
        self._tz_ny = self._safe_zoneinfo(tz_new_york)

    def _safe_zoneinfo(self, name: str):
        if ZoneInfo is None:
            # Fallback: use local timezone to avoid crash; phases still compute
            try:
                return datetime.now().astimezone().tzinfo
            except Exception:
                return timezone.utc
        try:
            return ZoneInfo(name)
        except Exception:
            logger.warning(f"ZoneInfo load failed for {name}; using local tz")
            return datetime.now().astimezone().tzinfo

    def _local_times(self, now_utc: Optional[datetime]):
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        london = now_utc.astimezone(self._tz_london)
        ny = now_utc.astimezone(self._tz_ny)
        return now_utc, london, ny

    @staticmethod
    def _in_range(t: time, start: time, end: time) -> bool:
        """Return True if time t is within [start, end], handling ranges that cross midnight."""
        if start <= end:
            return start <= t <= end
        return t >= start or t <= end

    def is_trade_window(self, now_utc: Optional[datetime] = None) -> bool:
        """Hard gating only for rollover window. Everything else stays additive."""
        phase = self.get_phase(now_utc)
        return not phase.is_blackout

    def minutes_to_boundary(self, now_utc: Optional[datetime] = None) -> int:
        """Approximate minutes to next notable boundary (phase change)."""
        now_utc, london, ny = self._local_times(now_utc)

        candidates = []
        # London notable times
        for hh, mm in ((8, 0), (8, 15), (9, 30), (12, 0), (16, 0)):
            dt = london.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if dt < london:
                dt += timedelta(days=1)
            candidates.append(dt.astimezone(timezone.utc))

        # New York notable times
        for hh, mm in ((8, 0), (8, 30), (9, 30), (11, 30), (13, 30), (17, 0)):
            dt = ny.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if dt < ny:
                dt += timedelta(days=1)
            candidates.append(dt.astimezone(timezone.utc))

        mins = min(int((c - now_utc).total_seconds() // 60) for c in candidates)
        return max(0, mins)

    def get_reference_times(self, now_utc: Optional[datetime] = None):
        """Return a dict of reference times (UTC, London, New York, Local)."""
        now_utc, london, ny = self._local_times(now_utc)
        local = now_utc.astimezone()
        return {
            'utc': now_utc,
            'london': london,
            'new_york': ny,
            'local': local,
        }

    def get_phase(self, now_utc: Optional[datetime] = None, symbol: Optional[str] = None) -> SessionPhase:
        """Determine current phase based on London and New York clocks.

        Priority: Rollover > NY 08:30 > LO sweep > LO expansion > Overlap > NY cash open > Lunch > LC > Asia > Other
        """
        _now_utc, london, ny = self._local_times(now_utc)
        lt = london.timetz()
        nyt = ny.timetz()

        # 1) Rollover blackout: 17:00 ET ± 20 minutes
        if self._in_range(nyt, time(16, 40), time(17, 20)):
            return SessionPhase("Rollover", "Other", weight=-0.2, ttl_minutes=0, runner_bias=False, is_blackout=True)

        # 2) NY 08:30 data catalyst window (±10 minutes around 08:30)
        if self._in_range(nyt, time(8, 20), time(8, 40)):
            return SessionPhase("NY_data_0830", "NewYork", weight=0.10, ttl_minutes=45, runner_bias=True)

        # 3) London open microstructure
        if self._in_range(lt, time(8, 0), time(8, 15)):
            return SessionPhase("LO_sweep", "London", weight=0.05, ttl_minutes=25, runner_bias=False)
        if self._in_range(lt, time(8, 15), time(9, 30)):
            return SessionPhase("LO_expansion", "London", weight=0.15, ttl_minutes=90, runner_bias=True)

        # 4) London->NY overlap (broader momentum window)
        if self._in_range(lt, time(12, 0), time(16, 0)):
            return SessionPhase("Overlap", "London", weight=0.10, ttl_minutes=60, runner_bias=True)

        # 5) NY cash open influence
        if self._in_range(nyt, time(9, 30), time(10, 0)):
            return SessionPhase("NY_cash_open", "NewYork", weight=0.05, ttl_minutes=30, runner_bias=False)

        # 6) Lunch lull (NY)
        if self._in_range(nyt, time(11, 30), time(13, 30)):
            return SessionPhase("Lunch", "NewYork", weight=-0.05, ttl_minutes=30, runner_bias=False)

        # 7) London close behavior
        if self._in_range(lt, time(16, 0), time(16, 30)):
            return SessionPhase("London_close", "London", weight=-0.05, ttl_minutes=30, runner_bias=False)

        # 8) Asia build (box formation)
        if self._in_range(lt, time(23, 0), time(8, 0)):
            return SessionPhase("Asia", "Asia", weight=0.05, ttl_minutes=45, runner_bias=False)

        # 9) NY pre-data compression window
        if self._in_range(nyt, time(8, 0), time(8, 30)):
            return SessionPhase("NY_predata", "NewYork", weight=0.05, ttl_minutes=30, runner_bias=False)

        # Default
        return SessionPhase("Other", "Other", weight=0.0, ttl_minutes=60, runner_bias=False)
