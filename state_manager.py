"""
State persistence manager for live trading bot.

Provides fast, crash-safe JSON persistence for critical in-memory state.
Uses atomic write (temp file + rename) to prevent corruption on crash.

Performance: < 1ms for small state files (<1KB)
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages persistence of critical bot state to JSON files.

    Features:
    - Atomic writes (crash-safe)
    - Fast read/write (<1ms)
    - Human-readable JSON
    - Minimal file I/O (write on change only)
    """

    def __init__(self, state_dir: str = "."):
        """
        Initialize state manager.

        Args:
            state_dir: Directory to store state files (default: current dir)
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _get_state_path(self, component: str) -> Path:
        """Get path for component state file."""
        return self.state_dir / f".{component}_state.json"

    def save_state(self, component: str, state: Dict[str, Any]) -> bool:
        """
        Save state to JSON file atomically.

        Uses temp file + rename pattern for crash safety.

        Args:
            component: Component name (e.g., 'amd', 'strategy')
            state: State dictionary to persist

        Returns:
            True if successful, False otherwise
        """
        try:
            state_path = self._get_state_path(component)
            temp_path = state_path.with_suffix('.tmp')

            # Serialize with custom encoder for datetime/date
            def json_encoder(obj):
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} not serializable")

            # Write to temp file first
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, default=json_encoder, indent=2)

            # Atomic rename (crash-safe on most filesystems)
            temp_path.replace(state_path)

            logger.debug(f"State saved: {component} ({len(state)} keys)")
            return True

        except Exception as e:
            logger.error(f"Failed to save state for {component}: {e}")
            return False

    def load_state(self, component: str) -> Optional[Dict[str, Any]]:
        """
        Load state from JSON file.

        Args:
            component: Component name (e.g., 'amd', 'strategy')

        Returns:
            State dictionary if exists, None otherwise
        """
        try:
            state_path = self._get_state_path(component)

            if not state_path.exists():
                logger.debug(f"No saved state for {component}")
                return None

            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)

            logger.info(f"State loaded: {component} ({len(state)} keys)")
            return state

        except Exception as e:
            logger.warning(f"Failed to load state for {component}: {e}")
            return None

"""Convenience functions for common use cases"""

def save_strategy_state(state_manager: StateManager, strategy) -> bool:
    """
    Save strategy state (duplicate prevention).

    Args:
        state_manager: StateManager instance
        strategy: PurePriceActionStrategy instance

    Returns:
        True if successful
    """
    try:
        # Convert _last_breakout_bar timestamps to ISO format
        last_breakout = {}
        for key, timestamp in strategy._last_breakout_bar.items():
            # key is tuple (symbol, type), need to convert to string
            key_str = f"{key[0]}_{key[1]}"
            last_breakout[key_str] = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)

        state = {
            'last_breakout_bar': last_breakout,
            'saved_at': datetime.now().isoformat()
        }
        return state_manager.save_state('strategy', state)
    except Exception as e:
        logger.error(f"Failed to save strategy state: {e}")
        return False


def load_strategy_state(state_manager: StateManager, strategy) -> bool:
    """
    Load strategy state (duplicate prevention).

    Args:
        state_manager: StateManager instance
        strategy: PurePriceActionStrategy instance

    Returns:
        True if state was loaded
    """
    try:
        state = state_manager.load_state('strategy')
        if not state:
            return False

        # Restore _last_breakout_bar
        last_breakout = state.get('last_breakout_bar', {})
        for key_str, timestamp_str in last_breakout.items():
            try:
                # Parse key string back to tuple
                parts = key_str.split('_', 1)
                if len(parts) == 2:
                    symbol, breakout_type = parts
                    key = (symbol, breakout_type)
                    # Parse timestamp
                    import pandas as pd
                    timestamp = pd.Timestamp(timestamp_str)
                    strategy._last_breakout_bar[key] = timestamp
            except Exception as e:
                logger.warning(f"Failed to parse breakout bar entry {key_str}: {e}")
                continue

        logger.info(f"Strategy: Restored {len(strategy._last_breakout_bar)} breakout bar entries")
        return True

    except Exception as e:
        logger.error(f"Failed to load strategy state: {e}")
        return False
