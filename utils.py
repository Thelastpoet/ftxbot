def get_pip_size(symbol_info: any) -> float:
    """
    Backward-compatible pip size heuristic.

    Note: Prefer resolve_pip_size(symbol: str, symbol_info, config) when possible
    to allow per-symbol overrides and more accurate handling for metals/others.
    """
    point = getattr(symbol_info, 'point', 0.00001)
    digits = getattr(symbol_info, 'digits', 5)

    # Common FX quoting: 5 or 3 digits => pip = 10 points; 4 or 2 digits => pip = 1 point
    if digits in (5, 3):
        return point * 10.0
    if digits in (4, 2):
        return point * 1.0
    # Fallback
    return point * 10.0


def resolve_pip_size(symbol: str, symbol_info: any, config: any) -> float:
    """
    Resolve pip size for a symbol with optional per-symbol override from config.
    - If config.symbols contains an entry with name == symbol and pip_unit set, use that.
    - Else, derive from digits/point using the common FX rules.
    """
    try:
        for s in getattr(config, 'symbols', []) or []:
            if s.get('name') == symbol:
                pip_unit = s.get('pip_unit')
                if pip_unit is not None:
                    return float(pip_unit)
                break
    except Exception:
        pass
    return get_pip_size(symbol_info)
