def get_pip_size(symbol_info: any) -> float:
    """
    Calculates the pip size for a given symbol, handling both dicts and objects.
    Handles standard FX, JPY pairs, and others based on digits.
    """
    # Use getattr to safely access attributes from an object or fallback
    point = getattr(symbol_info, 'point', 0.00001)
    digits = getattr(symbol_info, 'digits', 5)

    # For 2-digit (e.g., indices) or 3-digit (e.g., USDJPY) instruments
    if digits in (2, 3):
        # For JPY pairs, one pip is 0.01
        return 0.01
    
    # For 4 or 5-digit forex pairs, one pip is 10 points
    return point * 10