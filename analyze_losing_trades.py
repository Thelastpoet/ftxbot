import json
from collections import defaultdict
from datetime import datetime

def analyze_losing_trades():
    """
    Analyzes the losing trades in closed_trades.json to identify common patterns.
    """
    try:
        with open("closed_trades.json", "r") as f:
            deals = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing closed_trades.json: {e}")
        return

    if not deals:
        print("No trades to analyze.")
        return

    losing_trades = [d for d in deals if d.get('entry') == 1 and d.get('profit', 0) < 0]

    if not losing_trades:
        print("No losing trades found.")
        return

    print(f"--- Analysis of {len(losing_trades)} Losing Trades ---")

    # --- Analysis by Symbol ---
    losses_by_symbol = defaultdict(int)
    for deal in losing_trades:
        losses_by_symbol[deal['symbol']] += 1
    
    print("\n--- Losses by Symbol ---")
    for symbol, count in sorted(losses_by_symbol.items(), key=lambda item: item[1], reverse=True):
        print(f"{symbol}: {count} losing trades")

    # --- Analysis by Time of Day ---
    losses_by_hour = defaultdict(int)
    for deal in losing_trades:
        hour = datetime.strptime(deal['time'], '%Y-%m-%d %H:%M:%S').hour
        losses_by_hour[hour] += 1

    print("\n--- Losses by Hour of Day (UTC) ---")
    for hour, count in sorted(losses_by_hour.items()):
        print(f"{hour:02d}:00 - {hour:02d}:59 : {count} losing trades")

    # --- Analysis by Trade Direction ---
    losses_by_direction = defaultdict(int)
    for deal in losing_trades:
        # For a closing deal (entry=1), the 'type' is the closing order type.
        # type=0 (DEAL_TYPE_BUY) means a sell position was closed.
        # type=1 (DEAL_TYPE_SELL) means a buy position was closed.
        # So, if type is 1, the original trade was a BUY.
        direction = "Buy" if deal['type'] == 1 else "Sell"
        losses_by_direction[direction] += 1
    
    print("\n--- Losses by Trade Direction ---")
    for direction, count in losses_by_direction.items():
        print(f"{direction}: {count} losing trades")

    # --- Analysis by Reason for Closing ---
    reason_map = {
        0: "Manual",
        3: "Expert Advisor",
        4: "Stop Loss",
        5: "Take Profit"
    }
    losses_by_reason = defaultdict(int)
    for deal in losing_trades:
        reason = reason_map.get(deal['reason'], f"Unknown ({deal['reason']})")
        losses_by_reason[reason] += 1

    print("\n--- Losses by Reason for Closing ---")
    for reason, count in losses_by_reason.items():
        print(f"{reason}: {count} losing trades")
        
    # --- Analysis by Comment ---
    losses_by_comment = defaultdict(int)
    for deal in losing_trades:
        comment = deal.get('comment', 'No comment')
        if not comment:
            comment = 'No comment'
        losses_by_comment[comment] += 1
        
    print("\n--- Losses by Comment ---")
    for comment, count in sorted(losses_by_comment.items(), key=lambda item: item[1], reverse=True):
        print(f"'{comment}': {count} losing trades")


if __name__ == "__main__":
    analyze_losing_trades()
