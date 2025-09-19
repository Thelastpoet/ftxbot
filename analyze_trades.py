import json
from collections import defaultdict

def analyze_trades_detailed_md():
    """
    Analyzes the trades in closed_trades.json and saves a detailed performance summary in Markdown format.
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

    closed_trades = [d for d in deals if d.get('entry') == 1]
    
    # --- Overall Analysis ---
    total_trades = len(closed_trades)
    winning_trades = 0
    losing_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for deal in closed_trades:
        profit = deal.get('profit', 0.0)
        if profit > 0:
            winning_trades += 1
            gross_profit += profit
        elif profit < 0:
            losing_trades += 1
            gross_loss += abs(profit)

    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_winning_trade = gross_profit / winning_trades if winning_trades > 0 else 0
    avg_losing_trade = gross_loss / losing_trades if losing_trades > 0 else 0
    net_profit = gross_profit - gross_loss

    md_content = f"""# Trade Performance Analysis

## Overall Performance

| Metric | Value |
|---|---|
| Total Closed Trades | {total_trades} |
| Winning Trades | {winning_trades} |
| Losing Trades | {losing_trades} |
| Win Rate | {win_rate:.2f}% |
| Gross Profit | ${gross_profit:.2f} |
| Gross Loss | ${gross_loss:.2f} |
| Net Profit | ${net_profit:.2f} |
| Profit Factor | {profit_factor:.2f} |
| Average Winning Trade | ${avg_winning_trade:.2f} |
| Average Losing Trade | ${avg_losing_trade:.2f} |

"""

    # --- Analysis by Symbol ---
    trades_by_symbol = defaultdict(list)
    for deal in closed_trades:
        trades_by_symbol[deal['symbol']].append(deal)

    md_content += "## Performance by Currency Pair\n\n"

    for symbol, symbol_deals in trades_by_symbol.items():
        total_trades_symbol = len(symbol_deals)
        winning_trades_symbol = 0
        losing_trades_symbol = 0
        gross_profit_symbol = 0.0
        gross_loss_symbol = 0.0

        for deal in symbol_deals:
            profit = deal.get('profit', 0.0)
            if profit > 0:
                winning_trades_symbol += 1
                gross_profit_symbol += profit
            elif profit < 0:
                losing_trades_symbol += 1
                gross_loss_symbol += abs(profit)

        win_rate_symbol = (winning_trades_symbol / total_trades_symbol) * 100 if total_trades_symbol > 0 else 0
        profit_factor_symbol = gross_profit_symbol / gross_loss_symbol if gross_loss_symbol > 0 else float('inf')
        avg_winning_trade_symbol = gross_profit_symbol / winning_trades_symbol if winning_trades_symbol > 0 else 0
        avg_losing_trade_symbol = gross_loss_symbol / losing_trades_symbol if losing_trades_symbol > 0 else 0
        net_profit_symbol = gross_profit_symbol - gross_loss_symbol

        md_content += f"""### Analysis for {symbol}

| Metric | Value |
|---|---|
| Total Closed Trades | {total_trades_symbol} |
| Winning Trades | {winning_trades_symbol} |
| Losing Trades | {losing_trades_symbol} |
| Win Rate | {win_rate_symbol:.2f}% |
| Gross Profit | ${gross_profit_symbol:.2f} |
| Gross Loss | ${gross_loss_symbol:.2f} |
| Net Profit | ${net_profit_symbol:.2f} |
| Profit Factor | {profit_factor_symbol:.2f} |
| Average Winning Trade | ${avg_winning_trade_symbol:.2f} |
| Average Losing Trade | ${avg_losing_trade_symbol:.2f} |

"""

    try:
        with open("trade_analysis.md", "w") as f:
            f.write(md_content)
        print("Successfully saved analysis to trade_analysis.md")
    except IOError as e:
        print(f"Error writing to file trade_analysis.md: {e}")


if __name__ == "__main__":
    analyze_trades_detailed_md()