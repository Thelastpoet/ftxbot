import json
from datetime import datetime, timedelta
from mt5_client import MetaTrader5Client

def fetch_and_save_deals():
    """
    Fetches closed trades from the last 3 days from a MetaTrader 5 account
    and saves them into a JSON file.
    """
    mt5 = MetaTrader5Client()

    if not mt5.is_connected():
        print("Failed to connect to MetaTrader 5.")
        return

    # Calculate date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=3)

    # Fetch deals
    deals = mt5.get_history_deals(from_date, to_date)

    if not deals:
        print("No deals found in the last 3 days.")
        return

    # Convert deals to a list of dictionaries
    deals_list = []
    for deal in deals:
        deal_dict = deal._asdict()
        # Convert timestamp to datetime string
        deal_dict['time'] = datetime.fromtimestamp(deal_dict['time']).strftime('%Y-%m-%d %H:%M:%S')
        deals_list.append(deal_dict)

    # Save to JSON file
    file_path = "closed_trades.json"
    try:
        with open(file_path, "w") as f:
            json.dump(deals_list, f, indent=4)
        print(f"Successfully saved {len(deals_list)} deals to {file_path}")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")

if __name__ == "__main__":
    fetch_and_save_deals()
