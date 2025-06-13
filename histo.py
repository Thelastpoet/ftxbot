import MetaTrader5 as mt5
import pandas as pd
import json
from datetime import datetime, timedelta

# Initialize MT5
if not mt5.initialize():
    print("initialize() failed", mt5.last_error())
    quit()

# Time range: last 24 hours
to_date = datetime.now()
from_date = to_date - timedelta(days=1)

# Fetch deals
deals = mt5.history_deals_get(from_date, to_date)

if deals is None or len(deals) == 0:
    print("No deals found.")
    mt5.shutdown()
    quit()

# Convert to list of dicts using _asdict()
deals_dicts = [deal._asdict() for deal in deals]
df = pd.DataFrame(deals_dicts)

# Handle time columns
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'], unit='s')
elif 'time_msc' in df.columns:
    df['time'] = pd.to_datetime(df['time_msc'], unit='ms')

# Save to files
df.to_csv("order_history_24h.csv", index=False)
df.to_json("order_history_24h.json", orient='records', date_format='iso')
print("Saved to CSV and JSON.")

# Shutdown MT5
mt5.shutdown()
