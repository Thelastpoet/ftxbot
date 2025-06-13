import os
import requests
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup API key and endpoints from .env
API_KEY = os.getenv("JBLANKED_API")
ENDPOINTS = {
    "forex_factory_today": os.getenv("FR_ENDPOINT_TODAY"),
    "forex_factory_week": os.getenv("FR_ENDPOINT_WEEK"),
    "mql5_today": os.getenv("MQL5_ENDPOINT_TODAY"),
    "mql5_week": os.getenv("MQL5_ENDPOINT_WEEK"),
}

# HTTP headers with authentication
HEADERS = {
    "Authorization": f"Api-Key {API_KEY}"
}

# Fetch data from each endpoint
def fetch_data(name, url):
    try:
        print(f"Fetching: {name}")
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching {name}: {e}")
        return []

# Normalize all fields to a consistent structure
def normalize(events, source):
    normalized = []
    for event in events:
        normalized.append({
            "source": source,
            "name": event.get("Name"),
            "currency": event.get("Currency"),
            "category": event.get("Category"),
            "date": event.get("Date"),
            "actual": event.get("Actual"),
            "forecast": event.get("Forecast"),
            "previous": event.get("Previous"),
            "outcome": event.get("Outcome"),
            "strength": event.get("Strength"),
            "quality": event.get("Quality"),
            "projection": event.get("Projection")
        })
    return normalized

# Main runner
def main():
    all_events = []
    for name, url in ENDPOINTS.items():
        raw_data = fetch_data(name, url)
        events = normalize(raw_data, name)
        all_events.extend(events)

    filename = f"calendar_events_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    with open(filename, "w") as f:
        json.dump(all_events, f, indent=2)
    print(f"✅ Saved {len(all_events)} events to {filename}")

# Run the script
if __name__ == "__main__":
    main()
