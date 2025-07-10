import MetaTrader5 as mt5

symbol = "EURUSD"
if not mt5.initialize():
    print("MT5 init failed")
else:
    order_book = mt5.market_book_get(symbol)
    if order_book is None:
        print("Order book not available for", symbol)
    else:
        for entry in order_book:
            print(entry)
    mt5.shutdown()
