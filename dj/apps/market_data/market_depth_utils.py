from django.conf import settings
from binance.client import Client as BinanceClient
from datetime import datetime
from copy import copy

binance_client = BinanceClient(settings.BINANCE_API_KEY, settings.BINANCE_SECRET_KEY)


def list_symbol_pairs(symbol="ADA"):
    tickers = binance_client.get_all_tickers()
    ticker_symbols = {x["symbol"] for x in tickers if x["symbol"].startswith(symbol)}

    return list(ticker_symbols)


def transform_time(original_trade):
    time = original_trade["time"]
    dt_object = datetime.fromtimestamp(int(time) / 1000)
    dt_object = dt_object.replace(second=0)
    dt_object = dt_object.replace(microsecond=0)
    new_time = dt_object
    new_trade = copy(original_trade)
    new_trade["time"] = new_time
    return new_trade


def get_trades(symbol):
    original_trades = binance_client.get_historical_trades(symbol="ADAUSDT")
    new_trades = []
    for original_trade in original_trades:
        new_trade = transform_time(original_trade)
        new_trades.append(new_trade)
    return new_trades
