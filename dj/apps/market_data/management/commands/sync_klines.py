import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import List, Callable
from binance_f import RequestClient

import pytz
from binance.client import Client as BinanceClient
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone
from binance_f.model import CandlestickInterval
from apps.market_data.usdtfutures_utils import get_usdtfutures_historical_klines
from requests.exceptions import ConnectionError
import time

from apps.market_data.models import Kline

logger = logging.getLogger(__name__)

# TIME_INTERVAL = "60 days ago UTC"
TIME_INTERVAL = ["5 Jul, 2020", "25 Dec, 2020"]
#TIME_INTERVAL = ["Feb Apr, 2020", "25 Aug, 2020"]
TIME_INTERVAL = ["2 Feb, 2021", "8 May, 2021"]
EXCHANGE_FUTURES = settings.EXCHANGE_FUTURES
EXCHANGE_SPOT = settings.EXCHANGE_SPOT


def binance_timestamp_to_utc_datetime(binance_timestamp: str) -> datetime:
    return datetime.utcfromtimestamp(binance_timestamp / 1000).replace(tzinfo=pytz.utc)


def get_spot_klines(symbol: str, time_interval: str) -> List[List]:
    binance_client = BinanceClient(
        settings.BINANCE_API_KEY, settings.BINANCE_SECRET_KEY
    )
    interval = BinanceClient.KLINE_INTERVAL_1HOUR
    interval = BinanceClient.KLINE_INTERVAL_1MINUTE
    return binance_client.get_historical_klines(symbol, interval, *time_interval)


def get_usdt_futures_klines(symbol: str, time_interval: str) -> List[List]:
    request_client = RequestClient(
        api_key=settings.BINANCE_API_KEY, secret_key=settings.BINANCE_SECRET_KEY
    )
    get_klines: Callable = request_client.get_candlestick_data
    interval = CandlestickInterval.HOUR1
    interval = CandlestickInterval.MIN1
    return get_usdtfutures_historical_klines(
        get_klines, symbol, interval, *time_interval
    )


def insert_klines(symbol: str, get_klines: Callable, exchange: str, time_interval: str):
    for one in range(4):
        try:
            klines = get_klines(symbol, time_interval)
            break
        except ConnectionError:
            time.sleep(70)

    bulk_klines = []
    complete_symbol = exchange + "_" + symbol

    for kline in klines:
        open_time = binance_timestamp_to_utc_datetime(kline[0])
        close_time = binance_timestamp_to_utc_datetime(kline[6])

        if Kline.objects.filter(symbol=complete_symbol, close_time=close_time).exists():
            continue

        kline_fields = {
            "open_time": open_time,
            "open_price": kline[1],
            "high_price": kline[2],
            "low_price": kline[3],
            "close_price": kline[4],
            "volume": kline[5],
            "close_time": close_time,
            "quota_asset_volume": kline[7],
            "number_of_trades": kline[8],
            "taker_buy_base_asset_volume": kline[9],
            "taker_buy_quote_asset_volume": kline[10],
            "ignore": kline[11],
        }
        symbol_name = exchange + "_" + symbol
        kline_instance = Kline(symbol=symbol_name, **kline_fields)
        bulk_klines.append(kline_instance)
    Kline.objects.bulk_create(bulk_klines)
    logger.debug(f"Symbol {symbol} synced.")


def remove_too_old_klines(days=1):
    Kline.objects.filter(close_time__lte=timezone.now() - timedelta(days=days)).delete()


def main(max_workers=1, time_interval=None, coin=None):
    binance_client = BinanceClient(
        settings.BINANCE_API_KEY, settings.BINANCE_SECRET_KEY
    )
    if coin is None:
        key_currencies = settings.USDT_FUTURES_PAIRS
    else:
        key_currencies = [coin]
    tickers = binance_client.get_all_tickers()
    ticker_symbols = {x["symbol"] for x in tickers}
    for c in key_currencies:
        assert c in ticker_symbols

    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for ticker_symbol in key_currencies:
            futures[EXCHANGE_SPOT + ticker_symbol] = executor.submit(
                insert_klines,
                ticker_symbol,
                get_spot_klines,
                EXCHANGE_SPOT,
                time_interval,
            )
            futures[EXCHANGE_FUTURES + ticker_symbol] = executor.submit(
                insert_klines,
                ticker_symbol,
                get_usdt_futures_klines,
                EXCHANGE_FUTURES,
                time_interval,
            )

    for ticker_symbol in futures:
        futures[ticker_symbol].result()

    # remove_too_old_klines()


class Command(BaseCommand):
    help = "Sync klines"

    def add_arguments(self, parser):
        parser.add_argument("--max-workers", type=int, required=False)
        parser.add_argument(
            "--time-interval", type=str, required=False, default=TIME_INTERVAL
        )
        parser.add_argument("--coin", type=str, required=False)

    def handle(self, *args, **kwargs):
        start_time = timezone.now()
        logger.info("Started at %s" % start_time.strftime("%X"))
        main(kwargs["max_workers"], kwargs["time_interval"], kwargs["coin"])
        logger.info("End time is %s" % timezone.now().strftime("%X"))
        logger.info(
            "Duration %s seconds" % (timezone.now() - start_time).total_seconds()
        )
