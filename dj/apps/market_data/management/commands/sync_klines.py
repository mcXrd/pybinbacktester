import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import List

import pytz
from binance.client import Client as BinanceClient
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from apps.market_data.models import Kline

logger = logging.getLogger(__name__)

binance_client = BinanceClient(settings.BINANCE_API_KEY, settings.BINANCE_SECRET_KEY)


def binance_timestamp_to_utc_datetime(binance_timestamp: str) -> datetime:
    return datetime.utcfromtimestamp(binance_timestamp / 1000).replace(tzinfo=pytz.utc)


def get_klines(symbol: str, binance_timestamp: str) -> List[List]:
    return binance_client.get_historical_klines(symbol, BinanceClient.KLINE_INTERVAL_1MINUTE, binance_timestamp)


# 1 day period is meant for initializaton and 1 hour for updates
# note that updates can be run anytime and should be run once per kline
# interval eg. BinanceClient.KLINE_INTERVAL_5MINUTE
PERIODS = {1: "1 hour ago UTC", 2: "1 day ago UTC"}
PERIOD_CPU_COUNT_MAP = {1: os.cpu_count() * 8, 2: os.cpu_count() * 4}


def insert_klines(symbol: str, binance_timestamp: str):
    klines = get_klines(symbol, binance_timestamp)
    latest_kline = Kline.objects.order_by('open_time').filter(symbol=symbol).last()
    bulk_klines = []
    for kline in klines:
        open_time = binance_timestamp_to_utc_datetime(kline[0])
        if latest_kline and open_time <= latest_kline.open_time:
            continue

        close_time = binance_timestamp_to_utc_datetime(kline[6])
        kline_fields = {'open_time': open_time,
                        'open_price': kline[1],
                        'high_price': kline[2],
                        'low_price': kline[3],
                        'close_price': kline[4],
                        'volume': kline[5],
                        'close_time': close_time,
                        'quota_asset_volume': kline[7],
                        'number_of_trades': kline[8],
                        'taker_buy_base_asset_volume': kline[9],
                        'taker_buy_quote_asset_volume': kline[10],
                        'ignore': kline[11]}
        kline_instance = Kline(symbol=symbol, **kline_fields)
        bulk_klines.append(kline_instance)
    Kline.objects.bulk_create(bulk_klines)
    logger.debug(f'Symbol {symbol} synced.')


def remove_too_old_klines(days=1):
    Kline.objects.filter(open_time__lte=timezone.now() - timedelta(days=days)).delete()


def main(period: int):
    tickers = binance_client.get_all_tickers()
    ticker_symbols = {x['symbol'] for x in tickers}

    with ProcessPoolExecutor(max_workers=PERIOD_CPU_COUNT_MAP[period]) as executor:
        for ticker_symbol in ticker_symbols:
            executor.submit(insert_klines, ticker_symbol, PERIODS[period])


    remove_too_old_klines()


class Command(BaseCommand):
    help = 'Sync klines'

    def add_arguments(self, parser):
        parser.add_argument('--period', type=int, help='value 1 is for one hour, value 2 is for one day')

    def handle(self, *args, **kwargs):
        start_time = timezone.now()
        logger.info("Started at %s" % start_time.strftime('%X'))
        main(kwargs['period'])
        logger.info("End time is %s" % timezone.now().strftime('%X'))
        logger.info("Duration %s seconds" % (timezone.now() - start_time).total_seconds())
