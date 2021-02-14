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



def get_info_dataframe_for_kline(kline):
    # dataframe with features - X or model input
    pass

def get_result_dataframe_for_kline(kline):
    # dataframe with predictions - Y or model output
    pass

def merge_info_and_result_into_row(df_info, df_result):
    pass

def main():
    pass

class Command(BaseCommand):
    help = "Sync klines"

    def add_arguments(self, parser):
        parser.add_argument("--period", type=int, help="value 1 is for one hour, value 2 is for one day")

    def handle(self, *args, **kwargs):
        start_time = timezone.now()
        logger.info("Started at %s" % start_time.strftime("%X"))
        # main(kwargs["period"])
        logger.info("End time is %s" % timezone.now().strftime("%X"))
        logger.info("Duration %s seconds" % (timezone.now() - start_time).total_seconds())
