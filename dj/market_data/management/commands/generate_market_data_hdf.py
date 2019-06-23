import logging
import os
from datetime import datetime
from typing import List

import dateutil.parser
import django
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.timezone import get_current_timezone
from market_data.models import Kline

logger = logging.getLogger(__name__)


def get_symbols_from_qs(qs: django.db.models.query.QuerySet) -> List[str]:
    symbols = []
    for symbol in Kline.objects.values('symbol').distinct():
        symbols.append(symbol['symbol'])
    return symbols


def create_series_from_qs(qs: django.db.models.query.QuerySet, symbol: str) -> pd.Series:
    data = []
    index = []
    for kline in qs.filter(symbol=symbol):
        data.append(kline.close_price)
        index.append(kline.close_time)
    return pd.Series(data=data, index=index)


def fetch_input(time_interval: List[datetime]) -> django.db.models.query.QuerySet:
    qs = Kline.objects.filter(open_time__gte=time_interval[0], open_time__lte=time_interval[1])
    return qs


def save_output(df, filename) -> None:
    file_path = os.path.join(settings.HDF_STORAGE_PATH, filename)
    df.to_hdf(file_path, 'df')
    logger.info(f'Dataframe saved, file path: {file_path}')


def create_dataframe(input_queryset: django.db.models.query.QuerySet) -> pd.DataFrame:
    df = pd.DataFrame()
    symbols = get_symbols_from_qs(input_queryset)

    for symbol in symbols:
        df[symbol] = create_series_from_qs(input_queryset, symbol)

    return df


def command_datetime(s: str):
    return get_current_timezone().localize(dateutil.parser.parse(s))


class Command(BaseCommand):
    help = 'Sync klines'

    def add_arguments(self, parser):
        parser.add_argument('--time-interval', nargs='+', type=command_datetime, required=True,
                            help="'2002-12-25 00:00:00' '2019-01-1 00:00:00'")
        parser.add_argument('--hdf-filename', type=str, required=True)

    def handle(self, *args, **kwargs):
        qs = fetch_input(kwargs['time_interval'])
        output_df = create_dataframe(qs)
        save_output(output_df, kwargs['hdf_filename'])
