import logging
import os
from datetime import datetime
from typing import List

import dateutil.parser
import django
import pandas as pd
import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.timezone import get_current_timezone
from apps.market_data.models import Kline


logger = logging.getLogger(__name__)


def get_symbols_from_qs(qs: django.db.models.query.QuerySet) -> List[str]:
    symbols = []
    for symbol in qs.values("symbol").distinct():
        symbols.append(symbol["symbol"])
    return symbols


def create_series_from_qs(
    qs: django.db.models.query.QuerySet, symbol: str, kline_attr: str
) -> pd.Series:
    data = []
    index = []
    for kline in qs.filter(symbol=symbol):
        data.append(getattr(kline, kline_attr))
        index.append(kline.close_time)
    return pd.Series(data=data, index=index)


def fetch_input(time_interval: List[datetime]) -> django.db.models.query.QuerySet:
    if time_interval is None:
        return Kline.objects.all()
    qs = Kline.objects.filter(
        open_time__gte=time_interval[0], open_time__lte=time_interval[1]
    )
    return qs


def save_output(df, filename) -> None:
    file_path = os.path.join(settings.HDF_STORAGE_PATH, filename)
    df.to_hdf(file_path, "df")
    logger.info(f"Dataframe saved, file path: {file_path}")


def add_hyperfeatures_to_df(df):
    original_columns = list(df.columns)

    for c in original_columns:
        df["{}_ewm_mean".format(c)] = df[c].ewm(com=0.5).mean()
        df["{}_ewm_std".format(c)] = df[c].ewm(com=0.5).std()
        df["{}_ewm_cov".format(c)] = df[c].ewm(com=0.5).cov()
        for base_size in [2, 8, 32, 128]:
            df["{}_mean_{}".format(c, base_size)] = (
                df[c].rolling(window=base_size).mean()
            )
            df["{}_skew_{}".format(c, base_size)] = (
                df[c].rolling(window=base_size).skew()
            )
            df["{}_kurt_{}".format(c, base_size)] = (
                df[c].rolling(window=base_size).kurt()
            )
            df["{}_std_{}".format(c, base_size)] = df[c].rolling(window=base_size).std()
            df["{}_cov_{}".format(c, base_size)] = df[c].rolling(window=base_size).cov()

    return df[300:]


def add_forecasts_to_df(df):
    original_columns = list(df.columns)
    open_price_columns = [x for x in original_columns if x.endswith("_open_price")]

    for c in open_price_columns:
        trade_in_1h = "trade_in_1h_{}".format(c)
        trade_in_2h = "trade_in_2h_{}".format(c)
        trade_in_3h = "trade_in_3h_{}".format(c)
        df.insert(0, trade_in_1h, np.ones(df.shape[0]))
        df.insert(0, trade_in_2h, np.ones(df.shape[0]))
        df.insert(0, trade_in_3h, np.ones(df.shape[0]))
        df[trade_in_1h] = df.shift(-1)[c] / df[c] - 1
        df[trade_in_2h] = df.shift(-2)[c] / df[c] - 1
        df[trade_in_3h] = df.shift(-3)[c] / df[c] - 1
    return df[:-3]


def create_base_dataframe(input_queryset: django.db.models.query.QuerySet) -> pd.DataFrame:
    df = pd.DataFrame()
    symbols = get_symbols_from_qs(input_queryset)

    kline_attrs = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
        "quota_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]

    for symbol in symbols:
        for kline_attr in kline_attrs:
            feature_name = symbol + "_" + kline_attr
            df[feature_name] = create_series_from_qs(input_queryset, symbol, kline_attr)

    return df


def create_dataframe(input_queryset: django.db.models.query.QuerySet) -> pd.DataFrame:
    df = create_base_dataframe(input_queryset)
    return add_forecasts_to_df(add_hyperfeatures_to_df(df))


def command_datetime(s: str):
    return get_current_timezone().localize(dateutil.parser.parse(s))


class Command(BaseCommand):
    help = "Sync klines"

    def add_arguments(self, parser):
        parser.add_argument(
            "--time-interval",
            nargs="+",
            type=command_datetime,
            required=False,
            help="'2002-12-25 00:00:00' '2019-01-1 00:00:00'",
        )
        parser.add_argument("--hdf-filename", type=str, required=True)

    def handle(self, *args, **kwargs):
        qs = fetch_input(kwargs["time_interval"])
        output_df = create_dataframe(qs)
        save_output(output_df, kwargs["hdf_filename"])
