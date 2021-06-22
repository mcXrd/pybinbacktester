import django
import pandas as pd
from typing import List
import numpy as np
from django.conf import settings
from datetime import datetime
from apps.market_data.models import Kline
import logging

logger = logging.getLogger(__name__)
from django.utils.timezone import get_current_timezone
import dateutil.parser
import os


def command_datetime(s: str):
    return get_current_timezone().localize(dateutil.parser.parse(s))


def validate_pairs(pairs):
    """
    checking if symbol is synced
    """
    for pair in exchangify_pairs(pairs):
        if not Kline.objects.exists(symbol=pair):
            raise Exception("Pair {} is not synced".format(pair))


def save_output(df, filename) -> None:
    file_path = os.path.join(settings.HDF_STORAGE_PATH, filename)
    df.to_hdf(file_path, "df")
    logger.info(f"Dataframe saved, file path: {file_path}")


def fetch_input(
    time_interval: List[datetime], pairs: List[str]
) -> django.db.models.query.QuerySet:
    kwargs = {"symbol__in": exchangify_pairs(pairs)}
    if time_interval is None:
        return Kline.objects.filter(**kwargs)
    qs = Kline.objects.filter(
        close_time__gte=time_interval[0], close_time__lte=time_interval[1], **kwargs
    )
    return qs


def exchangify_pairs(pairs):
    ret = []
    for pair in pairs:
        ret.append(settings.EXCHANGE_FUTURES + "_" + pair)
        # ret.append(settings.EXCHANGE_SPOT + "_" + pair)
    return ret


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


def merge_symbols_and_kline_attrs(input_queryset, df, kline_attrs):
    symbols = get_symbols_from_qs(input_queryset)
    symbols_kline_attrs = []
    for symbol in symbols:
        for kline_attr in kline_attrs:
            feature_name = symbol + "_" + kline_attr
            symbols_kline_attrs.append(feature_name)
            df[feature_name] = create_series_from_qs(input_queryset, symbol, kline_attr)
    return symbols_kline_attrs


def create_base_dataframe(
    input_queryset: django.db.models.query.QuerySet,
    kline_attrs=None,
    skip_symbols_kline_attrs=False,
) -> pd.DataFrame:
    df = pd.DataFrame()
    kline_attrs = kline_attrs or [
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
    if not skip_symbols_kline_attrs:
        merge_symbols_and_kline_attrs(input_queryset, df, kline_attrs)

    return df


def add_hyperfeatures_to_df(df):
    original_columns = list(df.columns)

    df["daytime"] = df.index.hour / 24

    for c in original_columns:
        df["{}_ewm_mean".format(c)] = df[c].ewm(com=10).mean()
        df["{}_ewm_std".format(c)] = df[c].ewm(com=10).std()
        df["{}_ewm_cov".format(c)] = df[c].ewm(com=10).cov()
        windows = [10, 120, 240, 480, 1500, 4500]
        # assert max(windows) < settings.LARGEST_DF_WINDOW
        for base_size in windows:
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

    return df


def get_endswithc_column(df, endswith):
    original_columns = list(df.columns)
    close_price_columns = [
        x for x in original_columns if x.endswith(endswith) and "trade_in" not in x
    ]
    return close_price_columns


def get_close_price_columns(df):
    return get_endswithc_column(df, "_close_price")


def get_volume_columns(df):
    return get_endswithc_column(df, "_volume")


def get_number_of_trades_columns(df):
    return get_endswithc_column(df, "_number_of_trades")


def add_forecasts_to_df(df, live, shift=60):
    close_price_columns = get_close_price_columns(df)
    for c in close_price_columns:
        if "spot" in c:
            continue

        trade_in_1h = "trade_in_1h_{}".format(c)
        df.insert(0, trade_in_1h, np.ones(df.shape[0]))
        df[trade_in_1h] = df.shift(-1 * shift)[c] / df[c] - 1

    if live:
        return df
    return df[: -1 * shift]  # the latest one is incomplete


def create_dataframe(
    input_queryset: django.db.models.query.QuerySet, live=False
) -> pd.DataFrame:
    kline_attrs = ["close_price", "volume"]
    df = create_base_dataframe(input_queryset, kline_attrs=kline_attrs)
    base_columns = df.columns
    df = add_hyperfeatures_to_df(df)
    df = add_forecasts_to_df(df, live=live)
    df = df.drop(columns=base_columns)
    return df


def create_dataframe_v2(
    input_queryset: django.db.models.query.QuerySet,
    live: bool = False,
    minutes: int = 180,
) -> pd.DataFrame:
    kline_attrs = ["close_price", "volume"]
    df = create_base_dataframe(input_queryset, kline_attrs=kline_attrs)
    symbols_kline_attrs = merge_symbols_and_kline_attrs(input_queryset, df, kline_attrs)
    close_price_columns = get_close_price_columns(df)
    volume_columns = get_volume_columns(df)
    for c in close_price_columns + volume_columns:
        if "spot" in c:
            continue
        for one in range(minutes):
            if one == 0:
                continue
            df["{}_{}_ago".format(c, one)] = df.shift(one)[c] / df[c] - 1
    df = df.replace([np.inf, -np.inf], 0)
    if live:
        return df[minutes:]
    return df[minutes:-60]


def clean_initial_window_nans(df, rows_to_clean=None):
    return df[rows_to_clean or settings.LARGEST_DF_WINDOW :]


def stationarify_column(df, column_name):
    df = df.copy()
    df[column_name] = df[column_name] / df.shift(1)[column_name] - 1
    return df
