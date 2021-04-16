import django
import pandas as pd
from typing import List
import numpy as np
from django.conf import settings


def exchangify_pairs(pairs):
    ret = []
    for pair in pairs:
        ret.append(settings.EXCHANGE_FUTURES + "_" + pair)
        ret.append(settings.EXCHANGE_SPOT + "_" + pair)
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


def create_base_dataframe(
    input_queryset: django.db.models.query.QuerySet,
) -> pd.DataFrame:
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


def add_hyperfeatures_to_df(df):
    original_columns = list(df.columns)

    for c in original_columns:
        df["{}_ewm_mean".format(c)] = df[c].ewm(com=0.5).mean()
        df["{}_ewm_std".format(c)] = df[c].ewm(com=0.5).std()
        df["{}_ewm_cov".format(c)] = df[c].ewm(com=0.5).cov()
        windows = [2, 8, 32, 128]
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

    return df[300:]


def add_forecasts_to_df(df):
    original_columns = list(df.columns)
    close_price_columns = [x for x in original_columns if x.endswith("_close_price")]

    for c in close_price_columns:
        if "spot" in c:
            continue

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


def create_dataframe(input_queryset: django.db.models.query.QuerySet) -> pd.DataFrame:
    df = create_base_dataframe(input_queryset)
    return add_forecasts_to_df(add_hyperfeatures_to_df(df))
