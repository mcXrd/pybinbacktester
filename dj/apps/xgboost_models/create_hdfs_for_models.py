from apps.market_data.generate_market_data_hdf_utils import (
    create_dataframe_v2,
    save_output,
    fetch_input,
    command_datetime,
    create_base_dataframe,
    merge_symbols_and_kline_attrs,
    add_forecasts_to_df,
    get_close_price_columns,
    stationarify_column,
    get_volume_columns,
    get_number_of_trades_columns,
)
from apps.market_data.technical_indicators_utils import SMA
import numpy as np
from django.utils import timezone
from datetime import timedelta

kline_attrs = ["close_price", "volume", "number_of_trades"]


def A_transform(df, symbols_kline_attrs):
    output_df_v2_lagged = SMA(df, 5, symbols_kline_attrs, 0)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 30, symbols_kline_attrs, 1)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 60, symbols_kline_attrs, 1)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 90, symbols_kline_attrs, 1)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 120, symbols_kline_attrs, 1)
    output_df_v2_lagged = output_df_v2_lagged[270:]
    return output_df_v2_lagged


def B_transform(df, symbols_kline_attrs):
    output_df_v2_lagged = SMA(df, 5, symbols_kline_attrs, 0)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 30, symbols_kline_attrs, 1)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 60, symbols_kline_attrs, 1)
    output_df_v2_lagged = output_df_v2_lagged[270:]
    return output_df_v2_lagged


def C_transform(df, symbols_kline_attrs):
    output_df_v2_lagged = SMA(df, 5, symbols_kline_attrs, 0)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 60, symbols_kline_attrs, 0)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 120, symbols_kline_attrs, 0)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 240, symbols_kline_attrs, 0)
    output_df_v2_lagged = output_df_v2_lagged[270:]
    return output_df_v2_lagged


def create_base_hdf(coin, days):
    start = timezone.now()
    end = timezone.now() - timedelta(days=days)
    qs = fetch_input((start, end), coin)
    df = create_base_dataframe(qs, kline_attrs=kline_attrs)
    symbols_kline_attrs = merge_symbols_and_kline_attrs(qs, df, kline_attrs)
    df = add_forecasts_to_df(df, live=False)
    for column_name in (
        get_close_price_columns(df)
        + get_volume_columns(df)
        + get_number_of_trades_columns(df)
    ):
        df = stationarify_column(df, column_name)
    df = df.replace([np.inf, -np.inf], 0)
    return df, symbols_kline_attrs


def create_A_hdf():
    df, symbols_kline_attrs = create_base_hdf("ETHUSDT", 11)
    df = A_transform(df.copy(), symbols_kline_attrs)
    return df, "ETH"


def create_B_hdf():
    df, symbols_kline_attrs = create_base_hdf("ETHUSDT", 11)
    df = B_transform(df.copy(), symbols_kline_attrs)
    return df, "ETH"


def create_C_hdf():
    df, symbols_kline_attrs = create_base_hdf("ETHUSDT", 11)
    df = C_transform(df.copy(), symbols_kline_attrs)
    return df, "ETH"


def create_D_hdf():
    df, symbols_kline_attrs = create_base_hdf("ETHUSDT", 5)
    df = A_transform(df.copy(), symbols_kline_attrs)
    return df, "ETH"


def create_E_hdf():
    df, symbols_kline_attrs = create_base_hdf("ETHUSDT", 5)
    df = B_transform(df.copy(), symbols_kline_attrs)
    return df, "ETH"


def create_F_hdf():
    df, symbols_kline_attrs = create_base_hdf("ETHUSDT", 5)
    df = C_transform(df.copy(), symbols_kline_attrs)
    return df, "ETH"


def create_A2_hdf():
    df, symbols_kline_attrs = create_base_hdf("ADAUSDT", 11)
    df = A_transform(df.copy(), symbols_kline_attrs)
    return df, "ADA"


def create_B2_hdf():
    df, symbols_kline_attrs = create_base_hdf("ADAUSDT", 11)
    df = B_transform(df.copy(), symbols_kline_attrs)
    return df, "ADA"


def create_C2_hdf():
    df, symbols_kline_attrs = create_base_hdf("ADAUSDT", 11)
    df = C_transform(df.copy(), symbols_kline_attrs)
    return df, "ADA"


def create_D2_hdf():
    df, symbols_kline_attrs = create_base_hdf("ADAUSDT", 5)
    df = A_transform(df.copy(), symbols_kline_attrs)
    return df, "ADA"


def create_E2_hdf():
    df, symbols_kline_attrs = create_base_hdf("ADAUSDT", 5)
    df = B_transform(df.copy(), symbols_kline_attrs)
    return df, "ADA"


def create_F2_hdf():
    df, symbols_kline_attrs = create_base_hdf("ADAUSDT", 5)
    df = C_transform(df.copy(), symbols_kline_attrs)
    return df, "ADA"
