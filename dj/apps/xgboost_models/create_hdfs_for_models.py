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
import pandas as pd
import tqdm

kline_attrs = ["close_price", "volume", "number_of_trades"]


def A_transform(df, symbols_kline_attrs):
    output_df_v2_lagged = SMA(df, 5, symbols_kline_attrs, 0, 30)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 30, symbols_kline_attrs, 1, 30)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 60, symbols_kline_attrs, 1, 30)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 90, symbols_kline_attrs, 1, 30)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 120, symbols_kline_attrs, 1, 30)
    output_df_v2_lagged = output_df_v2_lagged[270:]
    return output_df_v2_lagged


def B_transform(df, symbols_kline_attrs):
    output_df_v2_lagged = SMA(df, 5, symbols_kline_attrs, 0, 30)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 30, symbols_kline_attrs, 1, 30)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 60, symbols_kline_attrs, 1, 30)
    output_df_v2_lagged = output_df_v2_lagged[270:]
    return output_df_v2_lagged


def C_transform(df, symbols_kline_attrs):
    output_df_v2_lagged = SMA(df, 5, symbols_kline_attrs, 0, 30)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 60, symbols_kline_attrs, 0, 30)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 120, symbols_kline_attrs, 0, 30)
    output_df_v2_lagged = SMA(output_df_v2_lagged, 240, symbols_kline_attrs, 0, 30)
    output_df_v2_lagged = output_df_v2_lagged[270:]
    return output_df_v2_lagged


class IncompleteDataframeException(Exception):
    pass


def create_base_hdf(coin, days, live=False):
    start = timezone.now() - timedelta(days=days)
    end = timezone.now()
    qs = fetch_input((start, end), [coin])
    df = create_base_dataframe(qs, kline_attrs=kline_attrs)
    symbols_kline_attrs = merge_symbols_and_kline_attrs(qs, df, kline_attrs)
    df = add_forecasts_to_df(df, live=live)
    for column_name in (
        get_close_price_columns(df)
        + get_volume_columns(df)
        + get_number_of_trades_columns(df)
    ):
        df = stationarify_column(df, column_name)
    df = df.replace([np.inf, -np.inf], 0)
    df = df.sort_index()

    for i in tqdm.tqdm(range(len(df) - 3)):
        t1 = pd.to_datetime(df[i : i + 1].index)
        t2 = pd.to_datetime(df[i + 1 : i + 2].index)
        if int((t2 - t1).total_seconds().values[0]) != 60:
            msg = "Dataframe is incompleted - t1 {} t2 {}".format(t1, t2)
            from apps.predictive_models.models import AlertLog

            AlertLog.objects.create(
                name="Incomplete dataframe !!!",
                log_message=msg,
            )
            raise IncompleteDataframeException(msg)

    return df, symbols_kline_attrs


def create_A_hdf(live=False):
    df, symbols_kline_attrs = create_base_hdf("ETHUSDT", 11, live=live)
    df = A_transform(df.copy(), symbols_kline_attrs)
    return df, "ETH"


def create_B_hdf(live=False):
    df, symbols_kline_attrs = create_base_hdf("ETHUSDT", 11, live=live)
    df = B_transform(df.copy(), symbols_kline_attrs)
    return df, "ETH"


def create_C_hdf(live=False):
    df, symbols_kline_attrs = create_base_hdf("ETHUSDT", 11, live=live)
    df = C_transform(df.copy(), symbols_kline_attrs)
    return df, "ETH"


def create_D_hdf(live=False):
    df, symbols_kline_attrs = create_base_hdf("ETHUSDT", 2, live=live)
    df = A_transform(df.copy(), symbols_kline_attrs)
    return df, "ETH"


def create_E_hdf(live=False):
    df, symbols_kline_attrs = create_base_hdf("ETHUSDT", 2, live=live)
    df = B_transform(df.copy(), symbols_kline_attrs)
    return df, "ETH"


def create_F_hdf(live=False):
    df, symbols_kline_attrs = create_base_hdf("ETHUSDT", 5, live=live)
    df = C_transform(df.copy(), symbols_kline_attrs)
    return df, "ETH"


def create_A2_hdf(live=False):
    df, symbols_kline_attrs = create_base_hdf("ADAUSDT", 11, live=live)
    df = A_transform(df.copy(), symbols_kline_attrs)
    return df, "ADA"


def create_B2_hdf(live=False):
    df, symbols_kline_attrs = create_base_hdf("ADAUSDT", 11, live=live)
    df = B_transform(df.copy(), symbols_kline_attrs)
    return df, "ADA"


def create_C2_hdf(live=False):
    df, symbols_kline_attrs = create_base_hdf("ADAUSDT", 11, live=live)
    df = C_transform(df.copy(), symbols_kline_attrs)
    return df, "ADA"


def create_D2_hdf(live=False):
    df, symbols_kline_attrs = create_base_hdf("ADAUSDT", 2, live=live)
    df = A_transform(df.copy(), symbols_kline_attrs)
    return df, "ADA"


def create_E2_hdf(live=False):
    df, symbols_kline_attrs = create_base_hdf("ADAUSDT", 2, live=live)
    df = B_transform(df.copy(), symbols_kline_attrs)
    return df, "ADA"


def create_F2_hdf(live=False):
    df, symbols_kline_attrs = create_base_hdf("ADAUSDT", 5, live=live)
    df = C_transform(df.copy(), symbols_kline_attrs)
    return df, "ADA"
