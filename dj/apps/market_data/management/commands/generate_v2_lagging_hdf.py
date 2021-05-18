from django.core.management.base import BaseCommand
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

        parser.add_argument(
            "--pairs",
            nargs="+",
            type=str,
            required=False,
            help="'BTCUSDT' 'ETHUSDT' 'BNBUSDT' 'ADAUSDT' 'XRPUSDT'",
        )

    def A_transform(self, df, symbols_kline_attrs):
        output_df_v2_lagged = SMA(df, 5, symbols_kline_attrs, 0, 30)
        output_df_v2_lagged = SMA(output_df_v2_lagged, 30, symbols_kline_attrs, 1, 30)
        output_df_v2_lagged = SMA(output_df_v2_lagged, 60, symbols_kline_attrs, 1, 30)
        output_df_v2_lagged = SMA(output_df_v2_lagged, 90, symbols_kline_attrs, 1, 30)
        output_df_v2_lagged = SMA(output_df_v2_lagged, 120, symbols_kline_attrs, 1, 30)
        output_df_v2_lagged = output_df_v2_lagged[270:]
        return output_df_v2_lagged

    def B_transform(self, df, symbols_kline_attrs):
        output_df_v2_lagged = SMA(df, 5, symbols_kline_attrs, 0, 30)
        output_df_v2_lagged = SMA(output_df_v2_lagged, 30, symbols_kline_attrs, 1, 30)
        output_df_v2_lagged = SMA(output_df_v2_lagged, 60, symbols_kline_attrs, 1, 30)
        output_df_v2_lagged = output_df_v2_lagged[270:]
        return output_df_v2_lagged

    def C_transform(self, df, symbols_kline_attrs):
        output_df_v2_lagged = SMA(df, 5, symbols_kline_attrs, 0, 30)
        output_df_v2_lagged = SMA(output_df_v2_lagged, 60, symbols_kline_attrs, 1, 30)
        output_df_v2_lagged = SMA(output_df_v2_lagged, 120, symbols_kline_attrs, 1, 30)
        output_df_v2_lagged = SMA(output_df_v2_lagged, 240, symbols_kline_attrs, 1, 30)
        output_df_v2_lagged = output_df_v2_lagged[270:]
        return output_df_v2_lagged

    def handle(self, *args, **kwargs):
        qs = fetch_input(kwargs["time_interval"], kwargs["pairs"])
        kline_attrs = ["close_price", "volume"]
        kline_attrs = ["close_price", "volume", "number_of_trades"]
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
        df = df.sort_index()
        output_df_v2_lagged = self.A_transform(df.copy(), symbols_kline_attrs)
        save_output(output_df_v2_lagged, "v2_lagging_A" + kwargs["hdf_filename"])
        output_df_v2_lagged = self.B_transform(df.copy(), symbols_kline_attrs)
        save_output(output_df_v2_lagged, "v2_lagging_B" + kwargs["hdf_filename"])
        output_df_v2_lagged = self.C_transform(df.copy(), symbols_kline_attrs)
        save_output(output_df_v2_lagged, "v2_lagging_C" + kwargs["hdf_filename"])
