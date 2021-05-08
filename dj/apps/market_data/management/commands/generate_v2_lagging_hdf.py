from django.core.management.base import BaseCommand
from apps.market_data.generate_market_data_hdf_utils import (
    create_dataframe_v2,
    save_output,
    fetch_input,
    command_datetime,
    create_base_dataframe,
    merge_symbols_and_kline_attrs,
    add_forecasts_to_df,
)
from apps.market_data.technical_indicators_utils import SMA


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

    def handle(self, *args, **kwargs):
        qs = fetch_input(kwargs["time_interval"], kwargs["pairs"])
        kline_attrs = ["close_price", "volume"]
        df = create_base_dataframe(qs, kline_attrs=kline_attrs)
        symbols_kline_attrs = merge_symbols_and_kline_attrs(qs, df, kline_attrs)
        output_df_v2_lagged = SMA(df, 5, symbols_kline_attrs, 1)
        output_df_v2_lagged = SMA(output_df_v2_lagged, 30, symbols_kline_attrs, 1)
        output_df_v2_lagged = SMA(output_df_v2_lagged, 60, symbols_kline_attrs, 1)
        output_df_v2_lagged = SMA(output_df_v2_lagged, 90, symbols_kline_attrs, 1)
        output_df_v2_lagged = SMA(output_df_v2_lagged, 120, symbols_kline_attrs, 1)
        output_df_v2_lagged = output_df_v2_lagged[150:]
        output_df_v2_lagged = add_forecasts_to_df(output_df_v2_lagged, live=False)
        save_output(output_df_v2_lagged, "v2_lagging_" + kwargs["hdf_filename"])
