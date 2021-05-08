from django.core.management.base import BaseCommand
from apps.market_data.generate_market_data_hdf_utils import (
    create_dataframe_v2,
    save_output,
    fetch_input,
    command_datetime,
    add_forecasts_to_df,
)


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
        output_df_v2 = create_dataframe_v2(qs, live=False, minutes = 45)
        output_df_v2 = add_forecasts_to_df(output_df_v2, live=False)
        save_output(output_df_v2, "v2_raw_" + kwargs["hdf_filename"])
