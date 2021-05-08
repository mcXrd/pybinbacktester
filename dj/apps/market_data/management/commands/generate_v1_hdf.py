import logging
from django.core.management.base import BaseCommand
from apps.market_data.generate_market_data_hdf_utils import (
    clean_initial_window_nans,
    create_dataframe,
    fetch_input,
    save_output,
    command_datetime,
)


logger = logging.getLogger(__name__)


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
        output_df = create_dataframe(qs, live=True)
        output_df = clean_initial_window_nans(output_df)
        save_output(output_df, kwargs["hdf_filename"])
