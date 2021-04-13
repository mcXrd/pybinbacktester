import logging
import os
from datetime import datetime
from typing import List

import dateutil.parser
import django
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.timezone import get_current_timezone
from apps.market_data.models import Kline
from apps.market_data.generate_market_data_hdf_utils import (
    exchangify_pairs,
    create_dataframe,
)


logger = logging.getLogger(__name__)


def validate_pairs(pairs):
    """
    checking if symbol is synced
    """
    for pair in exchangify_pairs(pairs):
        if not Kline.objects.exists(symbol=pair):
            raise Exception("Pair {} is not synced".format(pair))


def fetch_input(
    time_interval: List[datetime], pairs: List[str]
) -> django.db.models.query.QuerySet:
    kwargs = {"symbol__in": exchangify_pairs(pairs)}
    if time_interval is None:
        return Kline.objects.filter(**kwargs)
    qs = Kline.objects.filter(
        open_time__gte=time_interval[0], open_time__lte=time_interval[1], **kwargs
    )
    return qs


def save_output(df, filename) -> None:
    file_path = os.path.join(settings.HDF_STORAGE_PATH, filename)
    df.to_hdf(file_path, "df")
    logger.info(f"Dataframe saved, file path: {file_path}")


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

        parser.add_argument(
            "--pairs",
            nargs="+",
            type=str,
            required=False,
            help="'BTCUSDT' 'ETHUSDT' 'BNBUSDT' 'ADAUSDT'",
        )

    def handle(self, *args, **kwargs):
        qs = fetch_input(kwargs["time_interval"], kwargs["pairs"])
        output_df = create_dataframe(qs)
        save_output(output_df, kwargs["hdf_filename"])
