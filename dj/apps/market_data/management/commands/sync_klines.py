from django.core.management.base import BaseCommand
from django.utils import timezone
import logging
from apps.market_data.sync_kline_utils import main
import datetime

logger = logging.getLogger(__name__)
TIME_INTERVAL = ["5 Jul, 2020", "22 May, 2021"]
TIME_INTERVAL = ["5 Jul, 2020", "22 Dec, 2020"]
TIME_INTERVAL = ["1 Jan, 2020", "1 Jan, 2021"]

TIME_INTERVAL = [datetime.date(2021, 1, 1), datetime.date(2021, 10, 30)]


class Command(BaseCommand):
    help = "Sync klines"

    def add_arguments(self, parser):
        parser.add_argument(
            "--time-interval", type=str, required=False, default=TIME_INTERVAL
        )
        parser.add_argument("--coin", type=str, required=False)

    def handle(self, *args, **kwargs):
        start_time = timezone.now()
        logger.info("Started at %s" % start_time.strftime("%X"))
        main(kwargs["time_interval"], [kwargs["coin"]])
        logger.info("End time is %s" % timezone.now().strftime("%X"))
        logger.info(
            "Duration %s seconds" % (timezone.now() - start_time).total_seconds()
        )
