import logging

from django.core.management.base import BaseCommand
from django.core.management import call_command
import time
from django.utils.timezone import now
from apps.predictive_models.models import CronLog
from apps.market_data.sync_kline_utils import main as sync_kline_main
from apps.market_data.sync_kline_utils import remove_too_old_klines
from apps.xgboost_models.models import BestModelCode

logger = logging.getLogger(__name__)

DAYS = 12
TIME_INTERVAL = "{} days ago UTC".format(DAYS)


def main():
    remove_too_old_klines(days=DAYS + 1)
    sync_kline_main(
        max_workers=1, time_interval=[TIME_INTERVAL], coins=["ADAUSDT", "ETHUSDT"]
    )
    best_model_code = BestModelCode()
    best_model_code.evaluate()


class Command(BaseCommand):
    help = "Liquidate positions"

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **kwargs):
        main()
