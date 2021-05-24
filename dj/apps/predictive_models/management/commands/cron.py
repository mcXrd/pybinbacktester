import logging

from django.core.management.base import BaseCommand
from django.core.management import call_command
import time
from django.utils.timezone import now
from apps.predictive_models.models import CronLog, AlertLog
from apps.xgboost_models.create_hdfs_for_models import IncompleteDataframeException
from apps.market_data.models import Kline
from apps.xgboost_models.models import BestModelCode

logger = logging.getLogger(__name__)


def main():
    while True:
        try:
            time.sleep(1)
            currenct_second = now().second
            if currenct_second < 5:
                call_command("evaluate_best_model")
                call_command("evaluate_best_recommendation")
                call_command("liquidate_positions")
                time.sleep(3)
                call_command("open_positions_v2")
                time.sleep(6)

            if currenct_second > 25 and currenct_second < 30:
                call_command("delete_blowing_logs")
                time.sleep(6)
        except IncompleteDataframeException as e:
            AlertLog.objects.create(
                name="cron.py Exception - loop will continue in 6*60 sec - will try to delete klines and sync them again",
                log_message=str(e),
            )
            time.sleep(60 * 6)
            BestModelCode.objects.last().delete()
            BestModelCode.objects.last().delete()
            Kline.objects.all().delete()
            call_command("evaluate_best_model")

        except Exception as e:
            AlertLog.objects.create(
                name="cron.py Exception - loop will continue in 50 sec",
                log_message=str(e),
            )
            time.sleep(50)


class Command(BaseCommand):
    help = "Liquidate positions"

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **kwargs):
        main()
