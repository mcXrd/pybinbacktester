import logging

from django.core.management.base import BaseCommand
from django.core.management import call_command
import time
from django.utils.timezone import now
from apps.predictive_models.models import CronLog, AlertLog

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
                call_command("open_positions_v2")
                time.sleep(6)

            if currenct_second > 25 and currenct_second < 30:
                call_command("delete_blowing_logs")
                time.sleep(6)
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