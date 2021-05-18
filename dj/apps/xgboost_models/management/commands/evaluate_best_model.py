import logging

from django.core.management.base import BaseCommand
from apps.market_data.sync_kline_utils import main as sync_kline_main
from apps.market_data.sync_kline_utils import remove_too_old_klines
from apps.xgboost_models.models import BestModelCode
from apps.xgboost_models.commands_settings import DAYS, TIME_INTERVAL
from django.utils.timezone import now

logger = logging.getLogger(__name__)


def main():
    last_best_model_code = BestModelCode.objects.last()
    if last_best_model_code and not last_best_model_code.should_recreate():
        return

    best_model_code = BestModelCode.objects.create()
    best_model_code.start_evaluating = now()
    best_model_code.save()
    remove_too_old_klines(days=DAYS + 1)
    sync_kline_main(
        max_workers=1,
        time_interval=[TIME_INTERVAL],
        coins=["ADAUSDT", "ETHUSDT"],
        use_spot=False,
        use_futures=True,
    )
    best_model_code.evaluate()


class Command(BaseCommand):
    help = "Evaluate best model"

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **kwargs):
        main()
