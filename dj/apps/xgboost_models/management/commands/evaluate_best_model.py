import logging

from django.core.management.base import BaseCommand
from apps.market_data.sync_kline_utils import main as sync_kline_main
from apps.market_data.sync_kline_utils import remove_too_old_klines
from apps.xgboost_models.models import BestModelCode
from apps.xgboost_models.commands_settings import DAYS, TIME_INTERVAL
from django.utils.timezone import now
from concurrent.futures import TimeoutError, ProcessPoolExecutor
from apps.market_data.sync_kline_utils import stop_process_pool
from apps.market_data.models import Kline

logger = logging.getLogger(__name__)


def main():
    last_best_model_code = BestModelCode.objects.last()
    if last_best_model_code and not last_best_model_code.should_recreate():
        return

    best_model_code = BestModelCode.objects.create()
    best_model_code.start_evaluating = now()
    best_model_code.save()
    # remove_too_old_klines(days=DAYS + 1)
    Kline.objects.all().delete()

    sync_kline_main(
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
        with ProcessPoolExecutor(max_workers=1) as executor:
            from apps.xgboost_models.models import BestModelCode

            timeout = 60 * BestModelCode.TIMEOUT_MINUTES
            future = executor.submit(main)
            try:
                future.result(timeout=timeout)
            except TimeoutError:
                from apps.predictive_models.models import AlertLog

                stop_process_pool(executor)

                AlertLog.objects.create(
                    name="Best model evaluation timeouted - {}".format(now())
                )
