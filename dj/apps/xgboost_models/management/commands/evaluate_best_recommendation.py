import logging

from django.core.management.base import BaseCommand
from apps.xgboost_models.models import BestRecommendation
from apps.market_data.sync_kline_utils import main as sync_kline_main
from django.utils.timezone import now
from concurrent.futures import TimeoutError, ProcessPoolExecutor
from apps.market_data.sync_kline_utils import stop_process_pool
from django.core.management import call_command

logger = logging.getLogger(__name__)


def main():
    last_br = BestRecommendation.objects.last()
    if last_br and not last_br.should_recreate():
        return
    br = BestRecommendation.objects.create()
    br.start_evaluating = now()
    br.save()
    call_command("resync_klines_dynamically")
    br.evaluate()


class Command(BaseCommand):
    help = "Evaluate best recommendation"

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **kwargs):
        with ProcessPoolExecutor(max_workers=1) as executor:
            from apps.xgboost_models.models import BestRecommendation

            timeout = 60 * BestRecommendation.TIMEOUT_MINUTES
            future = executor.submit(main)
            try:
                future.result(timeout=timeout)
            except TimeoutError:
                from apps.predictive_models.models import AlertLog

                stop_process_pool(executor)

                AlertLog.objects.create(
                    name="Best model evaluation timeouted - {}".format(now())
                )
