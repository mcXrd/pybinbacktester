import logging

from django.core.management.base import BaseCommand
from apps.xgboost_models.models import BestRecommendation
from apps.market_data.sync_kline_utils import main as sync_kline_main
from django.utils.timezone import now

logger = logging.getLogger(__name__)


def main():
    last_br = BestRecommendation.objects.last()
    if not last_br.should_recreate():
        return
    br = BestRecommendation.objects.create()
    br.start_evaluating = now()
    br.save()
    sync_kline_main(
        max_workers=1,
        time_interval=["6 minutes ago UTC"],
        coins=["ADAUSDT", "ETHUSDT"],
        use_spot=False,
        use_futures=True,
    )
    br.evaluate()


class Command(BaseCommand):
    help = "Evaluate best recommendation"

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **kwargs):
        main()
