import logging

from django.core.management.base import BaseCommand
from apps.predictive_models.models import Position
from django.utils import timezone
from apps.predictive_models.models import TradeInterfaceBinanceFutures
from apps.predictive_models.models import CronLog
from django.utils.timezone import now, timedelta
import time

logger = logging.getLogger(__name__)


def main():
    stop_qs = Position.objects.filter(
        liquidation_being_processed=True, liquidated=False
    )
    if stop_qs.exists():
        CronLog.objects.create(
            name="Liquidation in process", log_message=str(list(stop_qs.all()))
        )
        return
    positions_qs = Position.objects.filter(
        liquidate_at__lte=timezone.now(), liquidated=False, open_finished__isnull=False
    )
    for position in positions_qs:
        CronLog.objects.create(name="Liquidating position", log_message=str(position))
        position.liquidate(TradeInterfaceBinanceFutures())

    positions_open_qs = Position.objects.filter(liquidated=False)

    if not positions_open_qs.exists():
        time.sleep(2)
        from apps.predictive_models.live_trade_utils import (
            liquidate_remaining_positions,
        )

        liquidate_remaining_positions()


class Command(BaseCommand):
    help = "Liquidate positions"

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **kwargs):
        main()
