import logging

from django.core.management.base import BaseCommand
from apps.predictive_models.models import Position
from django.utils import timezone
from apps.predictive_models.models import TradeInterfaceBinanceFutures
from apps.predictive_models.models import CronLog
from django.utils.timezone import now, timedelta

logger = logging.getLogger(__name__)


def main():
    stucked_liquidations = Position.objects.filter(
        liquidated=False,
        liquidation_being_processed=True,
        start_to_liquidate__lte=now() - timedelta(minutes=15),
    )
    if stucked_liquidations.exists():
        if stucked_liquidations.exists():
            for position in stucked_liquidations:
                trade_interface = TradeInterfaceBinanceFutures()
                position.liquidate(trade_interface)
            return

    stop_qs = Position.objects.filter(
        liquidation_being_processed=True, liquidated=False
    )
    if stop_qs.exists():
        CronLog.objects.create(
            name="Liquidation in process", log_message=str(list(stop_qs.all()))
        )
        return

    positions_qs = Position.objects.filter(
        liquidate_at__lt=timezone.now(), liquidated=False, open_finished__isnull=False
    )
    for position in positions_qs:
        CronLog.objects.create(name="Liquidating position", log_message=str(position))
        position.liquidate(TradeInterfaceBinanceFutures())


class Command(BaseCommand):
    help = "Liquidate positions"

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **kwargs):
        main()
