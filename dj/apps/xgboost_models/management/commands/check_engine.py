import logging

from django.core.management.base import BaseCommand
from django.conf import settings
from apps.predictive_models.models import AlertLog
from django.utils.timezone import now, timedelta
from apps.predictive_models.models import Position
from apps.xgboost_models.sms import send_alert_sms


logger = logging.getLogger(__name__)


def main():
    send_sms = False
    positions_qs = Position.objects.filter(
        liquidated=False, liquidated_at__lte=now() - timedelta(minutes=20)
    )

    if positions_qs.exists():  # position not liquidated and liquidated_at not extended
        send_sms = True

    positions_qs = Position.objects.filter(liquidated=False)
    all_closed = True
    if positions_qs.exists():
        all_closed = False

    positions_qs = Position.objects.filter(
        liquidated_at__gte=now() - timedelta(minutes=20)
    )
    if (
        not positions_qs.exists() and all_closed
    ):  # all positions liquidated and no new opened
        send_sms = True

    if send_sms is False:
        return

    qs = AlertLog.objects.filter(
        name="sms", created_at__gt=now() - timedelta(minutes=60)
    )
    if qs.exists():
        return
    AlertLog.objects.create(name="sms", created_at=now())
    send_alert_sms()


class Command(BaseCommand):
    help = "Check engine"

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **kwargs):
        main()
