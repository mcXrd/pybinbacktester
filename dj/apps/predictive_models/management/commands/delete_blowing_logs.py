import logging

from django.core.management.base import BaseCommand
from apps.predictive_models.models import CronLog, PositionLog, AlertLog
from django.utils.timezone import now, timedelta
from django.conf import settings

logger = logging.getLogger(__name__)


def main():
    pass


class Command(BaseCommand):
    help = "Liquidate positions"

    def add_arguments(self, parser):
        qs = CronLog.objects.filter(created_at__gt=now() - timedelta(minutes=60))
        if qs.count() > settings.BLOWING_LOGS_HOUR_LIMIT:
            msgs = []
            jsons = []
            for cronlog in qs:
                msgs.append(cronlog.log_message)
                jsons.append(cronlog.log_json)
            AlertLog.objects.create(
                name="BLOWING_LOGS_HOUR_LIMIT - CronLog",
                log_message=str(msgs),
                log_json={"all_jsons": jsons},
            )
            qs.delete()

        qs = PositionLog.objects.filter(created_at__gt=now() - timedelta(minutes=60))
        if qs.count() > settings.BLOWING_LOGS_HOUR_LIMIT:
            msgs = []
            jsons = []
            for cronlog in qs:
                msgs.append(cronlog.log_message)
                jsons.append(cronlog.log_json)
            AlertLog.objects.create(
                name="BLOWING_LOGS_HOUR_LIMIT - PositionLog",
                log_message=str(msgs),
                log_json={"all_jsons": jsons},
            )
            qs.delete()

    def handle(self, *args, **kwargs):
        main()
