from apps.market_data.models import Kline
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Remove klines"

    def handle(self, *args, **kwargs):
        Kline.objects.all().delete()
