from apps.market_data.models import Kline
from django.core.management.base import BaseCommand
from django.utils.timezone import now
from django.conf import settings
from apps.market_data.sync_kline_utils import main as sync_kline_main


class Command(BaseCommand):
    help = "Resync klines"

    def handle(self, *args, **kwargs):
        minutes_to_sync = None
        if not Kline.objects.exists():
            minutes_to_sync = settings.KLINES_KEPT_FOR_DAYS * 24 * 60
        else:
            seconds_since_latest_kline_close_time = (
                now() - Kline.objects.latest("close_time").close_time
            ).total_seconds()
            minutes_to_sync = int(seconds_since_latest_kline_close_time / 60) + 2
        sync_kline_main(
            time_interval=["{} minutes ago UTC".format(minutes_to_sync)],
            coins=["ADAUSDT", "ETHUSDT"],
            use_spot=False,
            use_futures=True,
        )
