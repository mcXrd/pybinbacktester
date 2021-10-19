from typing import List, Tuple
from datetime import datetime
from apps.market_data.models import Kline
from apps.market_data.sync_kline_utils import create_complete_symbol_name
from django.conf import settings


def validate_db(
    coin: str, exchange: str = settings.EXCHANGE_SPOT
) -> Tuple[List[Tuple[Kline, Kline]], int]:
    missings = []
    last_kline = None
    symbol = create_complete_symbol_name(exchange, coin)
    for kline in Kline.objects.filter(symbol=symbol).order_by("close_time").all():
        if last_kline is None:
            last_kline = kline
            continue

        later = kline.close_time
        now = last_kline.close_time
        difference = (later - now).total_seconds()
        if difference != 60:
            print("Missing between {} {}".format(now, later))
            missings.append((last_kline, kline))
        last_kline = kline
    return missings, Kline.objects.all().count()


def remove_duplicates():
    from django.db.models import Count

    Kline.objects.values("name").annotate(Count("id")).order_by().filter(
        id__count__gt=1
    )
