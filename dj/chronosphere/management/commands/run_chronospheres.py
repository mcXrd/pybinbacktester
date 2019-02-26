import logging
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import requests
from chronosphere.models import Chronosphere, Decider, TickRecord
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.timezone import now
from jsonschema import validate

logger = logging.getLogger(__name__)

MINIMUN_CHRONO_INTERVAL = timedelta(seconds=60)


def get_all_ticks(chronosphere: Chronosphere, tick_period_seconds: int = 60) -> List[datetime]:
    return list(map(lambda x: x.to_pydatetime(), pd.date_range(start=chronosphere.start_time, end=chronosphere.end_time,
                                                               freq=f"{tick_period_seconds}S")))


def call_decider(decider: Decider, tick: datetime):
    decider_response_schema = {
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "pair": {"type": "string"},
            "action": {"type": "string"},
        },
        "required": ["action"]
    }
    payload = {'tick': tick}
    r = requests.post(decider.decider_url, data=payload)
    r.raise_for_status()
    validate(instance=r.json(), schema=decider_response_schema)
    return r.json()


def get_chrono_start_time(decider: Decider,
                          default_offset_hours: int = settings.CHRONOSPHERE_INITIAL_DURATION_HOURS) -> datetime:
    try:
        return Chronosphere.objects.filter(decider=decider).latest('end_time').end_time
    except Chronosphere.DoesNotExist:
        return now() - timedelta(hours=default_offset_hours)


def create_and_run_chrono(decider: Decider):
    start_time = get_chrono_start_time(decider)
    if start_time > now() - MINIMUN_CHRONO_INTERVAL:
        return
    chronosphere = Chronosphere.objects.create(decider=decider, end_time=now(),
                                               start_time=start_time)
    all_ticks = get_all_ticks(chronosphere)
    counter = 0
    for tick in all_ticks:
        tick_result = call_decider(decider, tick)
        if tick_result['action'] == 'buy':
            TickRecord.objects.create(chronosphere=chronosphere, action=tick_result['action'],
                                      amount=tick_result['amount'], pair=tick_result['pair'])

        counter += 1
        completed_percent = counter / len(all_ticks)
        if completed_percent.is_integer() and completed_percent % 5 == 0:
            logger.info(f"decider {decider_name} completed {completed_percent} %")
            chronosphere.completed_percent = completed_percent
            chronosphere.save()

    chronosphere.completed_percent = 100
    chronosphere.save()


class Command(BaseCommand):
    help = "Run this command with cron every 'period'"

    def handle(self, *args, **kwargs):
        for decider in Decider.objects.all():
            create_and_run_chrono(decider)
