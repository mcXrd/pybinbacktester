import logging
from datetime import datetime
from typing import List

from chronosphere.models import Chronosphere, Decider, TickRecord
from django.core.management.base import BaseCommand
from django.utils.timezone import now

logger = logging.getLogger(__name__)


def get_all_unprocessed_ticks(chronosphere: Chronosphere, tick_period_seconds: int) -> List[datetime]:
    # TODO - first need to be able to create hdf dataframe based on tick period (parametrized)
    pass


def call_decider(decider: Decider, tick) -> dict:
    # TODO - first need to be able to create hdf dataframe based on tick period (parametrized)
    pass


def register_decider(decider_url: str, decider_name: str, tick_period_seconds: int) -> None:
    decider = Decider.objects.get_or_create(decider_url=decider_url, decider_name=decider_name)
    chronosphere = Chronosphere.objects.create(decider=decider, end_time=now())

    all_ticks = get_all_unprocessed_ticks(chronosphere)
    counter = 0
    for tick in all_ticks:
        tick_result = call_decider(decider, tick)
        if tick_result['action'] == 'buy':
            TickRecord.objects.create(chronosphere=chronosphere, tick_result=tick_result)

        counter += 1
        completed_percent = counter / len(all_ticks)
        if completed_percent.is_integer():
            logger.info(f"decider {decider_name} completed {completed_percent} %")
            chronosphere.completed_percent = completed_percent
            chronosphere.save()

    chronosphere.completed_percent = 100
    chronosphere.save()


class Command(BaseCommand):
    help = 'Register single decider'

    def add_arguments(self, parser):
        parser.add_argument('--decider-url', type=str)
        parser.add_argument('--decider-name', type=str)

    def handle(self, *args, **kwargs):
        register_decider(kwargs['decider_url'], kwargs['decider_name'])
