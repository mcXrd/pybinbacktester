import logging

from django.core.management.base import BaseCommand
from django.core.management import call_command
import time
from django.utils.timezone import now

logger = logging.getLogger(__name__)


def main():
    while True:
        time.sleep(1)
        currenct_second = now().second
        if currenct_second < 5:
            call_command("liquidate_positions")
            call_command("open_positions")
            time.sleep(6)


class Command(BaseCommand):
    help = "Liquidate positions"

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **kwargs):
        main()
