from chronosphere.models import Decider
from django.core.management.base import BaseCommand


def register_decider(decider_url: str, decider_name: str, tick_period_seconds: int) -> None:
    Decider.objects.create(decider_url=decider_url, decider_name=decider_name)


class Command(BaseCommand):
    help = 'Register single decider'

    def add_arguments(self, parser):
        parser.add_argument('--decider-url', type=str)
        parser.add_argument('--decider-name', type=str)

    def handle(self, *args, **kwargs):
        register_decider(kwargs['decider_url'], kwargs['decider_name'])
