from django.core.management.base import BaseCommand
from apps.market_data.validate_db_utils import validate_db
from apps.market_data.models import Kline


class Command(BaseCommand):
    help = "Validate db for completnes of sync"

    def add_arguments(self, parser):
        parser.add_argument("--coin", type=str, required=True)

    def handle(self, *args, **kwargs):
        coin = kwargs["coin"]
        print(validate_db(coin))
        print("first {} -- last {}".format(Kline.objects.first(), Kline.objects.last()))
        print("Done.")
