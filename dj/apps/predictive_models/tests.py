from django.test import TestCase
from apps.predictive_models.jane1.load_models import load_autoencoder, load_predictmodel
from apps.predictive_models.models import Position
from django.utils.timezone import now
from apps.market_data.models import Kline
import datetime
from django.core.management import call_command


class PytorchTestCase(TestCase):
    def test_pytorch_init(self):
        import torch

    def test_load_models_from_state_dict(self):
        load_autoencoder()
        load_predictmodel()


class PositionTestCase(TestCase):
    def test_create_empty_position(self):
        Position.objects.create(
            liquidate_at=now(),
            quantity=6,
            symbol="usdtfutures_ADAUSDT",
            side=Position.LONG,
        )


class ModelPerformanceTestCase(TestCase):

    JANE1_PAIRS = ["BTCUSDT" "ETHUSDT" "BNBUSDT" "ADAUSDT"]
    TIME_INTERVAL = "30 days ago UTC"

    def setUp(self):
        if Kline.objects.count() == 0:
            call_command(
                "sync_klines",
                max_workers=1,
                time_interval=ModelPerformanceTestCase.TIME_INTERVAL,
            )  # max_workers needs to be 1 for tests

    def test_last_month(self):
        kwargs = {
            "open_time__gt": datetime.datetime.today() - datetime.timedelta(days=30)
        }
        raise Exception(Kline.objects.filter(**kwargs).count())
