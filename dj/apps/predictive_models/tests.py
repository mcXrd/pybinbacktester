from django.test import TestCase
from apps.predictive_models.jane1.load_models import load_autoencoder, load_predictmodel
from apps.predictive_models.models import Position
from django.utils.timezone import now


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
            quantity=20,
            symbol="usdtfutures_ADAUSDT",
            side=Position.SHORT,
        )
