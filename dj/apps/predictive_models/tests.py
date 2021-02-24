from django.test import TestCase
from apps.predictive_models.jane1.load_models import load_autoencoder, load_predictmodel

class PytorchTestCase(TestCase):
    def test_pytorch_init(self):
        import torch

    def test_load_models_from_state_dict(self):
        load_autoencoder()
        load_predictmodel()
