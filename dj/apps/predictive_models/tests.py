from django.test import TestCase


class PytorchTestCase(TestCase):
    def test_pytorch_init(self):
        import torch
