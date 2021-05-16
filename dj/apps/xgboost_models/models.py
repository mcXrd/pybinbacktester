from django.db import models
from apps.xgboost_models.run_xgboost_models import get_best_model_code
from django.utils.timezone import now

# Create your models here.


class BestModelCode(models.Model):
    code = models.CharField(null=True, blank=True, max_length=20)
    expected_profit = models.FloatField(null=True, blank=True)
    start_evaluating = models.DateTimeField(null=True, blank=True)
    done_evaluating = models.DateTimeField(null=True, blank=True)

    def evaluate(self):
        self.start_evaluating = now()
        self.save()
        code, expected_profit = get_best_model_code()
        self.code = code
        self.expected_profit = expected_profit
        self.done_evaluating = now()
        self.save()
