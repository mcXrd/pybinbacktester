from django.db import models
from apps.xgboost_models.run_xgboost_models import get_best_model_code
from django.utils.timezone import now
from apps.xgboost_models.run_xgboost_models import model_codes, hdf_create_functions
from apps.xgboost_models.run_xgboost_models import simulate
from apps.xgboost_models.run_xgboost_models import get_X_from_df

# Create your models here.


class BestModelCode(models.Model):
    code = models.CharField(null=True, blank=True, max_length=20)
    expected_profit = models.FloatField(null=True, blank=True)
    start_evaluating = models.DateTimeField(null=True, blank=True)
    done_evaluating = models.DateTimeField(null=True, blank=True)

    def evaluate(self):
        code, expected_profit = get_best_model_code()
        self.code = code
        self.expected_profit = expected_profit
        self.done_evaluating = now()
        self.save()


class BestRecommendation(models.Model):
    SHORT = "SHORT"
    LONG = "LONG"
    PASS = "PASS"
    SIDE_CHOICES = (
        (SHORT, "Short"),
        (LONG, "Long"),
        (PASS, "Pass"),
    )
    symbol = models.CharField(null=False, blank=False, max_length=20)
    side = models.CharField(
        null=False, blank=False, max_length=20, choices=SIDE_CHOICES
    )
    start_evaluating = models.DateTimeField(null=True, blank=True)
    done_evaluating = models.DateTimeField(null=True, blank=True)

    def evaluate(self):
        last_model = BestModelCode.objects.last()
        code = last_model.code
        if code.endswith("2"):
            self.symbol = "ADA"
        else:
            self.symbol = "ETH"

        hdf_function = None
        i = 0
        for one in model_codes:
            if one == code:
                hdf_function = hdf_create_functions[i]
                break
            i += 1
        assert hdf_function
        df, coin = hdf_function()
        assert coin == self.symbol
        (
            initial_bank,
            trading_hours,
            skipped_hours_ratio,
            hours_in_test,
            model,
        ) = simulate(df, self.symbol)

        live_df, coin = hdf_function(live=True)
        assert coin == self.symbol

        X = get_X_from_df(live_df, self.symbol)
        X = X[-1:]
        Y = model.predict(X)
        assert len(Y) == 1
        side_int_value = int(Y[0])
        t = {
            1: BestRecommendation.PASS,
            10: BestRecommendation.SHORT,
            20: BestRecommendation.LONG,
        }
        self.side = t[side_int_value]
        self.done_evaluating = now()
        self.save()
