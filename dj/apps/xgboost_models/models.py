from django.db import models
from apps.xgboost_models.run_xgboost_models import get_best_model_code
from apps.xgboost_models.run_xgboost_models import model_codes, hdf_create_functions
from apps.xgboost_models.run_xgboost_models import simulate
from apps.xgboost_models.run_xgboost_models import get_X_from_df
from django.utils.timezone import now, timedelta
from apps.predictive_models.models import AlertLog
import pandas as pd

# Create your models here.


class BestModelCode(models.Model):
    code = models.CharField(null=True, blank=True, max_length=20)
    expected_profit = models.FloatField(null=True, blank=True)
    start_evaluating = models.DateTimeField(null=True, blank=True)
    done_evaluating = models.DateTimeField(null=True, blank=True)
    mean_const = models.FloatField(null=True, blank=True)

    def evaluate(self):
        code, expected_profit, mean_const = get_best_model_code()
        self.mean_const = mean_const
        self.code = code
        self.expected_profit = expected_profit
        self.done_evaluating = now()
        self.save()

    def unstuck(self):
        minutes = 22
        if (
            not self.done_evaluating
            and (now() - self.start_evaluating).total_seconds() > 60 * minutes
        ):
            AlertLog.objects.create(
                name="BestModelCode was stuck for more than {} minutes - deleting".format(
                    minutes
                ),
                log_message="start eval {} ".format(self.start_evaluating),
            )
            self.delete()

    def is_fresh(self):
        moving = now() - timedelta(minutes=60)
        if not self.done_evaluating:
            self.unstuck()
            return False
        return self.done_evaluating > moving

    def should_recreate(self):
        moving = now() - timedelta(minutes=30)
        if not self.done_evaluating:
            self.unstuck()
            return False
        return self.done_evaluating < moving


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
    mean_const = models.FloatField(null=True, blank=True)

    def evaluate(self):
        last_model = BestModelCode.objects.last()
        if not last_model.is_fresh():
            return
        code = last_model.code
        if code.endswith("2"):
            self.symbol = "ADA"
        else:
            self.symbol = "ETH"
        self.mean_const = last_model.mean_const

        hdf_function = None
        i = 0
        for one in model_codes:
            if one == code:
                hdf_function = hdf_create_functions[i]
                break
            i += 1
        assert hdf_function
        df, coin = hdf_function(live=False)
        df = df[
            60 * 24 * 2 :
        ]  # removing first two days - because we wont have test/eval split
        assert coin == self.symbol
        (
            initial_bank,
            trading_hours,
            skipped_hours_ratio,
            hours_in_test,
            model,
        ) = simulate(df, self.symbol, days_eval=0, mean_const=self.mean_const)

        live_df, coin = hdf_function(live=True)
        assert coin == self.symbol

        X = get_X_from_df(live_df, self.symbol)
        X = X[-1:]
        row_datetime = pd.to_datetime(X.index)
        age_of_last_row = now() - row_datetime
        assert age_of_last_row.total_seconds() < 60 * 6
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

    def is_fresh(self):
        moving = now() - timedelta(minutes=4)
        return self.done_evaluating > moving

    def should_recreate(self):
        moving = now() - timedelta(minutes=2)
        if not self.done_evaluating:
            return False
        return self.done_evaluating < moving
