import logging
from django.core.management.base import BaseCommand
from django.utils.timezone import now
from concurrent.futures import TimeoutError, ProcessPoolExecutor
from apps.market_data.sync_kline_utils import stop_process_pool
from django.core.management import call_command
from apps.xgboost_models.create_hdfs_for_models import create_base_hdf, C_transform

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from apps.xgboost_models.models import BestRecommendation
import numpy as np
import tqdm

logger = logging.getLogger(__name__)


def check_x_for_leak(X):
    for column in X.columns:
        assert "trade_in_" not in column


def create_model(df_orig, coin):
    df_orig = df_orig.fillna(0)
    del df_orig["trade_in_3h_usdtfutures_{}USDT_close_price".format(coin)]
    del df_orig["trade_in_2h_usdtfutures_{}USDT_close_price".format(coin)]
    df = df_orig.copy()
    df = df.sample(frac=1 / 8)
    Y = df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)]

    Y_MEAN = np.mean(np.abs(Y))
    Y_MEAN = Y_MEAN * (1 / 8)

    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] > Y_MEAN] = 20
    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] < -Y_MEAN] = 10
    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] < 9] = 1

    first_element = "usdtfutures_{}USDT_close_price".format(coin)

    X = df.loc[:, first_element:]
    check_x_for_leak(X)

    test_size = 0.01
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=6, shuffle=False
    )
    hours_in_test = len(Y_test)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        # booster=["gbtree","gblinear"][0],
        gamma=0.1,
        # sampling_method=["uniform", "gradient_based"][1],
        # num_parallel_tree=1,
        # reg_alpha=1,
        # reg_lambda=0,
        # eta=0.3,
    )
    model.fit(X_train, Y_train)
    return model


def predict_row(df_orig, index, model, coin):
    del df_orig["trade_in_3h_usdtfutures_{}USDT_close_price".format(coin)]
    del df_orig["trade_in_2h_usdtfutures_{}USDT_close_price".format(coin)]
    df_orig = df_orig.iloc[index : index + 1]
    df_orig = df_orig.copy()
    df_orig = df_orig.fillna(0)

    first_element = "usdtfutures_{}USDT_close_price".format(coin)
    X = df_orig.loc[:, first_element:]
    check_x_for_leak(X)
    y_pred = model.predict(X)[0]
    Y_TEST_REAL = df_orig["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)][0]

    profit = -1

    side = -1
    if y_pred == 20:
        side = 1
    if y_pred == 1:
        return Y_TEST_REAL, 0

    if side * Y_TEST_REAL > 0:
        profit = 1

    return abs(Y_TEST_REAL) * profit, side


def main():
    br = BestRecommendation.objects.create()
    br.start_evaluating = now()
    br.save()
    df_train, symbols_kline_attrs = create_base_hdf("ADAUSDT", 60, live=False)
    df_train = C_transform(df_train, symbols_kline_attrs)

    expected_len = 60 * 24 * 60
    assert len(df_train) > (expected_len - 1000)
    assert len(df_train) < (expected_len + 1000)

    model = create_model(df_train, "ADA")

    df_eval, symbols_kline_attrs = create_base_hdf("ADAUSDT", 10, live=True)
    df_train = C_transform(df_train, symbols_kline_attrs)

    profit, side = predict_row(df_train, len(df_train) - 1, model, "ADA")
    t = {
        0: BestRecommendation.PASS,
        -1: BestRecommendation.SHORT,
        1: BestRecommendation.LONG,
    }
    br.symbol = "ADA"
    br.side = t[side]
    br.done_evaluating = now()
    br.save()


class Command(BaseCommand):
    help = "Evaluate best recommendation"

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **kwargs):
        with ProcessPoolExecutor(max_workers=1) as executor:
            from apps.xgboost_models.models import BestRecommendation

            timeout = 60 * BestRecommendation.TIMEOUT_MINUTES
            future = executor.submit(main)
            try:
                future.result(timeout=timeout)
            except TimeoutError:
                from apps.predictive_models.models import AlertLog

                stop_process_pool(executor)

                AlertLog.objects.create(
                    name="Best model evaluation timeouted - {}".format(now())
                )
