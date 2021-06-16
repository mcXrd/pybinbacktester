import logging
from django.core.management.base import BaseCommand
from django.utils.timezone import now
from concurrent.futures import TimeoutError, ProcessPoolExecutor
from apps.market_data.sync_kline_utils import stop_process_pool
from django.core.management import call_command
from apps.xgboost_models.create_hdfs_for_models import create_base_hdf, C_transform


from sklearn.model_selection import train_test_split
from apps.xgboost_models.models import BestRecommendation
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from apps.xgboost_models.create_hdfs_for_models import create_base_hdf_v1_moments

logger = logging.getLogger(__name__)


def check_x_for_leak(X):
    for column in X.columns:
        assert "trade_in_" not in column


def create_model(df_orig, coin):
    train_size = 60 * 24 * 55
    test_size = 60 * 24 * 5
    df_orig = df_orig.fillna(0)
    index = len(df_orig) - 2

    start = index - (train_size + test_size)
    assert start > 0
    end = index
    assert end > 0

    df_orig = df_orig[start:end]

    df = df_orig.copy()
    Y = df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)]

    Y_MEAN = np.mean(np.abs(Y))
    Y_MEAN = Y_MEAN * (1 / 2)

    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] > Y_MEAN] = 20
    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] < -Y_MEAN] = 10
    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] < 9] = 1

    first_element = "usdtfutures_{}USDT_close_price_ewm_mean".format(coin)

    X = df.loc[:, first_element:]
    check_x_for_leak(X)

    test_size = test_size / (test_size + train_size)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=6, shuffle=False
    )
    X_train = X_train.to_numpy()
    Y_train = Y_train.to_numpy()
    X_test = X_test.to_numpy()
    Y_test = Y_test.to_numpy()

    model = TabNetClassifier(seed=np.random.randint(1000, size=1)[0])

    model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)])

    return model


def predict_row(df_orig, index, model, coin):
    df_orig = df_orig.iloc[index : index + 1]
    df_orig = df_orig.copy()
    df_orig = df_orig.fillna(0)

    first_element = "usdtfutures_{}USDT_close_price_ewm_mean".format(coin)
    X = df_orig.loc[:, first_element:]
    check_x_for_leak(X)
    X = X.to_numpy()
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

    df_train = create_base_hdf_v1_moments("ADAUSDT", 61, live=False)

    expected_len = 60 * 24 * 60
    assert len(df_train) > (expected_len - 2000)
    assert len(df_train) < (expected_len + 2000)

    model = create_model(df_train, "ADA")

    call_command("resync_klines_dynamically")

    df_eval = create_base_hdf_v1_moments("ADAUSDT", 4, live=True)

    profit, side = predict_row(df_eval, len(df_eval) - 1, model, "ADA")
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
