from django.test import TestCase
from apps.predictive_models.jane1.load_models import (
    load_autoencoder,
    load_predictmodel,
    load_train_df,
)
from apps.predictive_models.models import Position
from django.utils.timezone import now
from apps.market_data.models import Kline
from django.core.management import call_command
from apps.predictive_models.jane1.load_models import preprocessing_scale_df
from apps.predictive_models.jane1.trade_strategy import MeanStrategy, Direct2hStrategy
from django.conf import settings
from apps.predictive_models.live_trade_utils import create_live_df
from apps.predictive_models.live_trade_utils import (
    get_feature_row_and_real_output,
    get_model_input,
    get_model_output,
    get_numpy_model_output,
    resolve_trade,
)


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


class TradeSession:
    def __init__(self, initial_value, rake):
        self.initial_value = initial_value
        self.rake = rake
        self.current_value = initial_value
        self.trades = 0

    def trade(self, true_change_percentage, considered_side):
        self.trades += 1
        buy_fee = self.current_value * self.rake
        self.current_value = self.current_value - buy_fee

        profit = -1
        if true_change_percentage * considered_side > 0:
            profit = 1

        change_value = self.current_value * abs(true_change_percentage)
        change_value = change_value * profit
        self.current_value = self.current_value + change_value

        sell_fee = self.current_value * self.rake
        self.current_value = self.current_value - sell_fee


class ModelPerformanceTestCase(TestCase):
    def _predict_from_train_df(self, days=30, revenue=1.375):
        df = load_train_df()
        df, last_min_max_scaler = preprocessing_scale_df(df, None)
        trade_strategy = MeanStrategy()
        trade_session = TradeSession(initial_value=400000, rake=0.00018)
        trade_skipped = 0
        for i in range(len(df) - 3):
            if trade_strategy.do_skip():
                continue
            if i < len(df) - (days * 24):
                continue
            feature_row, real_output = get_feature_row_and_real_output(df, i)
            model_input = get_model_input(feature_row)
            model_output = get_model_output(model_input)
            numpy_model_output = get_numpy_model_output(model_output)

            if trade_strategy.do_trade(numpy_model_output):
                true_change_index = trade_strategy.pick_true_change_index(
                    numpy_model_output
                )
                true_change_percentage = real_output[true_change_index]
                considered_side = trade_strategy.pick_side(numpy_model_output)
                trade_session.trade(true_change_percentage, considered_side)
            else:
                trade_skipped += 1
            [float(x) for x in model_output[0]]  # should raise with nan or inf

        print(
            "trades skipped: {}".format(
                trade_skipped / (trade_session.trades + trade_skipped)
            )
        )
        print(
            "Trade result after {} trades: {}".format(
                trade_session.trades, trade_session.current_value
            )
        )
        assert trade_session.initial_value * revenue < trade_session.current_value

    def test_predict_from_train_df_90d(self):
        self._predict_from_train_df(days=90, revenue=3.0)

    def test_predict_from_train_df_60d(self):
        self._predict_from_train_df(days=60, revenue=2.0)

    def test_predict_from_train_df_30d(self):
        self._predict_from_train_df(days=30, revenue=1.375)

    def test_predict_from_train_df_14d(self):
        self._predict_from_train_df(days=14, revenue=1.1)

    def test_predict_from_train_df_7d(self):
        self._predict_from_train_df(days=7, revenue=1.1)


class ModelPerformanceLiveDataTestCase(ModelPerformanceTestCase):

    JANE1_PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
    DAYS = 50
    TIME_INTERVAL = "{} days ago UTC".format(DAYS)

    def setUp(self):
        if Kline.objects.count() == 0:
            call_command(
                "sync_klines",
                max_workers=1,
                time_interval=ModelPerformanceLiveDataTestCase.TIME_INTERVAL,
            )  # max_workers needs to be 1 for tests

    def test_last_month(self):
        df, currencies = create_live_df(
            ModelPerformanceLiveDataTestCase.DAYS, live=False
        )

        trade_strategy = MeanStrategy()
        trade_session = TradeSession(initial_value=400000, rake=0.00018)
        trade_skipped = 0

        for i in range(len(df) - 3):
            if trade_strategy.do_skip():
                continue
            if i < len(df) - (30 * 24):
                continue
            feature_row, real_output = get_feature_row_and_real_output(df, i)
            model_input = get_model_input(feature_row)
            model_output = get_model_output(model_input)
            numpy_model_output = get_numpy_model_output(model_output)
            close_time = feature_row.name.to_pydatetime()

            if trade_strategy.do_trade(numpy_model_output):
                true_change_index = trade_strategy.pick_true_change_index(
                    numpy_model_output
                )
                test_true_change_percentage = real_output[true_change_index]

                currency = trade_strategy.pick_currency(numpy_model_output, currencies)
                considered_side = trade_strategy.pick_side(numpy_model_output)

                resolved_currency, resolved_side = resolve_trade(
                    df, i, trade_strategy, currencies
                )  # this function will by used in the production setting, so I wanna checked if it matches this test as well
                self.assertEqual(considered_side, resolved_side)
                self.assertEqual(currency, resolved_currency)

                current_kline = Kline.objects.get(
                    symbol=settings.EXCHANGE_FUTURES + "_" + currency,
                    close_time=close_time,
                )
                one_hour_kline = current_kline.get_kline_shifted_forward(2)
                true_change_percentage = (
                    one_hour_kline.close_price / current_kline.close_price
                ) - 1
                self.assertEqual(test_true_change_percentage, true_change_percentage)
                trade_session.trade(true_change_percentage, considered_side)
            else:
                trade_skipped += 1

        print(
            "trades skipped: {}".format(
                trade_skipped / (trade_session.trades + trade_skipped)
            )
        )
        print(
            "Trade result after {} trades: {}".format(
                trade_session.trades, trade_session.current_value
            )
        )
        assert trade_session.initial_value * 1.375 < trade_session.current_value
