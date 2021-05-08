from django.test import TestCase
from apps.predictive_models.jane1.load_models import (
    load_autoencoder,
    load_predictmodel,
    load_train_df,
)
from apps.predictive_models.models import Position
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
from django.utils.timezone import now, timedelta
from apps.predictive_models.jane1.load_models import take_currencies_from_df_columns


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
        currencies = take_currencies_from_df_columns(df)
        trade_strategy = MeanStrategy()
        trade_session = TradeSession(initial_value=400000, rake=0.00018)
        trade_skipped = 0
        trades_dict = {}
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
                currency = trade_strategy.pick_currency(numpy_model_output, currencies)
                dd = "{}_{}".format(currency, considered_side)
                if trades_dict.get(dd) is None:
                    trades_dict[dd] = 1
                else:
                    trades_dict[dd] += 1

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
        print("Trades dict {}".format(trades_dict))
        assert trade_session.initial_value * revenue < trade_session.current_value

    def test_predict_from_train_df_90d(self):
        self._predict_from_train_df(days=90, revenue=3.0)

    def test_predict_from_train_df_60d(self):
        self._predict_from_train_df(days=60, revenue=2.0)

    def test_predict_from_train_df_30d(self):
        self._predict_from_train_df(days=30, revenue=1.375)

    def test_predict_from_train_df_14d(self):
        self._predict_from_train_df(days=14, revenue=1.0)

    def test_predict_from_train_df_7d(self):
        self._predict_from_train_df(days=7, revenue=0.87)


class ModelPerformanceLiveDataTestCase(TestCase):

    TEST_CASE_PLUS_30_DAYS = 90
    TIME_INTERVAL = "{} days ago UTC".format(TEST_CASE_PLUS_30_DAYS)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if Kline.objects.count() == 0:
            call_command(
                "sync_klines",
                max_workers=1,
                time_interval=ModelPerformanceLiveDataTestCase.TIME_INTERVAL,
            )  # max_workers needs to be 1 for tests

        pair_count = len(settings.USDT_FUTURES_PAIRS)
        exchange_count = 2
        count_in_db = Kline.objects.count()
        expected_count = cls.TEST_CASE_PLUS_30_DAYS * 24 * exchange_count * pair_count
        assert abs(abs(count_in_db) - abs(expected_count)) < 100

    def template_is_create_live_df_ordered(self, days, live=False):
        df, currencies = create_live_df(days, live=live)
        last_close_time = None
        for i in range(len(df)):
            feature_row, real_output = get_feature_row_and_real_output(df, i)
            close_time = feature_row.name.to_pydatetime()
            if last_close_time is None:
                last_close_time = close_time
                continue

            self.assertLess(last_close_time, close_time)
            self.assertEqual(close_time - last_close_time, timedelta(hours=1))
            last_close_time = close_time

    def test_is_create_live_df_ordered_60(self):
        self.template_is_create_live_df_ordered(60)

    def test_is_create_live_df_ordered_30(self):
        self.template_is_create_live_df_ordered(30)

    def test_is_create_live_df_ordered_30_live(self):
        self.template_is_create_live_df_ordered(30, live=True)

    def test_is_last_close_time_actual(self):
        df, currencies = create_live_df(3, live=True)
        feature_row, real_output = get_feature_row_and_real_output(df, len(df) - 2)
        close_time = feature_row.name.to_pydatetime()
        self.assertLess(timedelta(minutes=0), now() - close_time)
        self.assertLess(now() - close_time, timedelta(minutes=now().minute + 1))

    def test_is_create_live_df_ordered_7(self):
        self.template_is_create_live_df_ordered(7)

    def test_is_create_live_df_ordered_7_live(self):
        self.template_is_create_live_df_ordered(7, live=True)

    def template_n_days_test(self, days=30, revenue=1.01):
        assert self.TEST_CASE_PLUS_30_DAYS > days + 29
        df, currencies = create_live_df(days, live=False)

        trade_strategy = MeanStrategy()
        trade_session = TradeSession(initial_value=400000, rake=0.00018)
        trade_skipped = 0

        trades_dict = {}
        for i in range(len(df)):
            if trade_strategy.do_skip():
                continue
            if i < len(df) - (days * 24):
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
                df_true_change_percentage = real_output[true_change_index]

                currency = trade_strategy.pick_currency(numpy_model_output, currencies)
                considered_side = trade_strategy.pick_side(numpy_model_output)

                dd = "{}_{}".format(currency, considered_side)
                if trades_dict.get(dd) is None:
                    trades_dict[dd] = 1
                else:
                    trades_dict[dd] += 1

                resolved_currency, resolved_side = resolve_trade(
                    df, i, trade_strategy, currencies
                )  # this function will by used in the production setting, so I wanna checked if it matches this test as well
                self.assertEqual(considered_side, resolved_side)
                self.assertEqual(currency, resolved_currency)

                current_kline = Kline.objects.get(
                    symbol=settings.EXCHANGE_FUTURES + "_" + currency,
                    close_time=close_time,
                )
                try:
                    one_hour_kline = current_kline.get_kline_shifted_forward(2)
                except Kline.DoesNotExist:
                    id1 = one_hour_kline.id
                    id2 = Kline.objects.order_by("-close_time").first()
                    id3 = Kline.objects.order_by("-close_time").last()
                    raise Exception("{} - {} - {}".format(id1, id2, id3))

                self.assertLess(current_kline.id, one_hour_kline.id)

                true_change_percentage = (
                    one_hour_kline.close_price / current_kline.close_price
                ) - 1
                msg = "{} - {} - {}".format(
                    one_hour_kline.close_price,
                    current_kline.close_price,
                    str(df.iloc[i, :]),
                )
                self.assertEqual(
                    df_true_change_percentage, true_change_percentage, msg=msg
                )
                trade_session.trade(true_change_percentage, considered_side)
            else:
                trade_skipped += 1

        print(
            "Live trades skipped: {}".format(
                trade_skipped / (trade_session.trades + trade_skipped)
            )
        )
        print(
            "Live Trade result after {} trades: {}".format(
                trade_session.trades, trade_session.current_value
            )
        )
        print("Trades dict {}".format(trades_dict))
        self.assertLess(
            trade_session.initial_value * revenue,
            trade_session.current_value,
            msg="{} days test - after trade value {}".format(
                days, trade_session.current_value
            ),
        )

    def test_live_60_days(self):
        self.template_n_days_test(days=60, revenue=2.0)

    def test_live_30_days(self):
        self.template_n_days_test(days=30, revenue=1.3)

    def test_live_14_days(self):
        self.template_n_days_test(days=14, revenue=1.1)

    def test_live_7_days(self):
        self.template_n_days_test(days=7, revenue=0.92)

    def test_live_2_days(self):
        self.template_n_days_test(days=2, revenue=0.88)

    def test_live_1_days(self):
        self.template_n_days_test(days=1, revenue=0.88)
