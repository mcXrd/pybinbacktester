from django.test import TestCase
from apps.predictive_models.jane1.load_models import (
    load_autoencoder,
    load_predictmodel,
    load_train_df,
    resort_columns,
)
from apps.predictive_models.models import Position
from django.utils.timezone import now
from apps.market_data.models import Kline
import datetime
from django.core.management import call_command
from apps.market_data.generate_market_data_hdf_utils import (
    exchangify_pairs,
    create_dataframe,
)
from apps.predictive_models.jane1.load_models import (
    JaneStreetEncode1Dataset_Y_START_COLUMN,
    JaneStreetEncode1Dataset_Y_END_COLUMN,
)
import torch
import numpy as np
from apps.predictive_models.jane1.load_models import preprocessing_scale_df
from apps.predictive_models.jane1.trade_strategy import MeanStrategy, Direct2hStrategy
from functools import lru_cache
from django.conf import settings


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
    @staticmethod
    def take_currencies_from_df_columns(df):
        columns = []
        for column in list(df.columns):
            if "close_price" not in column:
                break
            columns.append(column)
        res = []
        for column in columns:
            sc = column.split("_")
            curr = sc[-3]
            if not curr in res:
                res.append(curr)
        return res

    def _get_feature_row_and_real_output(self, df, int_index):
        feature_row = df.loc[
            :,
            JaneStreetEncode1Dataset_Y_START_COLUMN:JaneStreetEncode1Dataset_Y_END_COLUMN,
        ].iloc[int_index, :]
        real_output = df.loc[
            :,
            :JaneStreetEncode1Dataset_Y_START_COLUMN,
        ].iloc[int_index, :-1]
        return feature_row, real_output

    def _get_model_input(self, feature_row):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_input = torch.from_numpy(np.array([feature_row])).float().to(device)
        return model_input

    @lru_cache
    def _get_models(self):
        autoencoder = load_autoencoder()
        predictmodel = load_predictmodel()
        return autoencoder, predictmodel

    def _get_model_output(self, model_input):
        autoencoder, predictmodel = self._get_models()
        model_output = predictmodel(autoencoder.encoder(model_input))
        return model_output

    def _get_numpy_model_output(self, model_output):
        numpy_output = model_output.detach().numpy()
        numpy_output = numpy_output[0]
        return numpy_output

    def _predict_from_train_df(self, days=30, revenue=1.375):
        df = load_train_df()
        df, last_min_max_scaler = preprocessing_scale_df(df, None)
        trade_strategy = Direct2hStrategy()
        trade_strategy = MeanStrategy()
        trade_session = TradeSession(initial_value=400000, rake=0.00018)
        trade_skipped = 0
        for i in range(len(df) - 3):
            if trade_strategy.do_skip():
                continue
            if i < len(df) - (days * 24):
                continue
            feature_row, real_output = self._get_feature_row_and_real_output(df, i)
            model_input = self._get_model_input(feature_row)
            model_output = self._get_model_output(model_input)
            numpy_model_output = self._get_numpy_model_output(model_output)

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
        train_df = load_train_df()
        train_df, last_scaler = preprocessing_scale_df(train_df, None)
        currencies = self.take_currencies_from_df_columns(train_df)
        symbols = exchangify_pairs(currencies)
        kwargs = {
            "open_time__gt": datetime.datetime.today()
            - datetime.timedelta(days=ModelPerformanceLiveDataTestCase.DAYS),
            "symbol__in": symbols,
        }
        df = create_dataframe(Kline.objects.filter(**kwargs))
        df = df.fillna(0)
        df = resort_columns(train_df, df)
        df, last_min_max_scaler = preprocessing_scale_df(df, last_scaler)
        assert list(df.columns) == list(train_df.columns)

        trade_strategy = MeanStrategy()
        trade_session = TradeSession(initial_value=400000, rake=0.00018)
        trade_skipped = 0

        for i in range(len(df) - 3):
            if trade_strategy.do_skip():
                continue
            if i < len(df) - (30 * 24):
                continue
            feature_row, real_output = self._get_feature_row_and_real_output(df, i)
            model_input = self._get_model_input(feature_row)
            model_output = self._get_model_output(model_input)
            numpy_model_output = self._get_numpy_model_output(model_output)
            close_time = feature_row.name.to_pydatetime()

            if trade_strategy.do_trade(numpy_model_output):
                true_change_index = trade_strategy.pick_true_change_index(
                    numpy_model_output
                )
                test_true_change_percentage = real_output[true_change_index]

                currency = trade_strategy.pick_currency(numpy_model_output, currencies)
                current_kline = Kline.objects.get(
                    symbol=settings.EXCHANGE_FUTURES + "_" + currency,
                    close_time=close_time,
                )
                one_hour_kline = current_kline.get_kline_shifted_forward(2)
                true_change_percentage = (
                    one_hour_kline.close_price / current_kline.close_price
                ) - 1
                self.assertEqual(test_true_change_percentage, true_change_percentage)
                considered_side = trade_strategy.pick_side(numpy_model_output)
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

    def test_last_week(self):
        pass
