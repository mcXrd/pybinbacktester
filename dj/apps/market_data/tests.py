from django.test import TestCase
from apps.market_data.generate_market_data_hdf_utils import create_base_dataframe
from apps.market_data.sync_kline_utils import (
    insert_klines,
    remove_too_old_klines,
    EXCHANGE_FUTURES,
    EXCHANGE_SPOT,
)
from apps.market_data.models import Kline
from apps.market_data.market_depth_utils import list_symbol_pairs, get_trades

TIME_INTERVAL = "360 days ago UTC"

TEST_KLINES = [
    [
        1544628120000,
        "0.02612900",
        "0.02612900",
        "0.02612900",
        "0.02612900",
        "0.04600000",
        1544628179999,
        "0.00120193",
        1,
        "0.04600000",
        "0.00120193",
        "0",
    ],
    [
        1554628120000,
        "0.02612900",
        "0.02612900",
        "0.02612900",
        "0.02612900",
        "0.04600000",
        1554628179999,
        "0.00120193",
        1,
        "0.04600000",
        "0.00120193",
        "0",
    ],
]

VERY_OLD_TEST_KLINES = [
    [
        44628120000,
        "0.02612900",
        "0.02612900",
        "0.02612900",
        "0.02612900",
        "0.04600000",
        44628179999,
        "0.00120193",
        1,
        "0.04600000",
        "0.00120193",
        "0",
    ],
    [
        54628120000,
        "0.02612900",
        "0.02612900",
        "0.02612900",
        "0.02612900",
        "0.04600000",
        54628179999,
        "0.00120193",
        1,
        "0.04600000",
        "0.00120193",
        "0",
    ],
]


def get_klines_mock_one_kline(symbol, time_interval):
    return [TEST_KLINES[0]]


def get_klines_mock_two_klines(symbol, time_interval):
    return TEST_KLINES


def get_klines_mock_old_ones(symbol, time_interval):
    return VERY_OLD_TEST_KLINES


class SyncKlinesTestCase(TestCase):
    def test_insert_klines_dont_add_duplicates(self):
        assert Kline.objects.all().count() == 0
        insert_klines("ethbtc", get_klines_mock_one_kline, EXCHANGE_SPOT, TIME_INTERVAL)
        insert_klines("ethada", get_klines_mock_one_kline, EXCHANGE_SPOT, TIME_INTERVAL)
        assert Kline.objects.all().count() == 2
        insert_klines("ethbtc", get_klines_mock_one_kline, EXCHANGE_SPOT, TIME_INTERVAL)
        insert_klines("ethada", get_klines_mock_one_kline, EXCHANGE_SPOT, TIME_INTERVAL)
        assert Kline.objects.all().count() == 2

    def test_insert_klines_dont_add_duplicates_futures(self):
        assert Kline.objects.all().count() == 0
        insert_klines(
            "ethbtc", get_klines_mock_one_kline, EXCHANGE_FUTURES, TIME_INTERVAL
        )
        insert_klines(
            "ethada", get_klines_mock_one_kline, EXCHANGE_FUTURES, TIME_INTERVAL
        )
        assert Kline.objects.all().count() == 2
        insert_klines(
            "ethbtc", get_klines_mock_one_kline, EXCHANGE_FUTURES, TIME_INTERVAL
        )
        insert_klines(
            "ethada", get_klines_mock_one_kline, EXCHANGE_FUTURES, TIME_INTERVAL
        )
        self.assertEqual(Kline.objects.all().count(), 2)

    def test_insert_klines_can_update_newer_klines(self):
        assert Kline.objects.all().count() == 0
        insert_klines("ethbtc", get_klines_mock_one_kline, EXCHANGE_SPOT, TIME_INTERVAL)
        insert_klines("ethada", get_klines_mock_one_kline, EXCHANGE_SPOT, TIME_INTERVAL)
        assert Kline.objects.all().count() == 2
        insert_klines(
            "ethbtc", get_klines_mock_two_klines, EXCHANGE_SPOT, TIME_INTERVAL
        )
        insert_klines(
            "ethada", get_klines_mock_two_klines, EXCHANGE_SPOT, TIME_INTERVAL
        )
        assert Kline.objects.all().count() == 4

    def test_insert_klines_can_update_newer_klines_futures(self):
        assert Kline.objects.all().count() == 0
        insert_klines(
            "ethbtc", get_klines_mock_one_kline, EXCHANGE_FUTURES, TIME_INTERVAL
        )
        insert_klines(
            "ethada", get_klines_mock_one_kline, EXCHANGE_FUTURES, TIME_INTERVAL
        )
        assert Kline.objects.all().count() == 2
        insert_klines(
            "ethbtc", get_klines_mock_two_klines, EXCHANGE_FUTURES, TIME_INTERVAL
        )
        insert_klines(
            "ethada", get_klines_mock_two_klines, EXCHANGE_FUTURES, TIME_INTERVAL
        )
        assert Kline.objects.all().count() == 4

    def test_remove_too_old_klines(self):
        assert Kline.objects.all().count() == 0
        insert_klines("ethbtc", get_klines_mock_one_kline, EXCHANGE_SPOT, TIME_INTERVAL)
        insert_klines("ethada", get_klines_mock_one_kline, EXCHANGE_SPOT, TIME_INTERVAL)
        assert Kline.objects.all().count() == 2
        remove_too_old_klines()
        assert Kline.objects.all().count() == 0


class GenerateMarketDataHDFTestCase(TestCase):
    def test_close_price_is_found_in_hdf(self):
        test_symbol = "ethbtc"
        insert_klines(
            test_symbol, get_klines_mock_one_kline, EXCHANGE_SPOT, TIME_INTERVAL
        )
        df = create_base_dataframe(Kline.objects.all())
        close_price = TEST_KLINES[0][4]
        assert float(df[EXCHANGE_SPOT + "_ethbtc_close_price"][0]) == float(close_price)


class MarketDepthTest(TestCase):
    def test_pairs_listing(self):
        self.assertLess(0, len(list_symbol_pairs()))

    def test_get_trades(self):
        res = get_trades(list_symbol_pairs()[0])
        raise Exception(res)
