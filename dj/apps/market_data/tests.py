import mock
from django.test import TestCase
from apps.market_data.management.commands.generate_market_data_hdf import create_base_dataframe
from apps.market_data.management.commands.sync_klines import insert_klines, remove_too_old_klines
from apps.market_data.models import Kline

TEST_KLINES = [[
    1544628120000,
    '0.02612900',
    '0.02612900',
    '0.02612900',
    '0.02612900',
    '0.04600000',
    1544628179999,
    '0.00120193',
    1,
    '0.04600000',
    '0.00120193',
    '0'],
    [
        1554628120000,
        '0.02612900',
        '0.02612900',
        '0.02612900',
        '0.02612900',
        '0.04600000',
        1554628179999,
        '0.00120193',
        1,
        '0.04600000',
        '0.00120193',
        '0']]

VERY_OLD_TEST_KLINES = [[
    44628120000,
    '0.02612900',
    '0.02612900',
    '0.02612900',
    '0.02612900',
    '0.04600000',
    44628179999,
    '0.00120193',
    1,
    '0.04600000',
    '0.00120193',
    '0'],
    [
        54628120000,
        '0.02612900',
        '0.02612900',
        '0.02612900',
        '0.02612900',
        '0.04600000',
        54628179999,
        '0.00120193',
        1,
        '0.04600000',
        '0.00120193',
        '0']]


def get_klines_mock_one_kline(symbol):
    return [TEST_KLINES[0]]


def get_klines_mock_two_klines(symbol):
    return TEST_KLINES


def get_klines_mock_old_ones(symbol):
    return VERY_OLD_TEST_KLINES


class SyncKlinesTestCase(TestCase):

    @mock.patch("apps.market_data.management.commands.sync_klines.get_klines", get_klines_mock_one_kline)
    def test_insert_klines_dont_add_duplicates(self):
        assert Kline.objects.all().count() == 0
        insert_klines('ethbtc')
        insert_klines('ethada')
        assert Kline.objects.all().count() == 2
        insert_klines('ethbtc')
        insert_klines('ethada')
        assert Kline.objects.all().count() == 2

    def test_insert_klines_can_update_newer_klines(self):
        with mock.patch("apps.market_data.management.commands.sync_klines.get_klines", get_klines_mock_one_kline):
            assert Kline.objects.all().count() == 0
            insert_klines('ethbtc')
            insert_klines('ethada')
            assert Kline.objects.all().count() == 2
        with mock.patch("apps.market_data.management.commands.sync_klines.get_klines", get_klines_mock_two_klines):
            insert_klines('ethbtc')
            insert_klines('ethada')
            assert Kline.objects.all().count() == 4

    def test_insert_klines_dont_add_too_old_klines(self):
        with mock.patch("apps.market_data.management.commands.sync_klines.get_klines", get_klines_mock_one_kline):
            assert Kline.objects.all().count() == 0
            insert_klines('ethbtc')
            insert_klines('ethada')
            assert Kline.objects.all().count() == 2
        with mock.patch("apps.market_data.management.commands.sync_klines.get_klines", get_klines_mock_old_ones):
            insert_klines('ethbtc')
            insert_klines('ethada')
            assert Kline.objects.all().count() == 2

    @mock.patch("apps.market_data.management.commands.sync_klines.get_klines", get_klines_mock_one_kline)
    def test_remove_too_old_klines(self):
        assert Kline.objects.all().count() == 0
        insert_klines('ethbtc')
        insert_klines('ethada')
        assert Kline.objects.all().count() == 2
        remove_too_old_klines()
        assert Kline.objects.all().count() == 0


class GenerateMarketDataHDFTestCase(TestCase):

    @mock.patch("apps.market_data.management.commands.sync_klines.get_klines", get_klines_mock_one_kline)
    def test_close_price_is_found_in_hdf(self):
        test_symbol = 'ethbtc'
        insert_klines(test_symbol)
        df = create_base_dataframe(Kline.objects.all())
        close_price = TEST_KLINES[0][4]
        assert float(df['ethbtc_close_price'][0]) == float(close_price)
