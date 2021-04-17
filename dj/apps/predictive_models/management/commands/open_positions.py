import logging

from django.core.management.base import BaseCommand
from apps.predictive_models.models import Position
from apps.predictive_models.models import TradeInterfaceBinanceFutures
from apps.predictive_models.jane1.load_models import preprocessing_scale_df
from apps.predictive_models.jane1.load_models import load_train_df
from apps.predictive_models.jane1.load_models import (
    take_currencies_from_df_columns,
)
from apps.market_data.generate_market_data_hdf_utils import exchangify_pairs
from apps.market_data.generate_market_data_hdf_utils import (
    exchangify_pairs,
    create_dataframe,
)
from apps.predictive_models.jane1.load_models import resort_columns
from django.utils import timezone
from apps.market_data.models import Kline
from apps.predictive_models.jane1.trade_strategy import MeanStrategy, Direct2hStrategy
from apps.predictive_models.models import CronLog


logger = logging.getLogger(__name__)


def main():
    positions_qs = Position.objects.filter(liquidated=False)
    if positions_qs.exists():
        return

    trade_interface = TradeInterfaceBinanceFutures()
    ai = trade_interface.get_account_information()
    usdt_assset = None
    for asset in ai.assets:
        if asset.asset == "USDT":
            usdt_assset = asset

    train_df = load_train_df()
    train_df, last_scaler = preprocessing_scale_df(train_df, None)
    currencies = take_currencies_from_df_columns(train_df)
    symbols = exchangify_pairs(currencies)
    kwargs = {
        "open_time__gt": timezone.now() - timezone.timedelta(days=15),
        "symbol__in": symbols,
    }
    df = create_dataframe(Kline.objects.filter(**kwargs))
    df = df.fillna(0)
    df = resort_columns(train_df, df)
    df, last_min_max_scaler = preprocessing_scale_df(df, last_scaler)
    assert list(df.columns) == list(train_df.columns)

    trade_strategy = MeanStrategy()


    currency_to_trade = "BTCUSDT"
    side = "LONG"

    usdt_amount = usdt_assset.marginBalance


class Command(BaseCommand):
    help = "Liquidate positions"

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **kwargs):
        main()
