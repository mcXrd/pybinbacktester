import logging

from django.core.management.base import BaseCommand
from apps.predictive_models.models import Position
from django.core.management import call_command
from apps.predictive_models.live_trade_utils import create_live_df
from apps.predictive_models.live_trade_utils import SAFE_DAY_ADD
from django.utils.timezone import now, timedelta
from apps.predictive_models.models import CronLog
from apps.predictive_models.jane1.trade_strategy import MeanStrategy
from apps.predictive_models.live_trade_utils import (
    get_feature_row_and_real_output,
    resolve_trade,
    count_quantity,
)
from apps.predictive_models.live_trade_utils import NoTradeException
from apps.predictive_models.models import TradeInterfaceBinanceFutures
from django.conf import settings

logger = logging.getLogger(__name__)

LIMIT_TO_START_TRADE = 8  # in minutes - 60 is effective maximum


def main():
    positions_qs = Position.objects.filter(liquidated=False)
    if positions_qs.exists():
        return
    now_time = now()
    now_minutes = now_time.minute
    if now_minutes == 0 or now_minutes > LIMIT_TO_START_TRADE:
        return

    call_command("remove_klines")

    call_command(
        "sync_klines",
        max_workers=1,
        time_interval="{} days ago UTC".format(SAFE_DAY_ADD + 1),
    )

    df, currencies = create_live_df(1, live=True)
    df_index = len(df) - 2
    feature_row, real_output = get_feature_row_and_real_output(
        df, df_index
    )  # last one is not complete
    close_time = feature_row.name.to_pydatetime()
    assert str(close_time.tzinfo) == "UTC"
    assert str(now().tzinfo) == "UTC"
    total_seconds_from_last_complete_close = (now() - close_time).total_seconds()
    total_minutes_from_last_complete_close = int(
        total_seconds_from_last_complete_close / 60
    )
    if total_minutes_from_last_complete_close > LIMIT_TO_START_TRADE:
        CronLog.objects.create(
            name="It probably took to long to open position (invalid timeframe)",
            log_message="{} minutes from last complete close".format(
                str(total_minutes_from_last_complete_close)
            ),
        )
        return
    trade_strategy = MeanStrategy()

    try:
        symbol, side = resolve_trade(df, df_index, trade_strategy, currencies)
        side = Position.SHORT if side == -1 else Position.LONG

        trade_interface = TradeInterfaceBinanceFutures()
        trade_interface.change_initial_leverage(symbol, 1)
        ai = trade_interface.get_account_information()
        fee_tier = ai.feeTier
        usdt_assset = None
        for asset in ai.assets:
            if asset.asset == "USDT":
                usdt_assset = asset
        usdt_amount = usdt_assset.marginBalance
        price, std = trade_interface.get_current_close_price_and_std(symbol)
        considered_quantity = count_quantity(symbol, price, usdt_amount)
        positon = Position.objects.create(
            symbol="usdtfutures_" + symbol,
            side=side,
            liquidate_at=now() + timedelta(minutes=118),
            quantity=str(considered_quantity),
            fee_tier=str(fee_tier),
        )
        positon.open(trade_interface)
    except NoTradeException as e:
        CronLog.objects.create(
            name="Skipping trade oportunity",
            log_message=str(e),
        )


class Command(BaseCommand):
    help = "Liquidate positions"

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **kwargs):
        main()
