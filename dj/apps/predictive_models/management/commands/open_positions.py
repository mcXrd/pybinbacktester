import logging

from django.core.management.base import BaseCommand
from apps.predictive_models.models import Position
from django.core.management import call_command
from apps.predictive_models.live_trade_utils import create_live_df
from apps.predictive_models.live_trade_utils import SAFE_DAY_ADD
from django.utils.timezone import now, timedelta
from apps.predictive_models.models import CronLog, AlertLog
from apps.predictive_models.jane1.trade_strategy import MeanStrategy
from apps.predictive_models.live_trade_utils import (
    get_feature_row_and_real_output,
    resolve_trade,
    count_quantity,
)
from apps.predictive_models.live_trade_utils import NoTradeException
from apps.predictive_models.models import TradeInterfaceBinanceFutures
from apps.xgboost_models.models import BestRecommendation
from django.conf import settings

logger = logging.getLogger(__name__)


def main():
    positions_qs = Position.objects.filter(liquidated=False)
    open_positions_count = positions_qs.count()

    if open_positions_count >= settings.OPEN_POSITIONS_COUNT_LIMIT:
        return

    if open_positions_count:
        last_open_position = Position.objects.filter(liquidated=False).last()
        seconds_from_last_position = (
            now() - last_open_position.open_finished
        ).total_seconds()
        minutes_from_last_position = seconds_from_last_position / 60
        limit_to_open_again = (
            settings.POSITION_OPEN_FOR_MINUTES / settings.OPEN_POSITIONS_COUNT_LIMIT
        )

        if minutes_from_last_position < limit_to_open_again:
            return

    try:
        br = BestRecommendation.objects.last()
        if not br.is_fresh():
            AlertLog.objects.create(
                name="Skipping open because br is not fresh",
                log_message="",
            )
            return

        if br.side == BestRecommendation.PASS:
            return

        if br.side == BestRecommendation.LONG:
            side = Position.LONG
        elif br.side == BestRecommendation.SHORT:
            side = Position.SHORT
        symbol = br.symbol + "USDT"
        trade_interface = TradeInterfaceBinanceFutures()
        trade_interface.change_initial_leverage(symbol, 1)
        ai = trade_interface.get_account_information()
        fee_tier = ai.feeTier
        usdt_asset = None
        for asset in ai.assets:
            if asset.asset == "USDT":
                usdt_asset = asset
        total_usdt_amount = usdt_asset.marginBalance

        price, std, round_to_places = trade_interface.get_current_close_price_and_std(
            symbol
        )
        factor_for_usdt_amount_based_on_open_positions = (
            1 / settings.OPEN_POSITIONS_COUNT_LIMIT
        )
        considered_quantity = count_quantity(
            symbol,
            price,
            total_usdt_amount * factor_for_usdt_amount_based_on_open_positions,
        )

        reverse_qs = Position.objects.filter(
            symbol=symbol,
            side=Position.reverse_side(side),
            liquidated=False,
            open_finished__isnull=False,
        )
        if (
            reverse_qs.exists()
        ):  # this will avoid multiple fee charges - it will close reverse orders - and
            # it will just create new ones with small fake amount instead (timing of creation of new positions
            # is tied to existing positions - thats why these needs to be created)
            temp_pos = reverse_qs.last()
            temp_pos.liquidate(trade_interface)

            fake_quantity = settings.USDT_FUTURES_MINIMAL_TRADE_AMOUNT[
                temp_pos.base_symbol
            ]
            considered_quantity = fake_quantity
            position = Position.objects.create(
                symbol=symbol,
                side=side,
                liquidate_at=temp_pos.liquidate_at,
                quantity=str(considered_quantity),
                fee_tier=str(fee_tier),
            )
            position.open(trade_interface)

        position = Position.objects.create(
            symbol=symbol,
            side=side,
            liquidate_at=now() + timedelta(minutes=settings.POSITION_OPEN_FOR_MINUTES),
            quantity=str(considered_quantity),
            fee_tier=str(fee_tier),
        )
        position.open(trade_interface)
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
