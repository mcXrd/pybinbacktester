from django.db import models
from binance_f.model.constant import OrderType, OrderSide, PositionSide, TimeInForce
from binance_f.model import CandlestickInterval
from django.conf import settings
from binance_f import RequestClient
import time
import numpy as np
from decimal import Decimal
from typing import Tuple
from apps.xgboost_models.models import BestRecommendation
from django.utils.timezone import now, timedelta


class TradeInterface:
    pass


class TradeInterfaceTest(TradeInterface):
    pass


class TradeInterfaceBinanceFutures(TradeInterface):
    def short(self, position):
        side = OrderSide.SELL
        return self.order(side, position)

    def long(self, position):
        side = OrderSide.BUY
        return self.order(side, position)

    def get_current_fee_currency_price(self):
        request_client = self.get_request_client()
        price_result = request_client.get_candlestick_data(
            symbol=settings.FEE_CURRENCY + "USDT",
            interval=CandlestickInterval.MIN1,
            startTime=None,
            endTime=None,
            limit=1,
        )
        return float(price_result[0].close)

    def get_current_close_price_and_std(self, symbol) -> Tuple[float, float, int]:
        request_client = self.get_request_client()
        price_result = request_client.get_candlestick_data(
            symbol=symbol,
            interval=CandlestickInterval.MIN1,
            startTime=None,
            endTime=None,
            limit=2,
        )

        std = np.std(
            [float(price_result[0].close), float(price_result[-1].close)], ddof=1
        )
        close_price = price_result[-1].close
        round_to_places = len(str(close_price)) - 1
        return float(close_price), std, round_to_places

    def get_account_information(self):
        request_client = self.get_request_client()
        result = request_client.get_account_information()
        return result

    def change_initial_leverage(self, symbol, leverage):
        request_client = self.get_request_client()
        result = request_client.change_initial_leverage(symbol, leverage)
        return result

    def get_current_fee_currency_balance(self):
        result = self.get_account_information()
        balance = None
        for asset in result.assets:
            if asset.asset == settings.FEE_CURRENCY:
                balance = asset.marginBalance
        if balance is None:
            raise Exception("{} asset not found".format(settings.FEE_CURRENCY))
        return balance

    @staticmethod
    def add_noise_to_price(price, std):
        mu, sigma = 0, std  # mean and standard deviation
        s = np.random.normal(mu, sigma, 1)
        noise = s[0]
        price = price + noise
        return price

    def wait_for_orders_to_fill(self, request_client, position, order_id, symbol):
        iters_to_wait = 5
        iters = 0
        while True:
            if not position.alive:
                raise Exception("Executing position which is not alive")
            iters += 1
            time.sleep(3)
            last_order = request_client.get_all_orders(orderId=order_id, symbol=symbol)[
                0
            ]
            if last_order.status == "EXPIRED":
                return False

            res = request_client.get_open_orders()
            if len(res) == 0:
                return True
            if iters > iters_to_wait:
                for order in res:
                    log = PositionLog.objects.create(
                        position=position, name="Order not filled"
                    )
                    log.log_order(order)

                res = request_client.cancel_all_orders(symbol=position.base_symbol)
                PositionLog.objects.create(
                    position=position,
                    name="request_client.cancel_all_orders() ; code {}".format(
                        res.code
                    ),
                    log_message=res.msg,
                )
                time.sleep(1)
                return False

    def get_request_client(self):
        return RequestClient(
            api_key=settings.BINANCE_API_KEY, secret_key=settings.BINANCE_SECRET_KEY
        )

    def order(self, side, position):
        request_client = self.get_request_client()
        retries = -1
        noised_price = None
        last_order = None
        last_noised_price = None
        while True:
            if not position.alive:
                raise Exception("Executing position which is not alive")
            retries += 1
            price, std, round_to_places = self.get_current_close_price_and_std(
                position.base_symbol
            )
            if retries > 50:
                noised_price = self.add_noise_to_price(price, std)
            else:
                noised_price = price
            if position.base_symbol == "ADAUSDT":
                round_to_places = 3
            elif position.base_symbol == "ETHUSDT":
                round_to_places = 2
            else:
                raise Exception("Nondefined pair")
            noised_price = round(Decimal(noised_price), round_to_places)
            msg = "Price: {} ; Noised price {} ; STD: {} ; retries: {} ;".format(
                price, noised_price, std, retries
            )
            PositionLog.objects.create(
                position=position, name="Order attempt", log_message=msg
            )
            """
            !!! extremely important info
            To send a POST ONLY order,

            In Spot: set the order type parameter ( type) to LIMIT_MAKER
            In Futures : set the time in force parameter (timeInForce) to GTX
            """
            try:
                order = request_client.post_order(
                    symbol=position.base_symbol,
                    side=side,
                    ordertype=OrderType.LIMIT,
                    price=noised_price,
                    quantity=str(position.quantity),
                    timeInForce=TimeInForce.GTX,  # extremely important, otherwise maker order is not sure
                )
            except Exception as e:
                insufficient_margin_str = "argin is insufficient"
                if insufficient_margin_str in str(e):
                    if retries == 0:
                        raise Exception("Margin is insufficient. {}".format(str(e)))
                    if retries > 0:
                        PositionLog.objects.create(
                            position=position,
                            name="Race condition in order - Order should be already posted",
                            log_message=str(e),
                        )
                        log = PositionLog.objects.create(
                            position=position, name="Order posted"
                        )
                        log.log_order(last_order)
                        return last_noised_price

                PositionLog.objects.create(
                    position=position, name="Order post failed", log_message=str(e)
                )
                time.sleep(1)
                continue

            log = PositionLog.objects.create(position=position, name="Order posted")
            log.log_order(order)
            if self.wait_for_orders_to_fill(
                request_client, position, order.orderId, position.base_symbol
            ):  # order was filled
                break

            last_order = order
            last_noised_price = noised_price

        return noised_price


class PositionLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    position = models.ForeignKey(
        "predictive_models.Position", related_name="logs", on_delete=models.CASCADE
    )
    log_message = models.TextField(null=True, blank=True)
    log_json = models.JSONField(null=True, blank=True)
    name = models.CharField(null=True, blank=True, max_length=200)

    def log_order(self, order):
        log = {}
        log["orderId"] = order.orderId
        log["status"] = order.status
        log["symbol"] = order.symbol
        log["side"] = order.side
        log["price"] = order.price
        log["updateTime"] = order.updateTime
        self.log_json = log
        self.save()


class CronLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    log_message = models.TextField(null=True, blank=True)
    log_json = models.JSONField(null=True, blank=True)
    name = models.CharField(null=True, blank=True, max_length=200)


class AlertLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    log_message = models.TextField(null=True, blank=True)
    log_json = models.JSONField(null=True, blank=True)
    name = models.CharField(null=True, blank=True, max_length=200)


class Position(models.Model):

    """
    result = request_client.post_order(symbol="BTCUSDT",
     side=OrderSide.SELL, ordertype=OrderType.LIMIT,
      stopPrice=8000.1, closePosition=True, positionSide=PositionSide.LONG)
    """

    SHORT = "SHORT"
    LONG = "LONG"
    SIDE_CHOICES = (
        (SHORT, "Short"),
        (LONG, "Long"),
    )

    symbol = models.CharField(null=False, blank=False, max_length=20)
    side = models.CharField(
        null=False, blank=False, max_length=20, choices=SIDE_CHOICES
    )
    liquidation_being_processed = models.BooleanField(default=False)
    liquidated = models.BooleanField(default=False)
    start_to_open = models.DateTimeField(null=True, blank=True)
    open_finished = models.DateTimeField(null=True, blank=True)
    liquidate_at = models.DateTimeField(null=False, blank=False)
    start_to_liquidate = models.DateTimeField(null=True, blank=True)
    liquidation_finished = models.DateTimeField(null=True, blank=True)
    open_price = models.FloatField(null=True, blank=True)
    liquidated_price = models.FloatField(null=True, blank=True)
    quantity = models.FloatField(null=False, blank=False)

    initial_fee_balance = models.FloatField(null=True, blank=True)
    after_fee_balance = models.FloatField(null=True, blank=True)
    open_fee_currency_price = models.FloatField(null=True, blank=True)

    liq_initial_fee_balance = models.FloatField(null=True, blank=True)
    liq_after_fee_balance = models.FloatField(null=True, blank=True)
    liq_fee_currency_price = models.FloatField(null=True, blank=True)

    alive = models.BooleanField(default=True)
    fee_tier = models.CharField(null=True, blank=True, max_length=100)

    def extend_liquidation(self):
        self.liquidate_at = self.liquidate_at + timedelta(
            minutes=settings.POSITION_OPEN_FOR_MINUTES
        )
        self.save()

    @staticmethod
    def reverse_side(side):
        if side == Position.SHORT:
            return Position.LONG
        return Position.SHORT

    def liquidate(self, trade_interface: TradeInterface, ignore_recommendation=False):
        br = BestRecommendation.objects.last()
        if not br.is_fresh():
            log = PositionLog.objects.create(
                position=self, name="Reco not fresh - returning liquidate"
            )
            return
        if not ignore_recommendation and br.side == BestRecommendation.PASS:
            log = PositionLog.objects.create(
                position=self, name="Reco is PASS - extending liquidation"
            )
            self.extend_liquidation()
            return
        if (
            not ignore_recommendation
            and br.symbol == self.coin
            and br.side == self.side
        ):
            log = PositionLog.objects.create(
                position=self,
                name="Reco is same as this position already is - extending liquidation",
            )
            self.extend_liquidation()
            return

        if not self.open_finished:
            raise Exception("Position is not even opened yet.")
        if self.liquidated:
            raise Exception("Already liquidated.")
        self.liquidation_being_processed = True
        self.save()
        try:
            self.start_to_liquidate = now()
            side = self.reverse_side(self.side)
            self.liq_initial_fee_balance = (
                trade_interface.get_current_fee_currency_balance()
            )
            self.liq_fee_currency_price = (
                trade_interface.get_current_fee_currency_price()
            )
            trade_price = self._trade(trade_interface, side)
            self.liquidated_price = trade_price
            self.liq_after_fee_balance = (
                trade_interface.get_current_fee_currency_balance()
            )
            self.liquidation_finished = now()
            self.liquidated = True
        finally:
            self.save()

    def open(self, trade_interface: TradeInterface):
        if self.open_finished:
            raise Exception("Open already finished.")
        try:
            self.start_to_open = now()
            self.initial_fee_balance = (
                trade_interface.get_current_fee_currency_balance()
            )
            self.open_fee_currency_price = (
                trade_interface.get_current_fee_currency_price()
            )
            trade_price = self._trade(trade_interface, self.side)
            self.open_price = trade_price
            self.after_fee_balance = trade_interface.get_current_fee_currency_balance()
            self.open_finished = now()
        finally:
            self.save()

    def _trade(self, trade_interface: TradeInterface, side: str):
        if side == Position.LONG:
            trade_price = trade_interface.long(self)
        elif side == Position.SHORT:
            trade_price = trade_interface.short(self)
        else:
            raise Exception("Invalid side")
        return trade_price

    @property
    def base_symbol(self):
        return self.symbol

    @property
    def coin(self):
        return self.base_symbol[:-4]
