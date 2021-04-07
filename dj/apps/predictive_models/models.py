from django.db import models
from binance_f.model.constant import OrderType, OrderSide, PositionSide, TimeInForce
from binance_f.model import CandlestickInterval
from django.conf import settings
from binance_f import RequestClient
import time
import numpy as np
from django.utils.timezone import now


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

    def get_current_close_price_and_std(self, request_client, symbol):
        price_result = request_client.get_candlestick_data(
            symbol=symbol,
            interval=CandlestickInterval.MIN1,
            startTime=None,
            endTime=None,
            limit=2,
        )

        std = np.std([price_result[0], price_result[-1]], ddof=1)
        close_price = price_result[-1].close
        return float(close_price), std

    def get_current_fee_currency_balance(self):
        request_client = self.get_request_client()
        result = request_client.get_account_information()
        balance = None
        for asset in result.assets:
            if asset.asset == "BNB":
                balance = asset.marginBalance
        if balance is None:
            raise Exception("BNB asset not found")
        return balance

    @staticmethod
    def add_noise_to_price(price, std):
        mu, sigma = 0, std  # mean and standard deviation
        s = np.random.normal(mu, sigma, 1)
        noise = s[0]
        return price + noise

    def wait_for_orders_to_fill(self, request_client, position):
        iters_to_wait = 5
        iters = 0
        while True:
            iters += 1
            time.sleep(1)
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
        while True:
            retries += 1
            price, std = self.get_current_close_price_and_std(
                request_client, position.base_symbol
            )
            noised_price = self.add_noise_to_price(price, std)
            msg = "Price: {} ; Noised price {} ; STD: {} ; retries: {} ;".format(
                price, noised_price, std, retries
            )
            PositionLog.objects.create(
                position=position, name="Order attempt", log_message=msg
            )
            try:
                order = request_client.post_order(
                    symbol=position.base_symbol,
                    side=side,
                    ordertype=OrderType.LIMIT,
                    price=round(float(noised_price), 5),
                    quantity=round(float(position.quantity), 5),
                    timeInForce=TimeInForce.GTC,
                )
            except Exception as e:
                PositionLog.objects.create(
                    position=position, name="Order post failed", log_message=str(e)
                )
                time.sleep(retries + 1)
                continue

            log = PositionLog.objects.create(position=position, name="Order posted")
            log.log_order(order)
            if self.wait_for_orders_to_fill():  # order was filled
                break
        return noised_price


class PositionLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    position = models.ForeignKey("predictive_models.Position", related_name="logs", on_delete=models.CASCADE)
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

    @staticmethod
    def reverse_side(side):
        if side == Position.SHORT:
            return Position.LONG
        return Position.SHORT

    def liquidate(self, trade_interface: TradeInterface):
        try:
            self.start_to_liquidate = now()
            side = self.reverse_side(self.side)
            trade_price = self._trade(trade_interface, side)
            self.liquidated_price = trade_price
            self.liquidation_finished = now()
        finally:
            self.save()

    def open(self, trade_interface: TradeInterface, side: str):
        try:
            self.start_to_open = now()
            self.initial_fee_balance = (
                trade_interface.get_current_fee_currency_balance()
            )
            trade_price = self._trade(trade_interface, side)
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
        return self.symbol.split("usdtfutures_")[1]
