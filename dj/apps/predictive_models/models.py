from django.db import models
from binance_f.model.constant import OrderType, OrderSide, PositionSide, TimeInForce
from binance_f.model import CandlestickInterval
from django.contrib.postgres.fields import JSONField
from django.conf import settings
from binance_f import RequestClient
import time


class TradeInterface:
    pass


class TradeInterfaceTest(TradeInterface):
    pass


class TradeInterfaceBinanceFutures(TradeInterface):
    def short(self, position):
        side = OrderSide.SELL
        self.order(side, position)

    def long(self, position):
        side = OrderSide.BUY
        self.order(side, position)

    def get_current_close_price(self, request_client, symbol):
        price_result = request_client.get_candlestick_data(
            symbol=symbol,
            interval=CandlestickInterval.MIN1,
            startTime=None,
            endTime=None,
            limit=1,
        )
        close_price = price_result[0].close
        return float(close_price)

    @staticmethod
    def add_noise_to_price(price):
        price

    def wait_for_orders_to_fill(self, request_client, position):
        iters_to_wait = 30
        iters = 0
        while True:
            iters += 1
            time.sleep(0.2)
            res = request_client.get_open_orders()
            if len(res) == 0:
                return True
            if iters > iters_to_wait:
                for order in res:
                    log = PositionLog.objects.create(
                        position=position, name="Orders not filled."
                    )
                    log.log_order(order)

                res = request_client.cancel_all_orders(symbol=position.base_symbol)
                return False

    def order(self, side, position):
        request_client = RequestClient(
            api_key=settings.BINANCE_API_KEY, secret_key=settings.BINANCE_SECRET_KEY
        )
        while True:
            price = self.get_current_close_price(request_client, position.base_symbol)
            price = self.add_noise_to_price(price)
            order = request_client.post_order(
                symbol=position.base_symbol,
                side=side,
                ordertype=OrderType.LIMIT,
                price=round(float(price), 6),
                quantity=12,
                timeInForce=TimeInForce.GTC,
            )
            log = PositionLog.objects.create(position=position, name="Order posted.")
            log.log_order(order)
            if self.wait_for_orders_to_fill():  # order was filled
                break


# request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)
# request_client.get_candlestick_data


class PositionLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    position = models.ForeignKey("predictive_models.Position", related_name="logs")
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

    symbol = models.CharField(null=False, blank=False, max_length=20)
    side = models.CharField(null=False, blank=False, max_length=20)
    liquidated = models.BooleanField(default=False)
    open_at = models.DateTimeField()
    liquidate_at = models.DateTimeField(null=False, blank=False)
    liquidated_at = models.DateTimeField()
    open_price = models.FloatField()
    liquidated_price = models.FloatField()
    rake = models.FloatField()  # ratio
    rake_amount = models.FloatField()  # total value

    def liquidate(self, trade_interface: TradeInterface):
        pass

    def open(self, trade_interface: TradeInterface):
        pass

    @property
    def base_symbol(self):
        return self.symbol.split("usdtfutures_")[1]
