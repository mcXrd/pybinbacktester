from django.db import models


class TradeInterface:
    pass


class TradeInterfaceTrade(TradeInterface):
    pass


class TradeInterfaceBinanceFutures(TradeInterface):
    pass


class Position(models.Model):
    symbol = models.CharField(null=False, blank=False, max_length=20)
    side = models.CharField(null=False, blank=False, max_length=20)
    liquidated = models.BooleanField(default=False)
    open_at = models.DateTimeField()
    liquidated_at = models.DateTimeField()
    open_price = models.FloatField()
    liquidated_price = models.FloatField()
    rake = models.FloatField()  # ration
    rake_amount = models.FloatField()  # total value

    def liquidate(self, trade_interface: TradeInterface):
        pass

    def open(self, trade_interface: TradeInterface):
        pass
