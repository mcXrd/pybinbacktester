from django.db import models
import datetime

"""
all tickers example:
[{'symbol': 'ETHBTC', 'price': '0.02601200'}, {'symbol': 'LTCBTC', 'price': '0.00707800'} ....
"""


class Kline(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    symbol = models.CharField(null=False, blank=False, max_length=20)
    open_time = models.DateTimeField(null=False, blank=False)
    open_price = models.FloatField(null=False, blank=False)
    high_price = models.FloatField(null=False, blank=False)
    low_price = models.FloatField(null=False, blank=False)
    close_price = models.FloatField(null=False, blank=False)
    volume = models.FloatField(null=False, blank=False)
    close_time = models.DateTimeField(null=False, blank=False)
    quota_asset_volume = models.FloatField(null=False, blank=False)
    number_of_trades = models.IntegerField(null=False, blank=False)
    taker_buy_base_asset_volume = models.FloatField(null=False, blank=False)
    taker_buy_quote_asset_volume = models.FloatField(null=False, blank=False)
    ignore = models.FloatField(null=False, blank=False, default=0.0)

    def get_kline_shifted_forward(self, hours: int):
        assert hours and isinstance(hours, int)
        k = Kline.objects.get(
            symbol=self.symbol,
            close_time=self.close_time + datetime.timedelta(hours=hours),
        )
        return k
