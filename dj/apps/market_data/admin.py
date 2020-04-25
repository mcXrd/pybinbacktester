from django.contrib import admin

from .models import Kline


class KlineAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'open_time', 'close_time', 'open_price', 'close_price')
    list_filter = list_display


admin.site.register(Kline, KlineAdmin)
