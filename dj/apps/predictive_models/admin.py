from django.contrib import admin
from apps.predictive_models.models import PositionLog, Position, CronLog, AlertLog


class PositionLogInline(admin.TabularInline):
    model = PositionLog


class PositionAdmin(admin.ModelAdmin):
    inlines = [
        PositionLogInline,
    ]
    list_display = (
        "symbol",
        "side",
        "open_finished",
        "liquidate_at",
        "liquidated",
        "quantity",
        "open_price",
    )


class PositionLogAdmin(admin.ModelAdmin):
    readonly_fields = ["created_at"]
    list_display = ("name", "created_at", "position")


class CronLogAdmin(admin.ModelAdmin):
    readonly_fields = ["created_at"]
    list_display = (
        "name",
        "created_at",
    )


admin.site.register(PositionLog, PositionLogAdmin)
admin.site.register(Position, PositionAdmin)
admin.site.register(CronLog, CronLogAdmin)
admin.site.register(AlertLog, CronLogAdmin)
