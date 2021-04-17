from django.contrib import admin
from apps.predictive_models.models import PositionLog, Position
from apps.predictive_models.models import CronLog


class PositionLogInline(admin.TabularInline):
    model = PositionLog


class PositionAdmin(admin.ModelAdmin):
    inlines = [
        PositionLogInline,
    ]


class PositionLogAdmin(admin.ModelAdmin):
    pass


class CronLogAdmin(admin.ModelAdmin):
    pass


admin.site.register(PositionLog, PositionLogAdmin)
admin.site.register(Position, PositionAdmin)
admin.site.register(CronLog, CronLogAdmin)
