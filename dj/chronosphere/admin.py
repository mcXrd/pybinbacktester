from django.contrib import admin

from .models import Chronosphere, Decider, TickRecord


class ChronosphereAdmin(admin.ModelAdmin):
    pass


class DeciderAdmin(admin.ModelAdmin):
    pass


class TickRecordAdmin(admin.ModelAdmin):
    pass


admin.site.register(Chronosphere, ChronosphereAdmin)
admin.site.register(Decider, DeciderAdmin)
admin.site.register(TickRecord, TickRecordAdmin)
