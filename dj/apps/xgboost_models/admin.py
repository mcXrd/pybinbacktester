from django.contrib import admin
from apps.xgboost_models.models import BestModelCode
from apps.xgboost_models.models import BestRecommendation


class BestModelCodeAdmin(admin.ModelAdmin):
    readonly_fields = ["start_evaluating", "done_evaluating"]
    list_display = (
        "code",
        "expected_profit",
        "start_evaluating",
        "done_evaluating",
    )


class BestRecommendationAdmin(admin.ModelAdmin):
    readonly_fields = ["start_evaluating"]
    list_display = (
        "symbol",
        "side",
        "start_evaluating",
        "done_evaluating",
    )


admin.site.register(BestModelCode, BestModelCodeAdmin)
admin.site.register(BestRecommendation, BestRecommendationAdmin)
