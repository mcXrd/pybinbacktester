from django.db import models


class Decider(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    decider_url = models.URLField(null=False, blank=False, max_length=1000)
    decider_name = models.CharField(null=False, blank=False, max_length=1000, unique=True)


class Chronosphere(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    decider = models.ForeignKey('Decider', null=False, blank=False, on_delete=models.CASCADE)
    completed_percent = models.IntegerField(null=True, blank=True)
    end_time = models.DateTimeField(null=False, blank=False)
    start_time = models.DateTimeField(null=False, blank=False)


class TickRecord(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

    chronosphere = models.ForeignKey('Chronosphere', null=False, blank=False, on_delete=models.CASCADE)

    amount = models.FloatField(null=False, blank=False)
    pair = models.CharField(null=False, blank=False, max_length=100)
    action = models.CharField(null=False, blank=False, max_length=100)
