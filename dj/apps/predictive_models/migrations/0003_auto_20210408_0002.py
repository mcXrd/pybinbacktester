# Generated by Django 3.1.6 on 2021-04-08 00:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictive_models', '0002_auto_20210407_2017'),
    ]

    operations = [
        migrations.AddField(
            model_name='position',
            name='liq_after_fee_balance',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='position',
            name='liq_initial_fee_balance',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
