# Generated by Django 3.1.6 on 2021-04-18 12:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('market_data', '0002_kline_ignore'),
    ]

    operations = [
        migrations.AlterField(
            model_name='kline',
            name='close_time',
            field=models.DateTimeField(db_index=True),
        ),
    ]
