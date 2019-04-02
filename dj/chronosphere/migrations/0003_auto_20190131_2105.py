# Generated by Django 2.1.5 on 2019-01-31 21:05

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('chronosphere', '0002_chronosphere_end_time'),
    ]

    operations = [
        migrations.AddField(
            model_name='chronosphere',
            name='start_time',
            field=models.DateTimeField(default=datetime.datetime(2019, 1, 31, 21, 5, 40, 596687, tzinfo=utc)),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='decider',
            name='decider_name',
            field=models.CharField(max_length=1000, unique=True),
        ),
    ]
