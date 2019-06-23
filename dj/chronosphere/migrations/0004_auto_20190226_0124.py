# Generated by Django 2.1.7 on 2019-02-26 01:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chronosphere', '0003_auto_20190131_2105'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='tickrecord',
            name='tick_result',
        ),
        migrations.AddField(
            model_name='tickrecord',
            name='action',
            field=models.CharField(default=1, max_length=100),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='tickrecord',
            name='amount',
            field=models.FloatField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='tickrecord',
            name='pair',
            field=models.CharField(default=1, max_length=100),
            preserve_default=False,
        ),
    ]