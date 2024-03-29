# Generated by Django 3.1.6 on 2021-04-17 11:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictive_models', '0005_auto_20210408_0055'),
    ]

    operations = [
        migrations.CreateModel(
            name='CronLog',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('log_message', models.TextField(blank=True, null=True)),
                ('log_json', models.JSONField(blank=True, null=True)),
                ('name', models.CharField(blank=True, max_length=200, null=True)),
            ],
        ),
        migrations.AddField(
            model_name='position',
            name='liquidation_being_processed',
            field=models.BooleanField(default=False),
        ),
    ]
