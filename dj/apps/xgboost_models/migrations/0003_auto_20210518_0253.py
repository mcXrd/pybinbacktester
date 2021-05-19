# Generated by Django 3.1.6 on 2021-05-18 02:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('xgboost_models', '0002_bestrecommendation'),
    ]

    operations = [
        migrations.AddField(
            model_name='bestmodelcode',
            name='mean_const',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='bestrecommendation',
            name='side',
            field=models.CharField(choices=[('SHORT', 'Short'), ('LONG', 'Long'), ('PASS', 'Pass')], max_length=20),
        ),
    ]