# Generated by Django 3.1.6 on 2021-05-16 10:46

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='BestModelCode',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('code', models.CharField(blank=True, max_length=20, null=True)),
                ('expected_profit', models.FloatField(blank=True, null=True)),
                ('start_evaluating', models.DateTimeField(blank=True, null=True)),
                ('done_evaluating', models.DateTimeField(blank=True, null=True)),
            ],
        ),
    ]
