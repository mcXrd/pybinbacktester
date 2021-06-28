from django.conf import settings
import boto3


def send_alert_sms():

    client = boto3.client(
        "sns",
        aws_access_key_id=settings.SNS_ACCESS_KEY,
        aws_secret_access_key=settings.SNS_SECRET_KEY,
        region_name="eu-central-1",
    )

    client.publish(PhoneNumber="+420739215015", Message="Trading is down!")
