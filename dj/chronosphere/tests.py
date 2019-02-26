import responses
from chronosphere.models import Decider, TickRecord
from django.core.management import call_command
from django.test import TestCase


# Create your tests here.


# TODO register_decider.py tests

class RegisterDeciderTestCase(TestCase):

    def test_register_valid_decider(self):
        call_command('register_decider', decider_name='n1', decider_url='u1')
        assert Decider.objects.filter(decider_name='n1').exists()


class RunChronospheresTestCase(TestCase):

    @responses.activate
    def test_decider_is_getting_called(self):
        pair_name = 'testpair111'
        endpoint = 'https://www.abc.com/services/'
        responses.add(
            responses.POST,
            endpoint,
            json={
                'action': 'buy',
                'pair': pair_name,
                'amount': 1.0
            }
        )
        call_command('register_decider', decider_name='n1', decider_url=endpoint)
        call_command('run_chronospheres')

        assert TickRecord.objects.filter(pair=pair_name).exists()
