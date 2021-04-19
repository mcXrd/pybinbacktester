from apps.predictive_models.jane1.load_models import (
    load_train_df,
    resort_columns,
    take_currencies_from_df_columns,
)
from apps.predictive_models.jane1.load_models import (
    preprocessing_scale_df,
    load_autoencoder,
    load_predictmodel,
)
from apps.market_data.generate_market_data_hdf_utils import (
    exchangify_pairs,
    create_dataframe,
    clean_initial_window_nans,
)
from apps.market_data.models import Kline
from django.utils import timezone
from typing import Tuple, List
import pandas as pd
from apps.predictive_models.jane1.load_models import (
    JaneStreetEncode1Dataset_Y_START_COLUMN,
    JaneStreetEncode1Dataset_Y_END_COLUMN,
)
import torch
import numpy as np
from functools import lru_cache
from django.conf import settings

SAFE_DAY_ADD = (int(settings.LARGEST_DF_WINDOW / 24)) + 2


def create_live_df(days: int = 25, live: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    train_df = load_train_df()
    train_df, last_scaler = preprocessing_scale_df(train_df, None)
    currencies = take_currencies_from_df_columns(train_df)
    symbols = exchangify_pairs(currencies)

    kwargs = {
        "close_time__gt": timezone.now() - timezone.timedelta(days=days + SAFE_DAY_ADD),
        "symbol__in": symbols,
    }
    df = create_dataframe(Kline.objects.filter(**kwargs), live=live)
    df = clean_initial_window_nans(df, rows_to_clean=len(df) - (days * 24))
    df = df.fillna(0)
    df = resort_columns(train_df, df)
    df, last_scaler = preprocessing_scale_df(df, last_scaler)
    assert list(df.columns) == list(train_df.columns)
    return df, currencies


def get_feature_row_and_real_output(df, int_index):
    feature_row = df.loc[
        :,
        JaneStreetEncode1Dataset_Y_START_COLUMN:JaneStreetEncode1Dataset_Y_END_COLUMN,
    ].iloc[int_index, :]
    real_output = df.loc[
        :,
        :JaneStreetEncode1Dataset_Y_START_COLUMN,
    ].iloc[int_index, :-1]
    return feature_row, real_output


def get_model_input(feature_row):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_input = torch.from_numpy(np.array([feature_row])).float().to(device)
    return model_input


@lru_cache
def get_models():
    autoencoder = load_autoencoder()
    predictmodel = load_predictmodel()
    return autoencoder, predictmodel


def get_model_output(model_input):
    autoencoder, predictmodel = get_models()
    model_output = predictmodel(autoencoder.encoder(model_input))
    return model_output


def get_numpy_model_output(model_output):
    numpy_output = model_output.detach().numpy()
    numpy_output = numpy_output[0]
    return numpy_output


class NoTradeException(Exception):
    pass


def resolve_trade(df, df_index, trade_strategy, currencies):

    feature_row, real_output = get_feature_row_and_real_output(df, df_index)
    model_input = get_model_input(feature_row)
    model_output = get_model_output(model_input)
    numpy_model_output = get_numpy_model_output(model_output)

    currency = trade_strategy.pick_currency(numpy_model_output, currencies)
    side = trade_strategy.pick_side(numpy_model_output)

    if not trade_strategy.do_trade(numpy_model_output):
        raise NoTradeException(trade_strategy.do_trade_explanation(numpy_model_output))

    return currency, side


def round_quantity(considered_quantity, symbol):
    considered_quantity = (
        considered_quantity / settings.USDT_FUTURES_MINIMAL_TRADE_AMOUNT[symbol]
    )
    considered_quantity = int(considered_quantity)
    considered_quantity = (
        considered_quantity * settings.USDT_FUTURES_MINIMAL_TRADE_AMOUNT[symbol]
    )
    return considered_quantity


def count_quantity(symbol, price, usdt_amount):
    considered_quantity = (usdt_amount * 0.95) / price
    considered_quantity = round_quantity(considered_quantity, symbol)
    return considered_quantity
