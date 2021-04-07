from apps.predictive_models.jane1.model_classes import (
    autoencoder,
    create_predict_model,
    generate_settings_from_df,
    ENCODED_FEATURES_COUNT,
)
import os
import torch
import pandas as pd

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

(
    JaneStreetDatasetPredict_Y_LEN,
    RESP_START,
    RESP_END,
    JaneStreetEncode1Dataset_Y_LEN,
    JaneStreetEncode1Dataset_Y_START_COLUMN,
    JaneStreetEncode1Dataset_Y_END_COLUMN,
) = generate_settings_from_df(
    pd.read_hdf("{}/learning_df.hdf".format(DIR_PATH), stop=5)
)


def load_autoencoder():
    model = autoencoder(
        input_size=JaneStreetEncode1Dataset_Y_LEN,
        output_size=JaneStreetEncode1Dataset_Y_LEN,
        bottleneck=ENCODED_FEATURES_COUNT,
    )
    model.load_state_dict(torch.load("{}/encoder_model_state_dict.pt".format(DIR_PATH)))
    model.eval()
    return model


def load_predictmodel():
    model = create_predict_model(JaneStreetDatasetPredict_Y_LEN)
    model.load_state_dict(torch.load("{}/predict_model_state_dict.pt".format(DIR_PATH)))
    model.eval()
    return model
