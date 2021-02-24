from apps.predictive_models.jane1.model_classes import (
    autoencoder,
    create_predict_model,
    JaneStreetEncode1Dataset_Y_LEN,
    ENCODED_FEATURES_COUNT,
)
import os
import torch

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


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
    model = create_predict_model()
    model.load_state_dict(torch.load("{}/predict_model_state_dict.pt".format(DIR_PATH)))
    model.eval()
    return model
