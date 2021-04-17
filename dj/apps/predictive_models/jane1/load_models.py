from apps.predictive_models.jane1.model_classes import (
    autoencoder,
    create_predict_model,
    generate_settings_from_df,
    ENCODED_FEATURES_COUNT,
)
import os
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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


def load_train_df(stop=None):
    path = "{}/learning_df.hdf".format(DIR_PATH)
    df = pd.read_hdf(path, stop=stop)
    df = df.fillna(0)
    return df


def take_currencies_from_df_columns(df):
    columns = []
    for column in list(df.columns):
        if "close_price" not in column:
            break
        columns.append(column)
    res = []
    for column in columns:
        sc = column.split("_")
        curr = sc[-3]
        if not curr in res:
            res.append(curr)
    return res


def resort_columns(learning_df, df_to_sort):
    df = df_to_sort.reindex(learning_df.columns, axis=1)
    return df


def preprocessing_scale_df(
    df: pd.DataFrame, last_min_max_scaler: StandardScaler
) -> pd.DataFrame:
    start_i = JaneStreetEncode1Dataset_Y_START_COLUMN
    end_i = JaneStreetEncode1Dataset_Y_END_COLUMN
    if not last_min_max_scaler:
        last_min_max_scaler = StandardScaler()
        df.loc[:, start_i:end_i] = last_min_max_scaler.fit_transform(
            df.loc[:, start_i:end_i]
        )
    else:
        df.loc[:, start_i:end_i] = last_min_max_scaler.transform(
            df.loc[:, start_i:end_i]
        )
    return df, last_min_max_scaler
