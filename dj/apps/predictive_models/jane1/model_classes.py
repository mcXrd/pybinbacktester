import torch
from typing import Callable

ENCODED_FEATURES_COUNT = 200


def generate_settings_from_df(df):
    last_trade_in_column = -1
    for col in df.columns:
        if col.startswith("trade_in"):
            last_trade_in_column += 1
        else:
            break
    RESP_START = df.iloc[:, 0].name
    RESP_END = df.iloc[:, last_trade_in_column].name
    JaneStreetDatasetPredict_Y_LEN = len(df.loc[:, RESP_START:RESP_END].columns)

    JaneStreetEncode1Dataset_Y_START_COLUMN = df.iloc[:, last_trade_in_column + 1].name
    JaneStreetEncode1Dataset_Y_END_COLUMN = df.iloc[:, -1].name

    JaneStreetEncode1Dataset_Y_LEN = len(
        df.loc[
            :,
            JaneStreetEncode1Dataset_Y_START_COLUMN:JaneStreetEncode1Dataset_Y_END_COLUMN,
        ].columns
    )

    return (
        JaneStreetDatasetPredict_Y_LEN,
        RESP_START,
        RESP_END,
        JaneStreetEncode1Dataset_Y_LEN,
        JaneStreetEncode1Dataset_Y_START_COLUMN,
        JaneStreetEncode1Dataset_Y_END_COLUMN,
    )


def get_core_model(
    input_size: int,
    output_size: int,
    hidden_count: int,
    dropout_p: float = 0.16,
    net_width: int = 32,
    activation: Callable = None,
) -> torch.nn.Module:
    assert hidden_count > 0
    layers = []

    def append_layer(layers, _input_size, _output_size, just_linear=False):
        layers.append(torch.nn.Linear(_input_size, _output_size))
        if just_linear:
            return
        layers.append(torch.nn.Dropout(p=dropout_p))
        if activation:
            layers.append(activation())
        else:
            # layers.append(torch.nn.ELU())
            # layers.append(torch.nn.LeakyReLU())
            layers.append(torch.nn.Sigmoid())
            # layers.append(torch.nn.ReLU())
        layers.append(torch.nn.BatchNorm1d(_output_size))

    append_layer(layers, input_size, net_width)

    for one in range(hidden_count):
        append_layer(layers, net_width, net_width)

    append_layer(layers, net_width, output_size, just_linear=True)
    return torch.nn.Sequential(*layers)


class autoencoder(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, bottleneck: int):
        super(autoencoder, self).__init__()
        self.encoder = get_core_model(
            input_size,
            bottleneck,
            1,
            dropout_p=0.5,
            net_width=800,
            activation=torch.nn.ELU,
        )
        self.decoder = get_core_model(
            bottleneck,
            output_size,
            1,
            dropout_p=0.5,
            net_width=800,
            activation=torch.nn.ELU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torch.sigmoid_(x)
        x = self.decoder(x)
        return x


def create_predict_model(JaneStreetDatasetPredict_Y_LEN):
    return get_core_model(
        ENCODED_FEATURES_COUNT,
        JaneStreetDatasetPredict_Y_LEN,
        hidden_count=5,
        dropout_p=0.5,
        net_width=2000,
        activation=torch.nn.ELU,
    )
