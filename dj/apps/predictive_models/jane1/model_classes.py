import torch
ENCODED_FEATURES_COUNT = 50
JaneStreetDatasetPredict_Y_LEN = 18
JaneStreetEncode1Dataset_Y_LEN = 1296

def get_core_model(
    input_size: int,
    output_size: int,
    hidden_count: int,
    dropout_p: float = 0.16,
    net_width: int = 32,
) -> torch.nn.Module:
    assert hidden_count > 0
    layers = []

    def append_layer(layers, _input_size, _output_size, just_linear=False):
        layers.append(torch.nn.Linear(_input_size, _output_size))
        if just_linear:
            return
        layers.append(torch.nn.Dropout(p=dropout_p))
        layers.append(torch.nn.ELU())
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
            input_size, bottleneck, 1, dropout_p=0.4, net_width=2000
        )
        self.decoder = get_core_model(
            bottleneck, output_size, 1, dropout_p=0.4, net_width=2000
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torch.sigmoid_(x)
        x = self.decoder(x)
        return x


def create_predict_model():
    return get_core_model(
        ENCODED_FEATURES_COUNT,
        JaneStreetDatasetPredict_Y_LEN,
        hidden_count=2,
        dropout_p=0.4,
        net_width=2000,
    )
