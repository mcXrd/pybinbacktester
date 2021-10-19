import pandas as pd
import numpy as np
from torch import nn
import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Callable, Tuple
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
COIN = "ADA"
BATCH_SIZE = 100
SEQ_LEN = 120
LSTM_ENABLED = True


df = pd.read_hdf("apr21a.hdf")
df = df[:-180]
df = df.fillna(0)

scaler = StandardScaler()


class InputDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, device: torch.DeviceObjType, transform: Callable = None, train=False
    ):
        self.seq_len = SEQ_LEN
        self.df = df
        if train:
            fc = scaler.fit_transform
        else:
            fc = scaler.transform

        self.x_df = df[
            [
                "usdtfutures_{}USDT_close_price".format(COIN),
                "usdtfutures_{}USDT_volume".format(COIN),
            ]
        ]
        self.x_df = fc(self.x_df)


        self.minutes_forward = 60
        self.len = len(self.df) - (self.seq_len + self.minutes_forward)
        self.transform = transform
        self.device = device

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        y_index = index + self.seq_len + 2
        x = self.x_df
        x = torch.from_numpy(x)[y_index - (self.seq_len + 1) : y_index - 1]
        y = torch.from_numpy(self.x_df)[
            y_index + self.minutes_forward  # 60 means 1 hour here
        ][0]
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        return x, y

    def __len__(self):
        return self.len


validation_size = int(len(df) / 5)
dataset_train = InputDataset(df=df.iloc[:-validation_size], device=DEVICE, train=True)
dataset_valid = InputDataset(df=df.iloc[-validation_size:], device=DEVICE, train=False)

train_loader1 = torch.utils.data.DataLoader(
    dataset=dataset_train, batch_size=BATCH_SIZE, drop_last=True
)
validation_loader1 = torch.utils.data.DataLoader(
    dataset=dataset_valid, batch_size=BATCH_SIZE, drop_last=True
)


class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size

        if LSTM_ENABLED:
            self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=False)
        else:
            self.lstm = nn.GRU(input_size, hidden_layer_size, batch_first=False)

        lin_layers_size = int(hidden_layer_size / 2)
        self.linear = nn.Linear(hidden_layer_size, lin_layers_size)

        self.act1 = nn.ELU()
        self.bn1 = nn.BatchNorm1d(lin_layers_size)

        self.estimator = nn.Linear(
            in_features=lin_layers_size, out_features=output_size
        )

        if LSTM_ENABLED:
            self.hidden_cell = (
                torch.randn(1, BATCH_SIZE, self.hidden_layer_size),
                torch.randn(1, BATCH_SIZE, self.hidden_layer_size),
            )
        else:
            self.hidden_cell = torch.randn(1, BATCH_SIZE, self.hidden_layer_size)

    def forward(self, input_seq):
        input_seq = torch.swapaxes(
            input_seq, 0, 1
        )  # (seq_len, batch, input_size) - is desired format
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        lstm_out = lstm_out[-1, :, :]  # many to one (seq to one)
        res = self.linear(lstm_out)
        res = self.act1(res)
        res = self.bn1(res)
        res = self.estimator(res)
        return res


model = LSTM()
loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


epochs = 10


for i in range(epochs):
    for x, y in tqdm(train_loader1):
        model.train()
        optimizer.zero_grad()
        if LSTM_ENABLED:
            model.hidden_cell[0].detach_()
            model.hidden_cell[1].detach_()
        else:
            model.hidden_cell.detach_()
        y_pred = model(x)

        single_loss = loss_function(y_pred, y)
        single_loss.backward(retain_graph=True)
        optimizer.step()

    val_losses = []
    ii = 0
    for x, y in tqdm(validation_loader1):
        ii += 1
        model.eval()
        y_pred = model(x)
        single_loss = loss_function(y_pred, y)
        if ii == 1:
            print("x {} --- ypred {} --- y {}".format(0, y_pred[0], y[0]))
        if ii % 100 == 0:
            print("x {} --- ypred {} --- y {}".format(0, y_pred[0], y[0]))
        val_losses.append(np.mean(single_loss.detach().numpy()))

    print("Epoch {} loss {}".format(i, np.mean(val_losses)))

    if i % 25 == 1:
        print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")

print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")
