import pandas as pd
import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField
from torch.utils.data import Dataset
from typing import Callable, Tuple
import torch
from torch import Tensor, nn, optim
import torchvision
import time
import copy
import cv2
from tqdm import tqdm



class InputDataset(Dataset):
    
    @staticmethod
    def transform_to_imagetensor(x):
        gadf = GramianAngularField(image_size=100, method='difference')
        x = gadf.fit_transform(x)
        res = []
        for img in x:
            img = np.float32(img)
            color_channeled_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            color_channeled_image = np.moveaxis(color_channeled_image, 2, 0)
            res.append(color_channeled_image)
        
        return np.array(res)
    
    def __init__(
        self, df: pd.DataFrame, device: torch.DeviceObjType, transform: Callable = None, train=False, real_profit=False
    ):
        self.len = len(df)
        self.device = device
        self.x = df.iloc[:, 1:]
        self.x = InputDataset.transform_to_imagetensor(self.x)
        self.y = df.iloc[:, :1]
        self.y = self.y.to_numpy()
        self.real_profit = real_profit

    

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        x = self.x[index]
        y = self.y[index]

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x = x.to(self.device).float()
        if self.real_profit:
            y = y.to(self.device).float()
        else:
            y = y.to(self.device).long()
        return x, y[0]

    def __len__(self):
        return self.len
