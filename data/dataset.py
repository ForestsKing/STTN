import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, his_len, pre_len):
        self.data = data
        self.his_len = his_len
        self.pre_len = pre_len

    def __getitem__(self, index):
        X = np.expand_dims(self.data[index:index + self.his_len, :], axis=2)
        y = np.expand_dims(self.data[index + self.his_len:index + self.his_len + self.pre_len, :], axis=2)
        return X, y

    def __len__(self):
        return len(self.data)-self.his_len-self.pre_len+1
