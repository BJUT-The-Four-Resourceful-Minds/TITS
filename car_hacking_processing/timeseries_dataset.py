import torch
from torch.utils.data import Dataset
import numpy as np

from NLT.NLT_main import likelihood_transformation
from car_hacking_processing.process_data import process_data


class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, window_size):
        features, labels = process_data(file_path)
        features = np.array(features)
        features = likelihood_transformation(features)
        features = features.reshape(len(features), 1)

        dataset_x, dataset_y = [], []
        for i in range(len(features) - window_size):
            _x = features[i:(i + 10)]
            dataset_x.append(_x)
            dataset_y.append(features[i + window_size])

        # 转换为 PyTorch 张量
        self.X = torch.tensor(dataset_x, dtype=torch.float32)
        self.y = torch.tensor(dataset_y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
