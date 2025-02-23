import torch
from torch.utils.data import Dataset
import numpy as np

from NLT.NLT_main import likelihood_transformation
from data_processing.car_hacking_process_data import car_hacking_process_data
from data_processing.nb15_process_data import nb15_process_data

# 将特征提取后的数据转变为可供训练的Dataset类
class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, window_size):
        if 'Car-Hacking' in file_path:
            features, label = car_hacking_process_data(file_path)
            features = np.array(features)
        elif 'NB15' in file_path:
            path = r".\unsw-nb15\versions\1\UNSW-"
            print(f"loading {file_path}")
            features,label = nb15_process_data(f"{path}{file_path}.csv")
            mask = np.isnan(features)
            mask = mask.any(axis=1)
            features = features[~mask]
        else:
            print('error')
            return

        features = likelihood_transformation(features)

        features = features.reshape(len(features), 1)

        # 滑动窗口为10
        dataset_x, dataset_y = [], []
        dataset_label=[]
        for i in range(len(features) - window_size):
            _x = features[i:(i + 10)]
            dataset_x.append(_x)
            dataset_y.append(features[i + window_size])
            # label中1是正常0是攻击
            for j in range(window_size):
                if(label[i + j] == 0):#10个中只要有一个为攻击则标记为攻击
                    dataset_label.append(0)
                else:
                    dataset_label.append(1)

        # 转换为 PyTorch 张量
        self.X = torch.tensor(dataset_x, dtype=torch.float32)
        self.y = torch.tensor(dataset_y, dtype=torch.float32)

        self.label=torch.tensor(dataset_label, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    #用于返回（X,label） 用来与判断出的类别比对
    def get_test_sample(self, index=None):#我想实现默认不输入时返回整个列表，但是不传入index时总是报错
        if index is None:
            return self.X, self.label
        return self.X[index], self.label[index]

