import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from NLT.NLT_main import likelihood_transformation
from data_processing.car_hacking_process_data import car_hacking_process_data
from data_processing.nb15_process_data import nb15_process_data
from model_test.CustomClass import SimpleConcatDataset


# 将特征提取后的数据转变为可供训练的Dataset类
class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, window_size):
        if 'Car-Hacking' in file_path:
            features, label = car_hacking_process_data(file_path)
            features = np.array(features)
        elif 'NB15' in file_path:
            path = r".\unsw-nb15\versions\1\UNSW-"
            print(f"loading {file_path}")
            features, label = nb15_process_data(f"{path}{file_path}.csv")
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
        dataset_label = []
        for i in range(len(features) - window_size):
            _x = features[i:(i + 10)]
            dataset_x.append(_x)
            dataset_y.append(features[i + window_size])
            # label中1是正常0是攻击
            for j in range(window_size):
                if (label[i + j] == 0):  #10个中只要有一个为攻击则标记为攻击
                    dataset_label.append(0)
                else:
                    dataset_label.append(1)

        # 转换为 PyTorch 张量
        self.X = torch.tensor(dataset_x, dtype=torch.float32)
        self.y = torch.tensor(dataset_y, dtype=torch.float32)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    #用于返回（X,label） 用来与判断出的类别比对
    def get_test_sample(self, index=None):  #我想实现默认不输入时返回整个列表，但是不传入index时总是报错
        if index is None:
            return self.X, self.y
        return self.X[index], self.y[index]


def loading_car_hacking(window_size):
    print("loading data")
    normal_run_path = r'.\Car-Hacking Dataset\normal_run_data\normal_run_data.txt'
    DoS_dataset_path = r'.\Car-Hacking Dataset\DoS_dataset.csv'
    Fuzzy_dataset_path = r'.\Car-Hacking Dataset\Fuzzy_dataset.csv'
    RPM_dataset_path = r'.\Car-Hacking Dataset\RPM_dataset.csv'
    gear_dataset_path = r'.\Car-Hacking Dataset\gear_dataset.csv'

    normal_run_dataset = TimeSeriesDataset(normal_run_path, window_size)
    DoS_dataset_dataset = TimeSeriesDataset(DoS_dataset_path, window_size)
    RPM_dataset_dataset = TimeSeriesDataset(RPM_dataset_path, window_size)
    gear_dataset_dataset = TimeSeriesDataset(gear_dataset_path, window_size)
    Fuzzy_dataset_dataset = TimeSeriesDataset(Fuzzy_dataset_path, window_size)

    print('loading success')

    car_hacking_dataset = SimpleConcatDataset(
        [normal_run_dataset, DoS_dataset_dataset, RPM_dataset_dataset, gear_dataset_dataset, Fuzzy_dataset_dataset])

    # car_hacking_dataset = normal_run_dataset

    train_size = int(0.8 * len(car_hacking_dataset))
    test_size = len(car_hacking_dataset) - train_size
    # 使用 random_split 函数将数据集分割成训练集和测试集
    train_dataset, train_dataset = random_split(car_hacking_dataset, [train_size, test_size])
    return train_dataset, train_dataset


def loading_nb15(window_size):
    NB15_1 = "NB15_1"
    NB15_2 = "NB15_2"
    NB15_3 = "NB15_3"
    NB15_4 = "NB15_4"

    nb15_1 = TimeSeriesDataset(NB15_1, 10)
    nb15_2 = TimeSeriesDataset(NB15_2, 10)
    nb15_3 = TimeSeriesDataset(NB15_3, 10)
    nb15_4 = TimeSeriesDataset(NB15_4, 10)

    nb15_dataset = SimpleConcatDataset([nb15_1, nb15_2, nb15_3, nb15_4])

    train_size = int(0.8 * len(nb15_dataset))
    test_size = len(nb15_dataset) - train_size
    # 使用 random_split 函数将数据集分割成训练集和测试集
    train_dataset, test_dataset = random_split(nb15_dataset, [train_size, test_size])
    return train_dataset, test_dataset
