import numpy as np
import torch
from torch.utils.data import Dataset, random_split, Subset

from NLT.NLT_main import likelihood_transformation
from data_processing.car_hacking_process_data import car_hacking_process_data
from data_processing.nb15_process_data import nb15_process_data
from model_test.CustomClass import SimpleConcatDataset, SimpleSubset


# 将特征提取后的数据转变为可供训练的Dataset类
class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, window_size, data_type):
        features = None
        label = []
        print(f"loading {data_type} data")
        if data_type == "car-hacking":
            for path in file_path:
                if features is None:
                    features, label = car_hacking_process_data(path)
                else:
                    feature, labels = car_hacking_process_data(path)
                    features = np.concatenate([features, feature], axis=0)
                    label = np.concatenate([label, labels], axis=0)
            features = np.array(features)
            label = np.array(label)

        elif data_type == "nb15":
            for path in file_path:
                if features is None:
                    features, label = nb15_process_data(path)
                else:
                    feature, labels = nb15_process_data(path)
                    features = np.concatenate([features, feature], axis=0)
                    label = np.concatenate([label, labels], axis=0)
            features = np.array(features)
            label = np.array(label)
            # indices_to_remove = [3, 4, 14786]  # 要丢弃的索引
            # mask = np.ones(len(features), dtype=bool)
            # mask[indices_to_remove] = False
            # features = features[mask]
            # label = label[mask]

        features = likelihood_transformation(features)

        # 数据预处理可视化
        # from matplotlib import pyplot as plt
        # import os
        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        # plt.plot([i for i in range(len(features))], features, label='num')
        # plt.plot([i for i in range(len(label))], label, label='label')
        # plt.show()

        features = features.reshape(len(features), 1)

        # 滑动窗口为10
        dataset_x, dataset_y = [], []
        dataset_label = []
        for i in range(len(features) - window_size):
            _x = features[i:(i + window_size)]
            dataset_x.append(_x)
            dataset_y.append(features[i + window_size])
            # label中1是正常0是攻击
            has_attack = False
            # 检查窗口内的每个标签
            for j in range(window_size):
                if label[i + j] == 0:
                    has_attack = True
                    break
            dataset_label.append(0 if has_attack else 1)  # 0=攻击，1=正常

        # 转换为 PyTorch 张量
        self.X = torch.tensor(dataset_x, dtype=torch.float32)
        self.y = self.X
        self.label = torch.tensor(dataset_label, dtype=torch.float32)

        print(f"{data_type} data loaded")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    #用于返回（X,label） 用来与判断出的类别比对
    def get_test_sample(self, index=None):  #我想实现默认不输入时返回整个列表，但是不传入index时总是报错
        if index is None:
            return self.X, self.label
        return self.X[index], self.label[index]


def loading_car_hacking(window_size):
    normal_run_path = r'.\Car-Hacking Dataset\normal_run_data\normal_run_data.txt'
    DoS_dataset_path = r'.\Car-Hacking Dataset\DoS_dataset.csv'
    Fuzzy_dataset_path = r'.\Car-Hacking Dataset\Fuzzy_dataset.csv'
    RPM_dataset_path = r'.\Car-Hacking Dataset\RPM_dataset.csv'
    gear_dataset_path = r'.\Car-Hacking Dataset\gear_dataset.csv'

    file_path = [normal_run_path, DoS_dataset_path, Fuzzy_dataset_path, RPM_dataset_path, gear_dataset_path]
    car_hacking_dataset = TimeSeriesDataset(file_path, window_size, 'car-hacking')

    # normal_run_dataset = TimeSeriesDataset(normal_run_path, window_size)
    # DoS_dataset_dataset = TimeSeriesDataset(DoS_dataset_path, window_size)
    # RPM_dataset_dataset = TimeSeriesDataset(RPM_dataset_path, window_size)
    # gear_dataset_dataset = TimeSeriesDataset(gear_dataset_path, window_size)
    # Fuzzy_dataset_dataset = TimeSeriesDataset(Fuzzy_dataset_path, window_size)
    #
    # car_hacking_dataset = SimpleConcatDataset(
    #     [normal_run_dataset, DoS_dataset_dataset, RPM_dataset_dataset, gear_dataset_dataset, Fuzzy_dataset_dataset])

    return car_hacking_dataset


def loading_nb15(window_size):
    NB15_1 = r"..\unsw-nb15\versions\1\UNSW-NB15_1.csv"
    NB15_2 = r"..\unsw-nb15\versions\1\UNSW-NB15_2.csv"
    NB15_3 = r"..\unsw-nb15\versions\1\UNSW-NB15_3.csv"
    NB15_4 = r"..\unsw-nb15\versions\1\UNSW-NB15_4.csv"
    file_path = [NB15_1, NB15_2, NB15_3, NB15_4]
    nb15_dataset = TimeSeriesDataset(file_path, window_size, 'nb15')

    # nb15_1 = TimeSeriesDataset(NB15_1, 10)
    # nb15_2 = TimeSeriesDataset(NB15_2, 10)
    # nb15_3 = TimeSeriesDataset(NB15_3, 10)
    # nb15_4 = TimeSeriesDataset(NB15_4, 10)

    # nb15_dataset = SimpleConcatDataset([nb15_1, nb15_2, nb15_3, nb15_4])

    return nb15_dataset


def dataClassifier(dataset):
    # for i in range(len(dataset)):
    #     print(dataset.get_test_sample(i)[1])
    indices_0 = [i for i in range(len(dataset)) if dataset.get_test_sample(i)[1] == 0]
    indices_1 = [i for i in range(len(dataset)) if dataset.get_test_sample(i)[1] == 1]
    # 步骤2：创建子集
    attack_dataset = SimpleSubset(dataset, indices_0)
    normal_dataset = SimpleSubset(dataset, indices_1)

    return normal_dataset, attack_dataset
