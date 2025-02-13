import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, random_split, DataLoader

from car_hacking_processing.timeseries_dataset import TimeSeriesDataset
from module.LSTM import LSTMAutoencoder, train_model

# setting
batch_size = 100
hidden_size = 100
epoch = 400
window_size = 10
input_size = 1
num_layers = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

normal_run_path = r'.\Car-Hacking Dataset\normal_run_data\normal_run_data.txt'
DoS_dataset_path = r'.\Car-Hacking Dataset\DoS_dataset.csv'
Fuzzy_dataset_path = r'.\Car-Hacking Dataset\Fuzzy_dataset.csv'
RPM_dataset_path = r'.\Car-Hacking Dataset\RPM_dataset.csv'
gear_dataset_path = r'.\Car-Hacking Dataset\gear_dataset.csv'

normal_run_dataset = TimeSeriesDataset(normal_run_path, window_size)
DoS_dataset_dataset = TimeSeriesDataset(DoS_dataset_path,window_size)
RPM_dataset_dataset = TimeSeriesDataset(RPM_dataset_path,window_size)
gear_dataset_dataset = TimeSeriesDataset(gear_dataset_path,window_size)
Fuzzy_dataset_dataset = TimeSeriesDataset(Fuzzy_dataset_path,window_size)

print('数据读取完成')

car_hacking_dataset = ConcatDataset(
    [normal_run_dataset, DoS_dataset_dataset, RPM_dataset_dataset, gear_dataset_dataset, Fuzzy_dataset_dataset])

# car_hacking_dataset = normal_run_dataset

train_size = int(0.8 * len(car_hacking_dataset))
test_size = len(car_hacking_dataset) - train_size
# 使用 random_split 函数将数据集分割成训练集和测试集
train_dataset, test_dataset = random_split(car_hacking_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

model = LSTMAutoencoder(input_size, hidden_size, num_layers)
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
train_model(model, train_loader, test_loader, criterion, optimizer, epoch, device)
