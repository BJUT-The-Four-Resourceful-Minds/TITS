import numpy as np
import torch
from torch import nn
from module.LSTM import LSTMAutoencoder
from torch.utils.data import random_split, DataLoader
from data_processing.timeseries_dataset import TimeSeriesDataset
from SimpleSubset import SimpleSubset
from CustomConcatDataset import SimpleConcatDataset
from CustomDataset import CustomDataset
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve,auc
import matplotlib.pyplot as plt

def g_mean(y_true, y_pred):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # 计算真正率
    tpr = tp / (tp + fn)
    # 计算假正率
    fpr = fp / (fp + tn)
    # 计算 G-Mean
    return np.sqrt(tpr * (1 - fpr))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.L1Loss()
# module setting
batch_size = 100
hidden_size = 100
input_size = 1
num_layers = 2
window_size = 10

print("a")#数据生成
# 从 DataLoader 中提取数据并转换为 numpy 数组
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
print("b")

car_hacking_dataset = SimpleConcatDataset(
    [normal_run_dataset, DoS_dataset_dataset, RPM_dataset_dataset, gear_dataset_dataset, Fuzzy_dataset_dataset])

train_size = int(0.8 * len(car_hacking_dataset))
test_size = len(car_hacking_dataset) - train_size
# 使用 random_split 函数将数据集分割成训练集和测试集
train_subset, test_subset = random_split(car_hacking_dataset, [train_size, test_size])

# 转换为支持 get_test_sample 的自定义子集
train_dataset = SimpleSubset(train_subset.dataset, train_subset.indices)
test_dataset = SimpleSubset(test_subset.dataset, test_subset.indices)

train_loss_loader = DataLoader(train_dataset)
print(len(train_loss_loader))
#print(test_dataset.get_test_sample())
X=[]
Label=[]
for i in range(len(train_dataset)):
    x,label=train_dataset.get_test_sample(i)
    X.append(x)
    Label.append(label)
#print(len(Label))
train_label_loader = CustomDataset(X, Label)
print(len(Label))


# 2. 初始化分类器
model = LSTMAutoencoder(input_size, hidden_size, num_layers)
model.load_state_dict(torch.load('model.pt'))

# 3. 修改参数网格
params_grid = {'threshold': np.arange(0.05, 0.96, 0.001)}
def test_model(model,loss_dataloader, criterion, device):
    model.to(device)
    loss_list=[]
    for X,X_next in loss_dataloader:
        X_next_hat=model(X)
        loss=criterion(X_next,X_next_hat)
        loss_list.append(loss)
    return loss_list

Loss=test_model(model,train_loss_loader, criterion,device)

Label_hat=[]
for threshold in params_grid['threshold']:
    for loss in Loss:
        if loss >= threshold:
            label_hat=0
        else:
            label_hat=1
        Label_hat.append(label_hat)

Label_hat_np=np.array(Label_hat)
Label_hat_np.reshape(len(params_grid['threshold']),-1)