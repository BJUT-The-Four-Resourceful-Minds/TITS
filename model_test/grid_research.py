import torch
import numpy as np
from torch import nn
from module.LSTM import LSTMAutoencoder
from torch.utils.data import random_split, DataLoader
from data_processing.timeseries_dataset import TimeSeriesDataset
from model_test.CustomClass import SimpleConcatDataset, SimpleSubset, CustomDataset
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def AUC(y_true, y_pred):#计算AUC指标 输入真指标与预测指标两个列表 指标的集合含义是距离左上角的距离
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return np.sqrt((1 - tpr) ** 2 + fpr ** 2)


def g_mean(y_true, y_pred):#计算G_Mean指标 输入真指标与预测指标两个列表
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return np.sqrt(tpr * (1 - fpr))


def load_datasets(window_size):#加载数据文件
    normal_run_path = r'../Car-Hacking Dataset/normal_run_data/normal_run_data.txt'
    DoS_dataset_path = r'../Car-Hacking Dataset/DoS_dataset.csv'
    Fuzzy_dataset_path = r'../Car-Hacking Dataset/Fuzzy_dataset.csv'
    RPM_dataset_path = r'../Car-Hacking Dataset/RPM_dataset.csv'
    gear_dataset_path = r'../Car-Hacking Dataset/gear_dataset.csv'

    normal_run_dataset = TimeSeriesDataset(normal_run_path, window_size)
    DoS_dataset = TimeSeriesDataset(DoS_dataset_path, window_size)
    RPM_dataset = TimeSeriesDataset(RPM_dataset_path, window_size)
    gear_dataset = TimeSeriesDataset(gear_dataset_path, window_size)
    Fuzzy_dataset = TimeSeriesDataset(Fuzzy_dataset_path, window_size)

    car_hacking_dataset = SimpleConcatDataset(
        [normal_run_dataset, DoS_dataset, RPM_dataset, gear_dataset, Fuzzy_dataset])

    train_size = int(0.8 * len(car_hacking_dataset))
    test_size = len(car_hacking_dataset) - train_size
    train_subset, test_subset = random_split(car_hacking_dataset, [train_size, test_size])

    train_dataset = SimpleSubset(train_subset.dataset, train_subset.indices)
    test_dataset = SimpleSubset(test_subset.dataset, test_subset.indices)

    return train_dataset, test_dataset


def prepare_data_loaders(train_dataset):#返回（X,Label)的数据集
    train_loss_loader = DataLoader(train_dataset)
    X = []
    Label = []
    for i in range(len(train_dataset)):
        x, label = train_dataset.get_test_sample(i)
        X.append(x)
        Label.append(label)
    train_label_loader = CustomDataset(X, Label)
    return train_loss_loader, train_label_loader, Label


def test_model(model, loss_dataloader, criterion, device):#遍历训练集生成重建偏差的列表
    model.to(device)
    loss_list = []
    for X, X_next in loss_dataloader:
        X_next_hat = model(X)
        loss = criterion(X_next, X_next_hat)
        loss_list.append(loss)
    return loss_list


def generate_predictions(Loss, params_grid):#Label_hat_np的形状是（阈值个数，每个阈值判断条件下的判断结果的个数）
    Label_hat = []
    for threshold in params_grid['threshold']:
        row = []
        for loss in Loss:
            if loss >= threshold:
                label_hat = 0
            else:
                label_hat = 1
            row.append(label_hat)
        Label_hat.append(row)
    Label_hat_np = np.array(Label_hat)
    return Label_hat_np


def evaluate_metrics(Label, Label_hat, params_grid):#不同阈值中寻找，不同指标分别达到最大时阈值的值
    best_accuracy = 0
    best_accuracy_threshold = 0
    best_f1 = 0
    best_f1_threshold = 0
    best_gmean = 0
    best_gmean_threshold = 0

    for i, label_hat in enumerate(Label_hat):
        accuracy = accuracy_score(Label, label_hat)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_accuracy_threshold = params_grid['threshold'][i]

        f1 = f1_score(Label, label_hat)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = params_grid['threshold'][i]

        gmean = g_mean(Label, label_hat)
        if gmean > best_gmean:
            best_gmean = gmean
            best_gmean_threshold = params_grid['threshold'][i]

    print("准确率最大值: {:.4f}, 对应的阈值: {:.4f}".format(best_accuracy, best_accuracy_threshold))
    print("F1 分数最大值: {:.4f}, 对应的阈值: {:.4f}".format(best_f1, best_f1_threshold))
    print("G - Mean 最大值: {:.4f}, 对应的阈值: {:.4f}".format(best_gmean, best_gmean_threshold))


def calculate_AUC_distance(Label, Label_hat, params_grid):#不同阈值中寻找，AUC指标达到最大时阈值的值
    AUC_distance = []
    for i in range(len(params_grid['threshold'])):
        AUC_distance.append(AUC(Label, Label_hat[i]))
    AUC_min = min(AUC_distance)
    AUC_index = np.argmin(AUC_distance)
    print(AUC_distance)
    print(f"最小值是 {AUC_min}，对应的索引是 {AUC_index}")
    print(params_grid['threshold'][AUC_index])


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss()

    hidden_size = 100
    input_size = 1
    num_layers = 2
    window_size = 10

    #候选的阈值
    params_grid = {'threshold': np.arange(0.05, 0.96, 0.01)}

    # 加载数据集
    train_dataset, test_dataset = load_datasets(window_size)

    # 准备数据加载器
    #这里都没有打乱顺序，所以一个索引对应的数据是关联的
    train_loss_loader, train_label_loader, Label = prepare_data_loaders(train_dataset)

    # 加载训练好的模型
    model = LSTMAutoencoder(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load('../model.pt'))

    # 测试模型并获取损失列表
    Loss = test_model(model, train_loss_loader, criterion, device)

    # 生成预测结果
    Label_hat_np = generate_predictions(Loss, params_grid)

    # 评估指标
    evaluate_metrics(Label, Label_hat_np, params_grid)

    # 计算 AUC 距离
    calculate_AUC_distance(Label, Label_hat_np, params_grid)


if __name__ == "__main__":
    main()
