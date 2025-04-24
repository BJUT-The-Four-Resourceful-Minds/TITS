import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_processing.timeseries_dataset import loading_car_hacking
from model_test.grid_research import grid_research
from model_test.value_display import value_display
from module.LSTM import LSTMAutoencoder, train_model

if __name__ == '__main__':
    # module setting
    batch_size = 100
    hidden_size = 100
    epoch = 500
    window_size = 10
    input_size = 1
    num_layers = 2
    learning_rate = 0.01
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module_file = 'car_hacking_module.pt'

    train_dataset, test_dataset = loading_car_hacking(window_size)
    if not os.path.exists(module_file):
        print("training module")
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
        model = LSTMAutoencoder(input_size, hidden_size, num_layers)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        model = train_model(model, train_loader, test_loader, criterion, optimizer, epoch, device)
        value_display(model, test_loader)
        #保存模型，后面直接在其他文件读取训练好的模型
        torch.save(model.state_dict(), module_file)
        print('training Done')

    grid_research(test_dataset, module_file)
