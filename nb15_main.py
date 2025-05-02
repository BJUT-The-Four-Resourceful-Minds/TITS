import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_processing.timeseries_dataset import loading_nb15, dataClassifier
from model_test.Custom_split import my_random_split
from model_test.grid_research import grid_research
from model_test.value_display import value_display
from module.LSTM import LSTMAutoencoder, train_model

if __name__ == '__main__':
    # module setting
    batch_size = 50
    hidden_size = 100
    epoch = 500
    window_size = 10
    input_size = 1
    num_layers = 2
    learning_rate = 0.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module_file = 'nb15_module.pt'

    nb15_dataset = loading_nb15(window_size)

    train_size = int(0.8 * len(nb15_dataset))
    test_size = len(nb15_dataset) - train_size
    train_dataset, test_dataset = my_random_split(nb15_dataset, [train_size, test_size])

    normal_dataset, attack_dataset = dataClassifier(train_dataset)
    # test_normal_dataset = SimpleConcatDataset([normal_dataset, attack_dataset])
    model = LSTMAutoencoder(input_size, hidden_size, num_layers)

    if not os.path.exists(module_file):
        print("training module")
        train_loader = DataLoader(normal_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
        model = LSTMAutoencoder(input_size, hidden_size, num_layers)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        train_model(model, train_loader, criterion, optimizer, epoch, device)
        value_display(model, test_loader)
        #保存模型，后面直接在其他文件读取训练好的模型
        torch.save(model.state_dict(), module_file)
        print('training Done')
    else:
        model.load_state_dict(torch.load(f'./{module_file}'))

    grid_research(test_dataset, model)
