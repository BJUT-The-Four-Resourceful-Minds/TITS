from torch import nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通过编码器
        out, _ = self.encoder(x)
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.decoder(out)
        out = self.sigmoid(out)
        return out


# 训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch={epoch + 1},loss={loss.item()}")
    return model
