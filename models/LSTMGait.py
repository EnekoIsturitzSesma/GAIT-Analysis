import torch.nn as nn

class LSTMGait(nn.Module):
    def __init__(self, num_channels, num_class, hidden_size=128, num_layers=2, dropout_rate=0.25):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class CNNBiLSTMGait(nn.Module):
    def __init__(self, num_channels, num_class, cnn_channels=(64, 128, 128), kernel_size=5, hidden_size=128, num_layers=2, dropout_rate=0.25):
        super().__init__()

        if isinstance(cnn_channels, int):
            cnn_channels = (cnn_channels,)

        layers = []
        in_ch = num_channels
        for out_ch in cnn_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_class)

    def forward(self, x):
        x = x.permute(0,2,1)   # Shape for Conv1d
        x = self.cnn(x)
        x = x.permute(0,2,1)   # Shape for lstm

        x, _ = self.lstm(x)
        x = self.fc(x)
        return x