import torch
import torch.nn as nn
import numpy as np
import math

###### Input dimensions: (Batch, T, 4)
# Output dimensions: (Batch, 1, 500)
class LSTM_TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=501, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=100, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=100, hidden_size=300, num_layers=1, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=300, hidden_size=500, num_layers=1, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, h = self.lstm4(x)
        
        return torch.squeeze(h[0] , dim = 0) # to output (BS , 500) insted of (BS ,1, 500)
