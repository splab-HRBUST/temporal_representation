import torch
import torch.nn as nn
import torch.nn.functional as F

# fix random seed
torch.manual_seed(808)

"""
Neural Networks model : LSTM
"""


class LSTM(nn.Module):
    
    def __init__(self, dim=768, num_layers=1, full_state=False):
        super(LSTM, self).__init__()

        self.dim = dim
        self.num_layers = num_layers
        self.full_state = full_state

        # lstm
        self.lstm = nn.LSTM(dim, dim, num_layers)





    def forward(self, x):
        batch = x.shape[0]
        dim = x.shape[1]

        # init hidden layer and cell layer
        # h0 = torch.randn(1, batch, dim)
        h0 = torch.zeros(1, batch, dim)
        # c0 = torch.randn(1, batch, dim)
        c0 = torch.zeros(1, batch, dim)

        # transpose tensor
        x = torch.transpose(x, 1, 2)
        x = torch.transpose(x, 0, 1)
        # lstm
        yn, (hn, _) = self.lstm(x, (h0, c0))

        if self.full_state:
            lstm_out=yn
        else:
            lstm_out = hn

        # transpose tensor
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = torch.tanh(lstm_out)
        
        return lstm_out




# test

# lstm = LSTM(dim=768, num_layers=1, full_state=False)
# x = torch.randn(512, 768, 13)
# x = lstm(x)
# print(x.shape)