import torch.nn as nn


class LinearBlock(nn.Module):

    def __init__(self, input_size, output_size, dropout_ratio=0.25, bias=True, act=None):
        super(LinearBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.model = nn.Sequential(
            # nn.BatchNorm1d(input_size),
            nn.Linear(input_size, output_size, bias=bias)
            # , nn.Dropout(dropout_ratio)
        )
        self.act = act

    def forward(self, x):
        x = self.model(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DQN(nn.Module):

    def __init__(self, dropout_ratio=0.25, final_layer=64):
        super(DQN, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.final_layer = final_layer
        self.linear_layers = nn.Sequential(
            LinearBlock(8, 64, self.dropout_ratio, True, nn.ReLU()),
            # LinearBlock(32, 64, self.dropout_ratio, True, nn.ReLU()),
            LinearBlock(64, 64, self.dropout_ratio, True, nn.ReLU()),
            LinearBlock(64, 4, self.dropout_ratio, True),
        )

    def forward(self, batch):
        return self.linear_layers(batch)

