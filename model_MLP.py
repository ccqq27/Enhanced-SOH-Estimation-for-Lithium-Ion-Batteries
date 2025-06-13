from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim=32, output_dim=1, layers_num=4, hidden_dim=50, dropout=0.2):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim
        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
                self.layers.append(nn.ReLU())
            elif i == layers_num-1:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*self.layers)

    def initialize(self, method='he'):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)
