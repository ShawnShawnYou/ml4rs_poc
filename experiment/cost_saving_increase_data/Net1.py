import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.input = nn.Linear(15, 20)
        self.fc1 = nn.Linear(20, 40)
        self.fc2 = nn.Linear(40, 80)
        self.fc3 = nn.Linear(80, 40)
        self.fc4 = nn.Linear(40, 20)
        self.output = nn.Linear(20, 1)

        self.norm_20 = nn.BatchNorm1d(20)
        self.norm_40 = nn.BatchNorm1d(40)
        self.norm_80 = nn.BatchNorm1d(80)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax()

    def forward(self, x):
        x = self.sigmoid(self.norm_20(self.input(x)))
        x = self.sigmoid(self.norm_40(self.fc1(x)))
        x = self.sigmoid(self.norm_80(self.fc2(x)))
        x = self.sigmoid(self.norm_40(self.fc3(x)))
        x = self.sigmoid(self.norm_20(self.fc4(x)))
        x = self.sigmoid(self.output(x))

        return x

    # def predict(self, x):
    #     pred = F.softmax(self.forward(x))
    #     ans = []
    #     for t in pred:
    #         if t[0] > t[1]:
    #             ans.append(0)
    #         else:
    #             ans.append(1)
    #     return torch.tensor(ans)