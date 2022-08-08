import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.input = nn.Linear(15, 30)
        self.fc1 = nn.Linear(30, 60)
        self.fc2 = nn.Linear(60, 30)
        self.output = nn.Linear(30, 1)

        self.norm_30 = nn.BatchNorm1d(30)
        self.norm_60 = nn.BatchNorm1d(60)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax()

    def forward(self, x):
        x = self.sigmoid(self.norm_30(self.input(x)))
        x = self.sigmoid(self.norm_60(self.fc1(x)))
        x = self.sigmoid(self.norm_30(self.fc2(x)))
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