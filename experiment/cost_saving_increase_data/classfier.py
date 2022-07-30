import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.metrics import accuracy_score
import torch.utils.data as Data
import os
from setting import *
torch.manual_seed(1)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.input = nn.Linear(10, 20)
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



try:
    net = torch.load('net80.pth')
except:
    net = Net()
net = net.cuda()

loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
# loss_func = nn.MSELoss()



def data_filter(path):
    original_data = numpy.load(path, allow_pickle=True)
    X = original_data[:, 0:10]
    Y = original_data[:, 10:11]
    X = torch.from_numpy(X).type(torch.FloatTensor)
    Y = torch.from_numpy(Y).type(torch.FloatTensor)
    return X, Y


def data_loader(x, y):
    nw = min([os.cpu_count(), MINIBATCH_SIZE if MINIBATCH_SIZE > 1 else 0, 8])
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=MINIBATCH_SIZE,
        shuffle=True,
        num_workers=4  # set multi-work num read data
    )
    return loader


def evaluate_accuracy(x, y, net):
    out = net(x)
    correct = (out.ge(0.5) == y).sum().item()
    n = y.shape[0]
    return correct / n


def train(net, train_x, train_y, loss_func, num_epochs, optimizer=None):

    evaluate_x, evaluate_y = data_filter('data_7.4_5.npy')
    evaluate_x = evaluate_x.cuda()
    evaluate_y = evaluate_y.cuda()

    train_data_loader = data_loader(train_x, train_y)

    for epoch in range(num_epochs):
        for step, (batch_x, batch_y) in enumerate(train_data_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            out = net(batch_x)
            loss = loss_func(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

        print(epoch)

        if (epoch + 1) % 1 == 0:

            # train_acc = evaluate_accuracy(train_x, train_y, net)
            train_acc = evaluate_accuracy(evaluate_x.cuda(), evaluate_y.cuda(), net)
            # train_acc = accuracy_score(net.predict(X), Y)
            print('epoch %d ,loss %.4f' % (epoch + 1, train_loss) +', train acc {:.2f}%'
                  .format(train_acc*100))
            torch.save(net, 'net80.pth')




if __name__ == '__main__':


    X, Y = data_filter('data_7.7_5.npy')
    num_epochs = 1000
    torch.cuda.set_device(0)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print(next(net.parameters()).device)
    print(X.device, Y.device)
    train(net, X, Y, loss_func, num_epochs, optimizer)
    torch.save(net, 'net80.pth')
    # torch.device(0)
    # print(os.cpu_count())
