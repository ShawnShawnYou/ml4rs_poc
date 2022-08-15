import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.metrics import accuracy_score
import torch.utils.data as Data
from ignite.metrics import Accuracy, Loss, Recall, Precision

from setting import *
from Net1 import Net
torch.manual_seed(1)



def data_filter(path):
    original_data = numpy.load(path, allow_pickle=True)
    X = original_data[:, 0:15]
    Y = original_data[:, 15:16]

    # X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    X = torch.from_numpy(X).type(torch.FloatTensor)
    Y = torch.from_numpy(Y).type(torch.FloatTensor)
    return X, Y


def data_loader(x, y):
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

    evaluate_x, evaluate_y = data_filter(os.path.join(os.path.dirname(__file__), 'new_data_7.7_5.npy'))

    test_acc = Accuracy()
    test_recall = Recall()
    test_precision = Precision()

    train_data_loader = data_loader(train_x, train_y)

    for epoch in range(num_epochs):
        for step, (batch_x, batch_y) in enumerate(train_data_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            out = net(batch_x)
            pred = out.ge(0.5)
            loss = loss_func(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            test_acc.update((pred, batch_y))
            test_recall.update((pred, batch_y))
            test_precision.update((pred, batch_y))

        # net.eval()
        val_acc = evaluate_accuracy(evaluate_x.cuda(), evaluate_y.cuda(), net)
        net.train()
        total_acc = test_acc.compute()
        total_recall = test_recall.compute()
        total_precision = test_precision.compute()

        print(f"Epoch: {epoch}",
              f"Avg loss: {train_loss:>8f}, "
              f"Train acc: {(100 * total_acc):>0.1f}%, "
              f"Valid acc: {(100 * val_acc):>0.1f}%, "
              f"Precision: {(100 * total_precision):>0.1f}%, "
              f"Recall: {(100 * total_recall):>0.1f}%")
        torch.save(net, os.path.join(os.path.dirname(__file__), 'net80.pth'))
        test_precision.reset()
        test_acc.reset()
        test_recall.reset()

        # if (epoch + 1) % 1 == 0:
        #
        #     # train_acc = evaluate_accuracy(train_x, train_y, net)
        #     train_acc = evaluate_accuracy(evaluate_x.cuda(), evaluate_y.cuda(), net)
        #     # train_acc = accuracy_score(net.predict(X), Y)
        #     print('epoch %d ,loss %.4f' % (epoch + 1, train_loss) +', train acc {:.2f}%'
        #           .format(train_acc*100))
        #     torch.save(net, 'net80.pth')


def main():
    try:
        net = torch.load(os.path.join(os.path.dirname(__file__), 'net80.pth'))
    except:
        net = Net()
    net = net.cuda()

    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    # loss_func = nn.MSELoss()

    X, Y = data_filter(os.path.join(os.path.dirname(__file__), 'new_data_7.4_5.npy'))

    # net.eval()
    # test_id = 50
    # print(net(X[0].unsqueeze(0).cuda())[0], Y[0])
    # print(net(X[0:100].cuda())[test_id], Y[test_id])
    # print(net(X[0:100000].cuda())[test_id], Y[test_id])
    # print(evaluate_accuracy(X.cuda(), Y.cuda(), net))

    # return 0
    def test():
        # todo: 检查下是因为时序导致的，还是因为batch size导致的, 先用dataloader shuffle，然后再看看区间正确率
        X, Y = data_filter(os.path.join(os.path.dirname(__file__), 'new_data_7.7_5.npy'))
        test_dataloader = torch.utils.data.DataLoader(dataset=Data.TensorDataset(X[10000:20000], Y[10000:20000]),
                                                      batch_size=128,
                                                      shuffle=True)
        acc = 0
        count = 0
        # print(evaluate_accuracy(X[0:100].cuda(), Y[0:100].cuda(), net))
        for step, (batch_x, batch_y) in enumerate(test_dataloader):
            count += 1
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            pre = net(batch_x)[-1].ge(0.5).item()
            pre = 1 if pre else 0
            if pre == int(batch_y[-1].item()):
                acc += 1

            # correct = (net(batch_x).ge(0.5) == batch_y).sum().item()
            # n = batch_y.shape[0]
            # acc += correct / n

        acc /= count
        print(acc)

        acc = 0
        count = 0
        for i in range(10000, 20000):
            count += 1
            data_x = X[i - 1024: i]
            pre = net(data_x.cuda())[-1].ge(0.5).item()
            pre = 1 if pre else 0
            if pre == int(Y[i].item()):
                acc += 1
        acc /= count
        print(acc)
        return

    test()
    return

    num_epochs = 1000
    torch.cuda.set_device(0)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print(next(net.parameters()).device)
    print(X.device, Y.device)
    train(net, X, Y, loss_func, num_epochs, optimizer)
    torch.save(net, os.path.join(os.path.dirname(__file__), 'net80.pth'))
    # torch.device(0)
    # print(os.cpu_count())


def predict_api():
    try:
        net = torch.load(os.path.join(os.path.dirname(__file__), 'net80.pth'))
    except:
        net = Net()
    net = net.cuda()

    with open(FEATURE_PATH, 'r') as f:
        lines = f.readlines()
        features_str = []
        for line in lines:
            features_str.append(line.strip().split(" "))

    features_np = numpy.asarray(features_str, int)
    data_x = torch.from_numpy(features_np).type(torch.FloatTensor)

    out = net(data_x.cuda())

    with open(PREDICT_RESULT_PATH, 'w') as f:
        for i in out:
            tmp = 1 if i else 0
            f.write(str(tmp))
            f.write('\n')

    print("predict finished")



if __name__ == '__main__':
    # main()
    predict_api()
