#!/usr/bin/env python

# =============================================================================
# test_lp_time_pt.py
# Author: Anmol Kabra (slightly changed from PyTorch's MNIST tutorial)
# Project: Solving the Avicaching Game Faster and Better (Summer 2017)
# -----------------------------------------------------------------------------
# Purpose of the Script:
#   Was trying to see if the CPU "set" synchronization delay was particular to 
#   our model. Therefore, ran the LP interspersed with a MNIST network from 
#   PyTorch's tutorial.
# -----------------------------------------------------------------------------
# Required Dependencies/Software:
#   - Python 2.x (obviously, Anaconda environment used originally)
#   - NumPy
#   - PyTorch
#   - Torchvision
# =============================================================================

from __future__ import print_function
import argparse, time, lp
import torch
import torch.nn as nn, numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

J=116
lp_A, lp_c = lp.build_A(J), lp.build_c(J)

###### ONLY LP
# lp_A, lp_c = lp.build_A(J), lp.build_c(J)
# lp_time_log = []
# for e in xrange(args.epochs):
#     r_on_cpu = np.random.randn(J)
#     start_lp_time = time.time()
# 
#     # CONSTRAIN -- LP
#     # 1.0 is the sum constraint of rewards
#     # the first J outputs are the new rewards
#     lp_res = lp.run_lp(lp_A, lp_c, J, r_on_cpu, 1.0)
#     lp_time = time.time() - start_lp_time
#     print(lp_time)
#     lp_time_log.append([e, lp_time])
# fname = "pt_onlylp, epochs=%d, time=%.0f" % (args.epochs, time.time())
# np.savetxt("./stats/" + fname + ".txt", lp_time_log, fmt="%.6f", delimiter=",")
# sys.exit()
###########

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    ### lp run
    r_on_cpu = np.random.randn(J)
    start_lp_time = time.time()
    lp_res = lp.run_lp(lp_A, lp_c, J, r_on_cpu, 1.0)
    lp_time = time.time() - start_lp_time
    print(lp_time)

        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data[0]))
    return lp_time

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

lp_time_log = []
for epoch in range(1, args.epochs + 1):
    lp_t = train(epoch)
    lp_time_log.append([epoch, lp_t])

if args.cuda:
    file_pre = "gpu, "
else:
    file_pre = "cpu, "
fname = "epochs=%d, time=%.0f" % (args.epochs, time.time())
np.savetxt("./stats/pt_" + file_pre + fname + ".txt", lp_time_log, fmt="%.6f", delimiter=",")
