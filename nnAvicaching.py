#!/usr/bin/env python
from __future__ import print_function
import argparse, time, math
import numpy as np
import avicaching_data as ad
import matplotlib.pyplot as plt
# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as torchfun
from torch.autograd import Variable
import torch.optim as optim

# training specs
parser = argparse.ArgumentParser(description="NN for Avicaching model")
parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
    help="inputs learning rate of the network (default=0.01)")
parser.add_argument("--momentum", type=float, default=0.5, metavar="M",
    help="inputs SGD momentum (default=0.5)")
parser.add_argument("--no-cuda", action="store_true", default=False,
    help="disables CUDA training")
parser.add_argument("--epochs", type=int, default=10, metavar="E",
    help="inputs the number of epochs to train for")
parser.add_argument("--locations", type=int, default=116, metavar="L",
    help="inputs the number of locations (default=116)")
parser.add_argument("--time", type=int, default=173, metavar="T",
    help="inputs total time of data collection; number of weeks (default=173)")
parser.add_argument("--eta", type=float, default=10.0, metavar="F",
    help="inputs parameter eta in the model (default=10.0)")
parser.add_argument("--lambda-L1", type=float, default=10.0, metavar="LAM",
    help="inputs the L1 regularizing coefficient")
parser.add_argument("--rand-xyr", action="store_true", default=False,
    help="uses random xyr data")
parser.add_argument("--log-interval", type=int, default=1, metavar="I",
    help="prints training information at I epoch intervals (default=1)")
parser.add_argument("--save-plot", action="store_true", default=False,
    help="saves the plot instead of opening it")
parser.add_argument("--train-percent", type=float, default=0.8, metavar="T",
    help="breaks the data into T percent training and rest testing (default=0.8)")
parser.add_argument("--seed", type=int, default=1, metavar="S",
    help="random seed (default=1)")

args = parser.parse_args()

# assigning cuda check to a single variable
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# parameters and data
J, T = args.locations, args.time
torchten = torch.FloatTensor
F_DIST, numFeatures = [], 0
trainX, trainY, trainR, testX, testY, testR = [], [], [], [], [], []

num_train = int(math.floor(args.train_percent * T))
num_test = T - num_train

# MyNet class
class MyNet(nn.Module):

    def __init__(self, J, numFeatures, eta):
        """
        Initializes MyNet
        """
        super(MyNet, self).__init__()
        self.J = J
        self.eta = eta
        self.w = nn.Parameter(torch.randn(J, numFeatures, 1).type(torchten))

    def forward(self, inp):
        """
        Forward in the network; multiply the weights and return the softmax
        """
        
        inp = torch.bmm(inp, self.w).view(-1, self.J)
        # for u in xrange(len(inp)):
        #     inp[u, u] = inp[u, u].clone() + self.eta    # inp[u][u]
        return torchfun.softmax(inp)

def train(net, optimizer, loss_normalizer):
    """
    Trains the network using MyNet
    """
    loss = 0
    start_time = time.time()

    for t in xrange(num_train):
        # build the input by appending trainR[t]
        trainR_extended = trainR[t].repeat(J, 1)
        inp = torch.cat([F_DIST, trainR_extended], dim=2)     # final F_DIST_processing
        if args.cuda:
            inp = inp.cuda()

        # standardize inp
        diff = inp - torch.mean(inp, dim=2).expand_as(inp)
        std = torch.std(inp, dim=2).expand_as(inp)
        inp = torch.div(diff, std)
        inp = Variable(inp)
        
        # feed in data
        P = net(inp).t()    # P is now weighted -> softmax
        
        # calculate loss
        Pxt = torch.mv(P, trainX[t])
        loss += (trainY[t] - Pxt).pow(2).sum()
    # loss += args.lambda_L1 * torch.norm(net.w.data)
    # print(loss, loss_normalizer)
    loss /= loss_normalizer.data[0]
    
    # backpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end_time = time.time()
    
    return (end_time - start_time, loss.data[0])

def test(net, loss_normalizer):
    """
    Test the network using MyNet
    """
    loss = 0
    start_time = time.time()
    
    for t in xrange(num_test):
        # build the input by appending testR[t]
        testR_extended = testR[t].repeat(J, 1)
        inp = torch.cat([F_DIST, testR_extended], dim=2)     # final F_DIST_processing
        if args.cuda:
            inp = inp.cuda()

        # standardize inp
        diff = inp - torch.mean(inp, dim=2).expand_as(inp)
        std = torch.std(inp, dim=2).expand_as(inp)
        inp = torch.div(diff, std)
        inp = Variable(inp)
        
        # feed in data
        P = net(inp).t()    # P is now weighted -> softmax
        
        # calculate loss
        Pxt = torch.mv(P, testX[t])
        loss += (testY[t] - Pxt).pow(2).sum() 
    # loss += args.lambda_L1 * torch.norm(net.w.data)
    loss /= loss_normalizer.data[0]

    end_time = time.time()
    return (end_time - start_time, loss.data[0])

def save_plot(file_name, x, y, xlabel, ylabel, title):
    train_flatten_res = [i for j in y[0] for i in j]
    test_flatten_res = [i for j in y[1] for i in j]
    print(train_flatten_res)
    print(test_flatten_res)

    train_label, = plt.plot(x, train_flatten_res[1::2], "r-", label="Train Loss") 
    test_label, = plt.plot(x, test_flatten_res[1::2], "b-", label="Test Loss")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(handles=[train_label, test_label])
    plt.title(title)
    if args.save_plot:
        plt.savefig(file_name, bbox_inches="tight", dpi=200)
    else:
        plt.show()

def save_log(file_name, x, y, title):
    with open(file_name, "wt") as f:
        f.write(title + "\n")
        print(len(x), len(y[0]), len(y[1]))
        for i in range(0, len(x), args.log_interval):
            print(x[i], y[0][i][1], y[0][i][0], y[1][i][1], y[1][i][0])
            f.write("epoch = %d\t\ttrainloss = %.8f, traintime = %.4f\t\ttestloss = %.8f, testtime = %.4f\n" % (
                    x[i], y[0][i][1], y[0][i][0],
                    y[1][i][1], y[1][i][0]))

def read_set_data():
    """
    Reads and sets up the datasets
    """
    # read xyr, f and dist datasets from file
    global trainX, trainY, trainR, testX, testY, testR, F_DIST, numFeatures
    if args.rand_xyr:
        X, Y, R = ad.read_XYR_file("./randXYR.txt", J, T)
    else:
        X, Y, R = ad.read_XYR_file("./density_shift_histlong_as_previous_loc_classical_drastic_price_0327_0813.txt", J, T)
    F = ad.read_F_file("./loc_feature_with_avicaching_combined.csv")
    DIST = ad.read_dist_file("./site_distances_km_drastic_price_histlong_0327_0813_combined.txt", J)
    
    # split the data
    trainX, testX = ad.split_along_row(X, num_train)
    trainY, testY = ad.split_along_row(Y, num_train)
    trainR, testR = ad.split_along_row(R, num_train)

    # process data for the NN
    numFeatures = len(F[0]) + 1     # distance included
    F_DIST = ad.combine_DIST_F(F, DIST, J, numFeatures)
    numFeatures += 1                # for reward later

    # change the input data into pytorch tensors
    trainX, trainY, trainR = torchten(trainX), torchten(trainY), torchten(trainR)
    testX, testY, testR = torchten(testX), torchten(testY), torchten(testR)
    F_DIST = torchten(F_DIST)

    trainX = Variable(trainX, requires_grad=False)
    trainY = Variable(trainY, requires_grad=False)
    testX = Variable(testX, requires_grad=False)
    testY = Variable(testY, requires_grad=False)

# ==========================================================
# MAIN PROGRAM
# ==========================================================
if __name__ == "__main__":
    # READY!!
    read_set_data()
    net = MyNet(J, numFeatures, args.eta)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    # SET!!
    if args.cuda:
        # transfer net and tensors to the gpu
        net.cuda()
        trainX, trainY, trainR = trainX.cuda(), trainY.cuda(), trainR.cuda()
        testX, testY, testR = testX.cuda(), testY.cuda(), testR.cuda()
        F_DIST = F_DIST.cuda()
        file_pre_gpu = "gpu, "
    else:
        file_pre_gpu = "cpu, "

    # scalar + tensor currently not supported in pytorch
    train_loss_normalizer = (trainY - torch.mean(trainY).expand_as(trainY)).pow(2).sum()
    test_loss_normalizer = (testY - torch.mean(testY).expand_as(testY)).pow(2).sum()
    # print(train_loss_normalizer)
    # GO!!
    train_time_loss, test_time_loss, total_time = [], [], 0.0
    for e in xrange(1, args.epochs + 1):
        # train
        train_res = train(net, optimizer, train_loss_normalizer)
        train_time_loss.append(train_res)
        print(train_res[1])
        # test
        test_res = test(net, test_loss_normalizer)
        test_time_loss.append(test_res)
        print(test_res[1])
        total_time += (train_res[0] + test_res[0])

    # FINISH!!
    # log and plot the results: epoch vs loss
    if args.rand_xyr:
        file_pre = "randXYR_epochs=%d, " % (args.epochs)
    else:
        file_pre = "origXYR_epochs=%d, " % (args.epochs)

    log_name = "lr=%.3e, mom=%.3f, eta=%.3f, lam=%.3f, time=%.4f sec" % (
        args.lr, args.momentum, args.eta, args.lambda_L1, total_time)
    
    epoch_data = np.arange(1, args.epochs + 1)

    save_plot("./plots/" + file_pre_gpu + file_pre + log_name + ".png",
        epoch_data, [train_time_loss, test_time_loss],
        "epoch", "loss", log_name)
    save_log("./logs/" + file_pre_gpu + file_pre + log_name + ".txt",
        epoch_data, [train_time_loss, test_time_loss], log_name)