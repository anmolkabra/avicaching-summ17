#!/usr/bin/env python
from __future__ import print_function
import argparse, time
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
parser.add_argument("--batch-size", type=int, default=64, metavar="B",
    help="inputs batch size for training (default=64)")
parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
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
    help="inputs total time of data collection; number of weeks (default=10)")
parser.add_argument("--eta", type=float, default=10.0, metavar="F",
    help="inputs parameter eta in the model (default=10.0)")
parser.add_argument("--lambda-L1", type=float, default=10.0, metavar="LAM",
    help="inputs the L1 regularizing coefficient")
parser.add_argument("--rand-xyr", action="store_true", default=False,
    help="uses random xyr data")
parser.add_argument("--log-interval", type=int, default=10, metavar="I",
    help="prints training information at I epoch intervals (default=10)")
parser.add_argument("--save-plot", action="store_true", default=False,
    help="saves the plot instead of opening it")


args = parser.parse_args()

# assigning cuda check to a single variable
args.cuda = not args.no_cuda and torch.cuda.is_available()

# parameters and data
J, T = args.locations, args.time
torchten = torch.DoubleTensor
X, Y, R, DIST, F, NN_in, numFeatures = [], [], [], [], [], [], 0
orig, rand = True, False

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
        return torchfun.softmax(inp + 1)

def train(net, epochs, optimizer):
    """
    Trains the network using MyNet
    """
    global X, Y, R, NN_in, J, numFeatures, T, args, orig, rand
    loss_data = []
    start_time = time.time()

    if args.cuda:
        X, Y = X.cuda(), Y.cuda()
        file_pre_gpu = "gpu, "
    else:
        file_pre_gpu = "cpu, "

    X, Y = Variable(X, requires_grad=False), Variable(Y, requires_grad=False)
    # scalar + tensor currently not supported in pytorch
    loss_normalizer_Y_mean = (Y - torch.mean(Y).expand_as(Y)).pow(2).sum().data[0]
    
    for e in xrange(epochs):
        loss = 0
        for t in xrange(T):
            # build the input by appending R[t]
            R_extended = R[t].repeat(J, 1)
            inp = torch.cat([NN_in, R_extended], dim=2)     # final NN_in_processing
            
            # standardize inp
            diff = inp - torch.mean(inp, dim=2).expand_as(inp)
            std = torch.std(inp, dim=2).expand_as(inp)
            inp = torch.div(diff, std)

            if args.cuda:
                inp = inp.cuda()
            inp = Variable(inp)
            
            # feed in data
            P = net(inp).t()    # P is now weighted -> softmax
            
            # calculate loss
            Pxt = torch.mv(P, X[t])
            loss += (Y[t] - Pxt).pow(2).sum()
        
        # loss += args.lambda_L1 * torch.norm(net.w.data)
        loss /= loss_normalizer_Y_mean
            
        loss_data.append(loss.data[0])
        
        # backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_time = time.time()
    # log and plot the results: epoch vs loss
    name = "lr=%.4e, mom=%.4f, eta=%.4f, lam=%.4f, time=%.5f sec" % (
        args.lr, args.momentum, args.eta, args.lambda_L1, end_time - start_time)
    
    if args.rand_xyr:
        file_pre = "randXYR_epochs=%d, " % (epochs)
    else:
        file_pre = "origXYR_epochs=%d, " % (epochs)
    
    epoch_data = np.arange(0, epochs)
    save_plot("./plots/" + file_pre_gpu + file_pre + name + ".png",
        epoch_data, loss_data,
        "epoch", "loss", name)
    save_log("./logs/" + file_pre_gpu + file_pre + name + ".txt",
        epoch_data, loss_data,
        name)

def test(net, epoch):
    """
    Test the network, unsure if needed
    """
    net.eval()  # comment out if dropout or batchnorm module not used
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size

def use_orig_data():
    global X, Y, R, F, DIST, J, T
    X, Y, R = ad.read_XYR_file("./density_shift_histlong_as_previous_loc_classical_drastic_price_0327_0813.txt", J, T)
    F = ad.read_F_file("./loc_feature_with_avicaching_combined.csv")
    DIST = ad.read_dist_file("./site_distances_km_drastic_price_histlong_0327_0813_combined.txt", J)

def use_rand_data():
    global X, Y, R, F, DIST, J, T
    X, Y, R = ad.read_XYR_file("./randXYR.txt", J, T)
    F = ad.read_F_file("./loc_feature_with_avicaching_combined.csv")
    DIST = ad.read_dist_file("./site_distances_km_drastic_price_histlong_0327_0813_combined.txt", J)

def save_plot(file_name, x, y, xlabel, ylabel, title):
    global args
    plt.plot(x, y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    if args.save_plot:
        plt.savefig(file_name, bbox_inches="tight", dpi=200)
    else:
        plt.show()

def save_log(file_name, x, y, title):
    global args
    with open(file_name, "wt") as f:
        f.write(title + "\n")
        for i in range(0, len(x), args.log_interval):
            f.write("epoch = %d, loss = %.10f\n" % (x[i], y[i]))
        f.write("epoch = %d, loss = %.10f\n" % (x[-1], y[-1]))


# ==========================================================
# MAIN PROGRAM
# ==========================================================

# ---------------- process data ----------------------------
# ----------------------------------------------------------
if args.rand_xyr:
    use_rand_data()
else:
    use_orig_data()

# process data for the NN
numFeatures = len(F[0]) + 1     # distance included
NN_in = ad.combine_DIST_F(F, DIST, J, numFeatures)
numFeatures += 1                # for reward later

# change the input data into pytorch nn variables
X, Y, R = torchten(X), torchten(Y), torchten(R)
F, DIST = torchten(F), torchten(DIST)
NN_in = torchten(NN_in)

# ---------------- train and test -------------------------
# ---------------------------------------------------------
net = MyNet(J, numFeatures, args.eta)
if args.cuda:
    net.cuda()

# optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
# optimizer = optim.Adam(net.parameters(), lr=args.lr)

train(net, args.epochs, optimizer)
# test(net, epoch)