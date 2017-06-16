#!/usr/bin/env python
from __future__ import print_function
import argparse, time, math, os, sys, numpy as np, matplotlib
try:
    os.environ["DISPLAY"]
except KeyError as e:
    # working without X/GUI environment
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import avicaching_data as ad
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
parser.add_argument("--momentum", type=float, default=1.0, metavar="M",
    help="inputs SGD momentum (default=1.0)")
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

args = parser.parse_args()

# assigning cuda check to a single variable
args.cuda = not args.no_cuda and torch.cuda.is_available()

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
        self.w = nn.Parameter(torch.randn(J, numFeatures, 1).type(torchten) * 10.0)

    def forward(self, inp):
        """
        Forward in the network; multiply the weights and return the softmax
        """
        
        inp = torch.bmm(inp, self.w).view(-1, self.J)
        # for u in xrange(len(inp)):
        #     inp[u, u] = inp[u, u].clone() + self.eta    # inp[u][u]
        return torchfun.softmax(inp)

def standard(x):
    """
    Standardizes a 3D Tensor x with mean 0 and std 1
    """
    diff = x - torch.mean(x, dim=2).expand_as(x)
    std = torch.std(x, dim=2).expand_as(x)
    return torch.div(diff, std)

def build_input(rt):
    # join fdist and expanded r
    return torch.cat([F_DIST, rt.repeat(J, 1, 1)], dim=2)

def expand_R(rt, R_max=15):
    # rt[u] must be expanded into a vector of 15 elements
    # which should contain rt[u] ones in the beginning
    # followed by zeros.
    newrt = torchten(J, 15)
    if args.cuda:
        newrt = newrt.cuda()
    for u in xrange(J):
        r = int(rt[u])
        newrt[u] = torch.cat([torch.ones(r), 
            torch.zeros(R_max - r)], dim=0)
    return newrt

def train(net, optimizer, loss_normalizer):
    """
    Trains the network using MyNet
    """
    loss = 0
    start_time = time.time()
    
    for t in xrange(num_train):
        # build the input by appending trainR[t] to F_DIST
        inp = build_input(trainR[t])
        if args.cuda:
            inp = inp.cuda()
        inp = Variable(standard(inp))   # standardize inp
    
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
        inp = build_input(testR[t])
        if args.cuda:
            inp = inp.cuda()
        inp = Variable(standard(inp))   # standardize inp
        
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
    # get the losses from data
    train_losses = [i for j in y[0] for i in j][1::2]
    test_losses = [i for j in y[1] for i in j][1::2]
    
    # plot details
    train_label, = plt.plot(x, train_losses, "r-", label="Train Loss") 
    test_label, = plt.plot(x, test_losses, "b-", label="Test Loss")
    plt.ylabel(ylabel)
    plt.grid(True, which="major", axis="both", color="k", ls="dotted", lw="1.0")
    plt.grid(True, which="minor", axis="y", color="k", ls="dotted", lw="0.5")
    plt.minorticks_on()
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
        for i in range(0, len(x), args.log_interval):
            f.write("epoch = %d\t\ttrainloss = %.8f, traintime = %.4f\t\ttestloss = %.8f, testtime = %.4f\n" % (
                    x[i], y[0][i][1], y[0][i][0],
                    y[1][i][1], y[1][i][0]))

def read_set_data():
    """
    Reads and sets up the datasets
    """
    global trainX, trainY, trainR, testX, testY, testR, F_DIST, numFeatures
    # read f and dist datasets from file, operate on them
    F = ad.read_F_file("./data/loc_feature_with_avicaching_combined.csv")
    DIST = ad.read_dist_file("./data/site_distances_km_drastic_price_histlong_0327_0813_combined.txt", J)

    # process data for the NN
    numFeatures = len(F[0]) + 1     # distance included
    F_DIST = torchten(ad.combine_DIST_F(F, DIST, J, numFeatures))
    numFeatures += 15               # for rewards later

    # operate on XYR data
    if args.rand_xyr:
        if not os.path.isfile("./data/randXYR.txt"):
            # file doesn't exists, make random data, write to file
            X, Y, R = make_rand_data()
            ad.save_rand_XYR("./data/randXYR.txt", X, Y, R, J, T)
        #
        # print("Verifying randXYR...")
        # X, Y, R = ad.read_XYR_file("./randXYR.txt", J, T)
        # w = ad.read_weights_file("./randXYR_weights.txt", J)
        # X, Y, R, w = Variable(torchten(X)), Variable(torchten(Y)), torchten(R), torchten(w)
        # w = Variable(torch.unsqueeze(w, dim=2))   # make w a 3d tensor
        
        # test_given_data(X, Y, R, w, J, T)
        #
        X, Y, R = ad.read_XYR_file("./data/randXYR.txt", J, T)
    else:
        X, Y, R = ad.read_XYR_file("./data/density_shift_histlong_as_previous_loc_classical_drastic_price_0327_0813.txt", J, T)
    # split the XYR data
    if args.train_percent != 1.0:
        # training and testing, shuffle and split the data
        shuffle_order = np.random.permutation(T)
        trainX, testX = ad.split_along_row(X[shuffle_order], num_train)
        trainY, testY = ad.split_along_row(Y[shuffle_order], num_train)
        trainR, testR = ad.split_along_row(R[shuffle_order], num_train)
    else:
        # no testing, split the data only to be merged later (bad code)
        trainX, testX = ad.split_along_row(X, num_train)
        trainY, testY = ad.split_along_row(Y, num_train)
        trainR, testR = ad.split_along_row(R, num_train)

    # change the input data into pytorch tensors and variables
    trainR, testR = torchten(trainR), torchten(testR)
    trainX = Variable(torchten(trainX), requires_grad=False)
    trainY = Variable(torchten(trainY), requires_grad=False)
    testX = Variable(torchten(testX), requires_grad=False)
    testY = Variable(torchten(testY), requires_grad=False)

    # expand R (trainR and testR)
    trainR_ext = torchten(num_train, J, 15)
    testR_ext = torchten(num_test, J, 15)
    for t in xrange(num_train):
        trainR_ext[t] = expand_R(trainR[t])
    for t in xrange(num_test):
        testR_ext[t] = expand_R(testR[t])
    trainR, testR = trainR_ext, testR_ext

    print(trainX)
    print(trainY)
    print(trainR)

def make_rand_data(X_max=100.0, R_max=100.0):
    """
    Creates random X and R and calculates Y based on random weights
    """
    # create random X and R and w
    X = np.floor(np.random.rand(T, J) * X_max)
    R = torchten(np.floor(np.random.rand(T, J) * R_max))
    w = Variable(torch.randn(J, numFeatures, 1))
    Y = np.empty([T, J])
    X, Y = Variable(torchten(X), requires_grad=False), Variable(torchten(Y), requires_grad=False)
    if args.cuda:
        X, Y, R = X.cuda(), Y.cuda(), R.cuda()
    
    # build Y
    for t in xrange(T):
        # build the input by appending testR[t]
        inp = build_input(R[t])
        if args.cuda:
            inp = inp.cuda()
        inp = Variable(standard(inp))   # standardize inp
        
        # feed in data
        inp = torch.bmm(inp, w).view(-1, J)
        # for u in xrange(len(inp)):
        #     inp[u, u] = inp[u, u].clone() + self.eta    # inp[u][u]
        P = torchfun.softmax(inp).t()   # P is now weighted -> softmax
        
        # calculate Y
        Y[t] = torch.mv(P, X[t])

    # for verification of random data, save weights ---------------------------
    w_matrix = w.data.view(-1, numFeatures).numpy()
    np.savetxt("./data/randXYR_weights.txt", w_matrix, fmt="%.15f", delimiter=" ")
    # -------------------------------------------------------------------------

    return (X.data.numpy(), Y.data.numpy(), R.numpy())

def test_given_data(X, Y, R, w, J, T):
    print(w.data.view(-1, numFeatures))
    loss_normalizer = (Y - torch.mean(Y).expand_as(Y)).pow(2).sum().data[0]
    loss = 0

    for t in xrange(T):
        # build the input by appending testR[t]
        inp = build_input(R[t])
        if args.cuda:
            inp = inp.cuda()
        inp = Variable(standard(inp))   # standardize inp
        
        # feed in data
        inp = torch.bmm(inp, w).view(-1, J)
        # for u in xrange(len(inp)):
        #     inp[u, u] = inp[u, u].clone() + self.eta    # inp[u][u]
        P = torchfun.softmax(inp).t()   # P is now weighted -> softmax
        
        # calculate loss
        Pxt = torch.mv(P, X[t])
        loss += (Y[t] - Pxt).pow(2).sum() 
    # loss += args.lambda_L1 * torch.norm(net.w.data)
    loss /= loss_normalizer
    print("Loss = %f" % loss.data[0])

# ==========================================================
# MAIN PROGRAM
# ==========================================================
if __name__ == "__main__":
    # READY!!
    read_set_data()
    net = MyNet(J, numFeatures, args.eta)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
    # optimizer = optim.Adagrad(net.parameters(), lr=args.lr)
    # optimizer = optim.Adamax(net.parameters(), lr=args.lr)
    optimizer = optim.Rprop(net.parameters(), lr=args.lr)
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
    #
    if args.train_percent == 1.0:
        # verifying that new random weights converge to stored ones
        data = []
        weights_before = net.w.data.view(-1, numFeatures).cpu().numpy()
        for e in xrange(1, args.epochs + 1):
            train_res = train(net, optimizer, train_loss_normalizer)
            if e % 200 == 0:
                print("e= %d,  loss=%.8f" % (e, train_res[1]))
            data.append([e, train_res[1]])
        data = np.array(data)
        weights = net.w.data.view(-1, numFeatures).cpu().numpy()
        curr_time = time.localtime()
        fname = "./stats/recovering_weights/w_" + str(curr_time.tm_year) + str(curr_time.tm_mon) + str(curr_time.tm_mday) + "_" + str(time.time()) + ".txt"
        with open(fname, "w") as f:
            np.savetxt(f, weights_before, fmt="%.15f")
            np.savetxt(f, data, fmt="%.8f")
            np.savetxt(f, weights, fmt="%.15f")
        sys.exit(0)
    #
    test_loss_normalizer = (testY - torch.mean(testY).expand_as(testY)).pow(2).sum()
    
    # GO!!
    train_time_loss, test_time_loss, total_time = [], [], 0.0
    #
    print("Testing loss before training...")
    test_given_data(
        torch.cat([trainX, testX]),
        torch.cat([trainY, testY]),
        torch.cat([trainR, testR]),
        net.w, J, T)
    #
    for e in xrange(1, args.epochs + 1):
        # train
        train_res = train(net, optimizer, train_loss_normalizer)
        train_time_loss.append(train_res)
        
        # test
        test_res = test(net, test_loss_normalizer)
        test_time_loss.append(test_res)
        
        total_time += (train_res[0] + test_res[0])
        
    # FINISH!!
    print(net.w.data.view(-1, numFeatures))
    # log and plot the results: epoch vs loss
    if args.rand_xyr:
        file_pre = "randXYR_epochs=%d, " % (args.epochs)
    else:
        file_pre = "origXYR_epochs=%d, " % (args.epochs)

    log_name = "lr=%.3e, mom=%.3f, eta=%.3f, lam=%.3f, time=%.4f sec" % (
        args.lr, args.momentum, args.eta, args.lambda_L1, total_time)
    
    epoch_data = np.arange(1, args.epochs + 1)

    save_plot("./stats/plots/" + file_pre_gpu + file_pre + log_name + ".png",
        epoch_data, [train_time_loss, test_time_loss],
        "epoch", "loss", log_name)
    save_log("./stats/logs/" + file_pre_gpu + file_pre + log_name + ".txt",
        epoch_data, [train_time_loss, test_time_loss], log_name)
