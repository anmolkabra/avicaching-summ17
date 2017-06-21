#!/usr/bin/env python
from __future__ import print_function
import torch, torch.nn as nn, torch.nn.functional as torchfun, torch.optim as optim
from torch.autograd import Variable
import numpy as np, argparse, time, os, sys
import avicaching_data as ad

# =============================================================================
# options
# =============================================================================
parser = argparse.ArgumentParser(description="NN Avicaching model for finding rewards")
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
parser.add_argument("--rewards", type=float, default=1000.0, metavar="R",
    help="inputs the total budget of rewards to be distributed (default=1000.0)")
parser.add_argument("--weights-file", type=str, 
    default="./stats/weights/normalizedR_gpu, origXYR_epochs=1000, train= 80%, time=98.6947 sec.txt", 
    metavar="f", help="inputs the location of the file to use weights from")
parser.add_argument("--log-interval", type=int, default=1, metavar="I",
    help="prints training information at I epoch intervals (default=1)")
parser.add_argument("--expand-R", action="store_true", default=False,
    help="expands the reward vectors into matrices with distributed rewards")

args = parser.parse_args()
# assigning cuda check and test check to single variables
args.cuda = not args.no_cuda and torch.cuda.is_available()

# ====
# parameters and constants
# =====
J, T, weights_file_name, totalR = args.locations, args.time, args.weights_file, args.rewards
X, W_for_r, F_DIST, numFeatures = [], [], [], 0
F_DIST_weighted = []
torchten = torch.DoubleTensor

# ==========
# data input
# =========
def read_set_data():
    global X, W, F_DIST, numFeatures, F_DIST_weighted, W_for_r
    # read f and dist datasets from file, operate on them
    F = ad.read_F_file("./data/loc_feature_with_avicaching_combined.csv", J)
    DIST = ad.read_dist_file("./data/site_distances_km_drastic_price_histlong_0327_0813_combined.txt", J)

    # read W and X
    W = ad.read_weights_file(weights_file_name, J)
    X, _, _ = ad.read_XYR_file("./data/density_shift_histlong_as_previous_loc_classical_drastic_price_0327_0813.txt", J, T)

    # process data for the NN
    F, DIST = ad.normalize(F, along_dim=0, using_max=True), ad.normalize(DIST, using_max=True)  # normalize using max
    numFeatures = len(F[0]) + 1     # distance included
    F_DIST = torchten(ad.combine_DIST_F(F, DIST, J, numFeatures))
    numFeatures += 1                # for rewards later
    
    # split W and join the multiply the fdist portion with F_DIST
    W = np.expand_dims(W, axis=2)
    W_for_fdist, W_for_r = ad.split_along_dim(W, numFeatures - 1, dim=1)
    F_DIST_weighted = Variable(torch.bmm(F_DIST, torchten(W_for_fdist)).squeeze(dim=2), requires_grad=False)

    # condense X along T into a single vector and normalize
    X = ad.normalize(X.sum(axis=0), using_max=False)
    
    W_for_r, X = Variable(torchten(W_for_r), requires_grad=False), Variable(torchten(X), requires_grad=False)

# ==============
# MyNet class
# ==================
class MyNet(nn.Module):

    def __init__(self, J, totalR, eta):
        super(MyNet, self).__init__()
        self.J, self.totalR, self.eta = J, totalR, eta

        # initiate R
        self.r = np.random.multinomial(self.totalR, [1 / float(J)] * J, size=1)
        # self.r = ad.normalize(self.r, using_max=False)
        self.R = nn.Parameter(torchten(self.r))
        print(self.R)

    def forward(self, inp):
        repeatedR = self.R.repeat(J, 1).unsqueeze(dim=2)
        inp = torch.bmm(repeatedR, W_for_r).view(-1, J)
        inp += F_DIST_weighted
        inp = torchfun.relu(inp)
        # add eta to inp[u][u]
        # eta_matrix = Variable(self.eta * torch.eye(J).type(torchten))
        # if args.cuda:
        #    eta_matrix = eta_matrix.cuda()
        # inp += eta_matrix
        return torchfun.softmax(inp)

def train(net, optimizer):
    global W_for_r
    start_time = time.time()

    # build input
    #inp = build_input()
    if args.cuda:
        W_for_r = W_for_r.cuda()
    #inp = Variable(inp)
    
    # feed in data
    P = net(W_for_r).t()    # P is now weighted -> softmax
    
    # calculate loss
    Y = torch.mv(P, X)
    loss = (Y - torch.mean(Y).expand_as(Y)).pow(2).sum() / J

    # print(net.R.grad)

    # backpropagate
    optimizer.zero_grad()
    loss.backward()
    
    # take the projection of the gradient so that everything adds up to totalR
    # gradients -> gradients - mean(gradients)

    #R_grad = net.R.grad.data
    #print(R_grad)
    #print(net.R.grad.data)
    net.R.grad.data = net.R.grad.data - torch.mean(net.R.grad.data)
    #print(net.R.grad.data)
    #print(torch.sum(net.R.grad.data))
    optimizer.step()
    #print(net.R.grad.data)

    # min R must be 0
    net.R.data = net.R.data.clamp(min=0)
    
    end_time = time.time()
    return (end_time - start_time, loss.data[0])

# =============================================================================
# utility functions for training and testing routines
# =============================================================================
def build_input(rt):
    """
    Builds the final input for the NN. Joins F_DIST and expanded R
    """
    if args.expand_R:
        return torch.cat([F_DIST, rt.repeat(J, 1, 1)], dim=2)
    return torch.cat([F_DIST, rt.repeat(J, 1)], dim=2)

# =============================================================================
# main program
# =============================================================================
if __name__ == "__main__":
    read_set_data()
    net = MyNet(J, totalR, args.eta)
    if args.cuda:
        net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    
    for e in xrange(1, args.epochs + 1):
        train_res = train(net, optimizer)
        if e % 200 == 0:
            print("epoch=%5d, loss=%.10f" % (e, train_res[1]))
    print(net.R.data)
    print(torch.sum(net.R.data, dim=1))