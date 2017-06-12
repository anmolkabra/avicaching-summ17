from __future__ import print_function
import argparse, csv
import numpy as np
import avicaching_data as ad
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

args = parser.parse_args()

# assigning cuda check to a single variable
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}

# parameters and data
J, T, eta, l = args.locations, args.time, args.eta, args.lambda_L1
torchten = torch.DoubleTensor
X, Y, R, DIST, F, NN_in, numFeatures = [], [], [], [], [], [], 0

# Net class
class Net(nn.Module):
    def __init__(self, J, numFeatures):
        """
        Initialiizes the Neural Network, takes in a 2D Tensor of size 
        (J * numFeatures+1) and returns a 1D vector of size J
        """
        super(Net, self).__init__()
        # 2 hidden layers 
        self.fc1 = nn.Linear(numFeatures, 1, bias=False).double()

    def forward(self, inp):
        """
        The Input is propagated forward in the network
        in -> conv1 -> pool -> relu -> conv2 -> dropout -> pool -> relu -> 
        linear -> relu -> linear -> relu -> linear -> softmax -> out
        """
        # go through the layers
        inp = self.fc1(inp)
        #print("after fc1, ", inp)
        # output the softmax
        #print("after softmax, ", torch.nn.functional.softmax(inp.t()))
        return torchfun.softmax(inp.t())

class MyNet(nn.Module):

    def __init__(self, J, numFeatures):
        """
        Initializes MyNet
        """
        super(MyNet, self).__init__()
        self.J = J
        self.numFeatures = numFeatures
        self.w = nn.Parameter(torch.randn(J, numFeatures, 1).type(torchten), requires_grad=True)
        #self.myparameters = nn.ParameterList([self.w])

    def forward(self, inp):
        """
        Multiply the weights and return the softmax
        """
        inp = torch.bmm(inp, self.w).view(-1, self.J)
        return torchfun.softmax(inp)

def train(net, epoch, criterion, optimizer):
    """
    Trains the network using MyNet
    """
    global X, Y, R, NN_in, J, numFeatures, T, args

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    X, Y = Variable(X, requires_grad=False), Variable(Y, requires_grad=False)
    #P = Variable(torchten(J, J), requires_grad=True)
    net = MyNet(J, numFeatures)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.MSELoss(size_average=False)

    for e in xrange(epoch):
        loss = 0
        #print(w_t.grad.data[0])
        
        for t in xrange(T):
            # build the input by appending R[t] and patch into the net
            R_extended = R[t].repeat(J, 1)
            inp = torch.cat([NN_in, R_extended], dim=2)     # final NN_in_processing
            inp = Variable(inp, requires_grad=True)
            
            P = net(inp).t()    # P is now weighted -> softmax
            Pxt = torch.mv(P, X[t])
            #print("Y[t] - Pxt", Y[t] - Pxt)
            loss += (Y[t] - Pxt).pow(2).sum()
            #loss += criterion(Y[t], Pxt)
        
        print("epoch = %d, loss = %.10f" % (e, loss.data[0]), )
        optimizer.zero_grad()
        loss.backward()
        #print("w.grad.data[0] = ", net.w.grad.data[0])
        #w_t.data -= args.lr * w_t.grad.data
        #print("before change: w.data[0] = ", net.w.data[0])
        #w_t.grad.data.zero_()
        optimizer.step()
        #print("after change: w.data[0][0] = %.10f" % (net.w.data[0][0][0]))
        #print(loss.data[0])

    # for batch_idx, (data, target) in enumerate(train_data):
    #     # for each batch
        
    #     if args.cuda:
    #         data, target = data.cuda(), target.cuda()
    #     data, target = Variable(data), Variable(target)
        
    #     # go forward in the network, process to get outputs
    #     # the output is a vector of u_i for a single v with softmax applied
    #     output = net(data)
    #     # backpropagate
    #     optimizer.zero_grad()
    #     loss = torch.nn.MSELoss(output, target)     # calculate loss
    #     loss.backward()                 # backpropagate loss
    #     optimizer.step()                # update the weights

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
    global X, Y, R, F, DIST, J, T, numFeatures
    X, Y, R = ad.read_XYR_file("./density_shift_histlong_as_previous_loc_classical_drastic_price_0327_0813.txt", J, T)
    F = ad.read_Fu_file("./loc_feature_with_avicaching_combined.csv")
    DIST = ad.read_dist_file("./site_distances_km_drastic_price_histlong_0327_0813_combined.txt", J)

def use_rand_data():
    global X, Y, R, F, DIST, J, T, numFeatures
    X, Y, R = ad.read_XYR_file("./randXYR.txt", J, T)
    F = ad.read_Fu_file("./loc_feature_with_avicaching_combined.csv")
    DIST = ad.read_dist_file("./site_distances_km_drastic_price_histlong_0327_0813_combined.txt", J)

if __name__ == "__main__":
    # ---------------- process data
    c = int(raw_input("Press 0 to use original data, 1 to use changed data: "))
    if c == 0:
        use_orig_data()
    elif c == 1:
        use_rand_data()

    # process data for the NN
    numFeatures = len(F[0]) + 1     # distance included
    NN_in = ad.combine_DIST_F(F, DIST, J, numFeatures)
    numFeatures += 1                # for reward later

    # change the input data into pytorch nn variables
    X, Y, R = torchten(X), torchten(Y), torchten(R)
    F, DIST = torchten(F), torchten(DIST)
    NN_in = torchten(NN_in)
    # print(X)
    # print(F)
    # print(R)
    # print(DIST)
    # print(NN_in)

    # ---------------- calculations
    net = Net(J, numFeatures)
    if args.cuda:
        # move the network to the GPU, if CUDA supported
        net.cuda()

    # using SGD as the optimizer function
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    train(net, args.epochs + 1, criterion, optimizer)
    # test(net, epoch)