from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 2)

    def forward(self, inp):
        #inp = self.conv1(inp)
        print("after fc1, ", self.fc1(inp))
        inp = self.fc1(inp)
        print("after softmax, ", nn.functional.softmax(inp))
        return nn.functional.softmax(inp)

m = Net()
#data = torch.FloatTensor(10, 5)
data = torch.randn(10, 5)
print("data, ", data)
res = m(Variable(data))
print("res, ", res)