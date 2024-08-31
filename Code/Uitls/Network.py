import torch
import torch.nn as nn
from collections import OrderedDict


class myDNN(torch.nn.Module):
    def __init__(self, layers,lastbias=False):
        super(myDNN, self).__init__()
        # parameters
        self.depth = len(layers) - 1
        # set up layer order dict
        self.activation = torch.nn.ReLU
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1], bias=lastbias)) #
        )
        layerDict = OrderedDict(layer_list)
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


# RK4GRUcell
class myRK4GRUcell(nn.Module):
    def __init__(self, args):
        super(myRK4GRUcell, self).__init__()
        self.layers = args.layers
        self.dt = args.dt
        self.DNN = myDNN(self.layers, lastbias=True)
        self.reset_gate =  myDNN(args.GRU_layers, lastbias=True)
        self.update_gate = myDNN(args.GRU_layers, lastbias=True)

    def forward(self, SVi, exci=None, excj=None):
        if SVi.dim() == 1:
            SVi = SVi.unsqueeze(0).unsqueeze(0)
        exch = 0.5 * (exci + excj)
        r = torch.sigmoid(self.reset_gate(SVi))
        SVi_r = r*SVi
        # K1
        SVK_INPUT = SVi_r; SVEXC = exci
        K1 = torch.cat(( SVK_INPUT[:,:,1:2],
                               -SVK_INPUT[:,:,2:3] - SVEXC,
                                self.DNN(torch.cat((SVK_INPUT,-SVK_INPUT[:,:,2:3]-SVEXC),-1)) ) ,-1)
        # K2
        SVK_INPUT = SVi_r+(self.dt/2)*K1; SVEXC = exch
        K2 = torch.cat(( SVK_INPUT[:,:,1:2],
                               -SVK_INPUT[:,:,2:3] - SVEXC,
                                self.DNN(torch.cat((SVK_INPUT,-SVK_INPUT[:,:,2:3]-SVEXC),-1)) ) ,-1)
        # K3
        SVK_INPUT = SVi_r+(self.dt/2)*K2; SVEXC = exch
        K3 = torch.cat(( SVK_INPUT[:,:,1:2],
                               -SVK_INPUT[:,:,2:3] - SVEXC,
                                self.DNN(torch.cat((SVK_INPUT,-SVK_INPUT[:,:,2:3]-SVEXC),-1)) ) ,-1)
        # K4
        SVK_INPUT = SVi_r+ self.dt *K3; SVEXC = excj
        K4 = torch.cat(( SVK_INPUT[:,:,1:2],
                               -SVK_INPUT[:,:,2:3] - SVEXC,
                                self.DNN(torch.cat((SVK_INPUT,-SVK_INPUT[:,:,2:3]-SVEXC),-1)) ) ,-1)
        # RK4
        SVj = SVi_r+(self.dt/6)*(K1+2*K2+2*K3+K4)
        z = torch.sigmoid(self.update_gate(SVi))
        SVj_z = z* SVj + (1-z)*SVi
        return SVj_z,r,z










