#!/usr/bin/env python
#
# Accoring to Dynamic pooling and RODAN code
# Author: Xuecheng Zhang
#

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import h5py
from torch.autograd import Variable
import sys
import argparse
import shutil
# import pickle
import os
import copy
import math
from itertools import product
from collections import OrderedDict
import ont
from torch.onnx import export
#from warp_rna import rna_loss
# from ranger21 import Ranger21
iterations = 0
# configuration
defaultconfig = {"layer": 0,"name": "default", "seqlen": 4096, "epochs": 30, "optimizer": "ranger", "lr": 3e-3, "weightdecay": 0.01, "batchsize": 3, "dropout": 0.1, "activation": "mish", "sqex_activation": "mish", "sqex_reduction": 32, "trainfile": "rna-train.hdf5", "validfile": "rna-valid.hdf5", "amp": False, "scheduler": "reducelronplateau", "scheduler_patience": 1, "scheduler_factor": 0.5, "scheduler_threshold": 0.1, "scheduler_minlr": 1e-05, "scheduler_reduce": 2, "gradclip": 0, "train_loopcount": 1000000, "valid_loopcount": 1000, "saveinit": False, "dp_dropout":0.05, "dp_activation":"glu","vocab": ['<PAD>', 'A', 'C', 'G', 'T']}


# the third layer has a 10 stride
rna_default = [[-1, 256, 0, 3, 1, 1, 0], [-1, 256, 1, 10, 1, 1, 1], [-1, 256, 1, 10, 10, 1, 1], [-1, 320, 1, 10, 1, 1, 1], [-1, 384, 1, 15, 1, 1, 1], [-1, 448, 1, 20, 1, 1, 1], [-1, 512, 1, 25, 1, 1, 1], [-1, 512, 1, 30, 1, 1, 1], [-1, 512, 1, 35, 1, 1, 1], [-1, 512, 1, 40, 1, 1, 1], [-1, 512, 1, 45, 1, 1, 1], [-1, 512, 1, 50, 1, 1, 1], [-1, 768, 1, 55, 1, 1, 1], [-1, 768, 1, 60, 1, 1, 1], [-1, 768, 1, 65, 1, 1, 1], [-1, 768, 1, 70, 1, 1, 1], [-1, 768, 1, 75, 1, 1, 1], [-1, 768, 1, 80, 1, 1, 1], [-1, 768, 1, 85, 1, 1, 1], [-1, 768, 1, 90, 1, 1, 1], [-1, 768, 1, 95, 1, 1, 1], [-1, 768, 1, 100, 1, 1, 1]]
dna_default = [[-1, 320, 0, 3, 1, 1, 0], [-1, 320, 1, 3, 3, 1, 1], [-1, 384, 1, 6, 1, 1, 1], [-1, 448, 1, 9, 1, 1, 1], [-1, 512, 1, 12, 1, 1, 1], [-1, 576, 1, 15, 1, 1,
                                                                                                                                                   1], [-1, 640, 1, 18, 1, 1, 1], [-1, 704, 1, 21, 1, 1, 1], [-1, 768, 1, 24, 1, 1, 1], [-1, 832, 1, 27, 1, 1, 1], [-1, 896, 1, 30, 1, 1, 1], [-1, 960, 1, 33, 1, 1, 1]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        self.orig = d


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))


class dataloader(Dataset):
    def __init__(self, recfile="/tmp/train.hdf5", seq_len=4096, index=False, elen=342):
        self.recfile = recfile
        self.seq_len = seq_len
        self.index = index
        h5 = h5py.File(self.recfile, "r")
        self.len = len(h5["events"])
        h5.close()
        self.elen = elen

    def __getitem__(self, index):
        h5 = h5py.File(self.recfile, "r")
        event = h5["events"][index]
        event_len = self.elen
        label = h5["labels"][index]
        label_len = h5["labels_len"][index]
        h5.close()
        if not self.index:
            return (event, event_len, label, label_len)
        else:
            return (event, event_len, label, label_len, index)

    def __len__(self):
        return self.len


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class Mish(nn.Module):  # mish function
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(torch.nn.functional.softplus(x)))


class squeeze_excite(torch.nn.Module):
    def __init__(self, in_channels=512, size=1, reduction="/16", activation=torch.nn.GELU):
        super(squeeze_excite, self).__init__()
        # squeeze part
        self.in_channels = in_channels
        # average adaaptive pooling to change the data to torch.size([1,1,C])
        self.avg = torch.nn.AdaptiveAvgPool1d(1)
        if type(reduction) == str:
            self.reductionsize = self.in_channels // int(reduction[1:])
        else:
            self.reductionsize = reduction  # they use this part
        # excitation part
        self.fc1 = nn.Linear(self.in_channels, self.reductionsize)
        self.activation = activation()  # was nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.reductionsize, self.in_channels)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.activation(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return input * x.permute(0, 2, 1).contiguous()  # follows different orders as 0,2,1


class convblock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, seperable=True, expansion=True, batchnorm=True, dropout=0.1, activation=torch.nn.GELU, sqex=True, squeeze=32, sqex_activation=torch.nn.GELU, residual=True):
        # no bias?
        super(convblock, self).__init__()
        self.seperable = seperable
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.activation = activation
        self.squeeze = squeeze
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        self.doexpansion = expansion
        # fix self.squeeze
        dwchannels = in_channels
        self.dypool = False
        # [-1, 256, 1, 10, 10, 1, 1],
        if stride > 1:
            layer = [4, 256, -1, 9, 1, 0, 1, 0, 1, 3, 0.05, 32]
            paddingarg = layer[0]
            out_channels =  layer[1]
            seperable = layer[2]
            kernel = layer[3]
            stride = layer[4]
            bias = layer[7]
            dilation = layer[8]
            norm = layer[9]
            dropout = layer[10]
            prediction_size = layer[11]
            self.dypool = True

        if seperable:
            if self.doexpansion and self.in_channels != self.out_channels:
                self.expansion = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=1, groups=1, bias=False)
                self.expansion_norm = torch.nn.BatchNorm1d(out_channels)
                self.expansion_act = self.activation()
                dwchannels = out_channels

            if self.dypool:
            # start here when stride = 10
                self.depthwise = dpool(in_channels, out_channels, kernel, stride=stride, padding=paddingarg, dilation=dilation, bias=False, norm=norm, dropout=dropout, activation=torch.nn.GLU, prediction_size=prediction_size)
                self.dypool = False
            else:
                self.depthwise = torch.nn.Conv1d(dwchannels, out_channels, kernel_size=kernel_size,
                                                stride=stride, padding=padding, dilation=dilation, bias=bias, groups=out_channels//groups)
            if self.batchnorm:
                self.bn1 = torch.nn.BatchNorm1d(out_channels)
            self.act1 = self.activation()
            if self.squeeze:
                # stride = 10
                self.sqex = squeeze_excite(
                    in_channels=out_channels, reduction=self.squeeze, activation=sqex_activation)
            self.pointwise = torch.nn.Conv1d(
                out_channels, out_channels, kernel_size=1, dilation=dilation, bias=bias, padding=0)
            if self.batchnorm:
                self.bn2 = torch.nn.BatchNorm1d(out_channels)
            self.act2 = self.activation()
            if self.dropout:
                self.drop = torch.nn.Dropout(self.dropout)
        else:
            # the smoothing layer, add a conv1d without bias
            self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=dilation, bias=bias)
            if self.batchnorm:
                # set the size as output size and the shape is the as input
                self.bn1 = torch.nn.BatchNorm1d(out_channels)
            self.act1 = self.activation()
            if self.squeeze:
                self.sqex = squeeze_excite(
                    in_channels=out_channels, reduction=self.squeeze, activation=sqex_activation)
            if self.dropout:
                self.drop = torch.nn.Dropout(self.dropout)
        if self.residual and self.stride == 1:
            self.rezero = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, x):
        orig = x
        # pointwise expansion
        # Batch norm
        # activation
        # depthwise conv
        # batch norm
        # activation mish
        # squeeze/excite linear
        # pointwise conv
        # batchnorm
        # activation
        if self.seperable:
            #  pointwise expansion + Batch norm + activation
            if self.in_channels != self.out_channels and self.doexpansion:
                x = self.expansion(x)
                x = self.expansion_norm(x)
                x = self.expansion_act(x)
            # deep wise
            x = self.depthwise(x)
            # batch norm
            if self.batchnorm:
                x = self.bn1(x)
            x = self.act1(x)
            if self.squeeze:
                x = self.sqex(x)
            x = self.pointwise(x)
            if self.batchnorm:
                x = self.bn2(x)
            x = self.act2(x)
            if self.dropout:
                x = self.drop(x)
        else:
            # the first convolution
            x = self.conv(x)
            # x = self.dpool(x)
            if self.batchnorm:
                x = self.bn1(x)
            x = self.act1(x)
            if self.dropout:
                x = self.drop(x)

        if self.residual and self.stride == 1 and self.in_channels == self.out_channels and x.shape[2] == orig.shape[2]:
            return orig + self.rezero * x  # rezero
            # return orig + x # normal residual
        else:
            return x

############# The structure of dynamic pooling#################################################################        
class dpool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, norm=None, dropout=0.05, activation=torch.nn.GLU, prediction_size=32):
        super(dpool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.kernel_size = kernel_size
        # get when the layer conresponds to the number of channels.
        # Directly add a TCS convolution to the first layer followed by a batchnorm
        # the block is to train three paoerameters which are feature vector, weight vector and length vector
        self.predictor = torch.nn.Sequential(
            torch.nn.Conv1d(2, prediction_size, 31, stride=1, padding=15),
            torch.nn.BatchNorm1d(prediction_size),
            torch.nn.SiLU(),
            torch.nn.Conv1d(prediction_size, prediction_size,
                            15, stride=1, padding=7),
            torch.nn.BatchNorm1d(prediction_size),
            torch.nn.SiLU(),
            torch.nn.Conv1d(prediction_size, 2, 15, stride=1, padding=7)
        )
        self.conv1x1 = nn.Conv1d(
            in_channels, 1, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias)

        # initialise the weight and bias
        self.predictor[-1].weight.data *= 0.01
        self.predictor[-1].bias.data *= 0
        self.predictor[-1].bias.data[0] -= np.log(2)
        # initialise the bias and weight
        self.predictor[-1].bias.data[1] -= np.log(2)
        self.activation = nn.Sequential(
            *self.get_activation(activation, dropout))  # split into two parts
        self.norm_target = norm

        self.register_buffer('norm_mean', torch.ones(1))

        self.conv1 = nn.Conv1d(
            in_channels, out_channels*2, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.batch_norm1 = torch.nn.BatchNorm1d(
            out_channels*2, eps=1e-3, momentum=0.1)

    def get_activation(self, activation, dropout):
        return activation(dim=1), nn.Dropout(p=dropout)

    def exp(self,x):
        return 1-torch.pow(x,math.e)

    def reci_exp(self,x):
        return 1-torch.pow(x,1.0/math.e)

    def line(self,x):
        return 1-x
    
    def sin(self,x):
        return torch.sin(math.pi/2 * x + math.pi/2) 

    def cos(self,x):
        return torch.cos(math.pi/2 * x + math.pi/2) + 1.0 

    def shift(self,x, k):
        return torch.cat([torch.zeros(k).to(x.device), x[:-k]])

    def shiftb(self, x, k):
        # Add several 0s at the first and move ks to the right
        return torch.cat([torch.zeros((x.shape[0], k)).to(x.device), x[:, :-k]], dim=1)
    
    def big_pool(self, features, moves, weights, func):
        fw = features * weights.unsqueeze(-1) # calculate the weighted features
        poses = torch.cumsum(moves.detach(), 1) + moves - moves.detach() # calculate p_i vector
        # This part is for resolving the divergence problem
        # Since, there may have very huge divergence, we limit the gradient to pass only through 20 elements
        for k in range(1, 21):
            smoves = self.shiftb(moves, k)
            poses = poses + smoves - smoves.detach()
        
        poses = poses.unsqueeze(-1) 
        floors = torch.floor(poses) #obtain the floor value of p_n which is $\ceil{p_n}$
        ceils = floors + 1  
        
        # the weight has been separated into two parts 
        w1 = (func(poses - floors)) # where pi - j $\leq$ 1
        w2 = (func(ceils - poses))  # where j - pi $\leq$ 1
         
        # get the fixed output size and output the maximum output length and padded with 0
        out = torch.zeros((features.shape[0], int(ceils.max().item())+1, features.shape[2])).to(features.device)
        
        # obtain the output vector fiwi*w1 + fiwi*w2
        out.scatter_add_(1, floors.to(torch.long).repeat_interleave(features.shape[2], dim=2), w1*fw)
        out.scatter_add_(1, ceils.to(torch.long).repeat_interleave(features.shape[2], dim=2), w2*fw)
        
        return out, ceils.squeeze(-1).max(dim=1)[0].int() + 1
    
    def row_pool(self, features, moves, weights, func):
        """
        This function is used for inference, where features, moves, weights are trained
        Args:
            features (_type_): 
            moves (_type_): 
            weights (_type_): 

        Returns:
            out: The output of pool
        """
        fw = features * weights.to(features.dtype).unsqueeze(1) # because of the type may dyamically change 
        poses = torch.cumsum(moves.detach(), 0)
       
        poses = poses.unsqueeze(1)
        
        floors = torch.floor(poses)
        ceils = floors + 1
        
        w1 = (func(poses - floors)).to(features.dtype) # >= -1
        w2 = (func(ceils - poses)).to(features.dtype) # <= 1

        out = torch.zeros((int(ceils[-1].item())+1, features.shape[1]), device=features.device, dtype=features.dtype)
       
        out.index_add_(0, floors.to(torch.long).squeeze(1), w1*fw)
        out.index_add_(0, ceils.to(torch.long).squeeze(1), w2*fw)
            
        return out
    
    def forward(self, x):
        """
        This function is used for training,

        Args:
            x (_type_): _description_
        Output:
            x_evs : expected output result
            lens  : the index of dynamic pooling
            bmoves: length vector
        """
        _x = x # 1,1,4096
        _x = self.conv1(_x) # 1, 512,4096
        _x = self.batch_norm1(_x)  # fit into the batchnorm layer
        # go through the activation layer with <activation> and <dropout>
        _x = self.activation(_x)
        features = _x
        
        if x.shape[1] != 1:
            __x =  self.conv1x1(x)
            # input two features x, x^2 as input for predictor network
            jumps_mat = self.predictor(torch.cat([__x, __x*__x], dim=1))
        else:
            jumps_mat = self.predictor(torch.cat([x, x*x], dim=1))
        # force weights and moves ranging from 0 to 1.
        weights = torch.sigmoid(jumps_mat[:,0,:])
        moves = torch.sigmoid(jumps_mat[:,1,:])
        if self.training:
            renorm = (1 / self.norm_target / moves.mean().detach()).detach()
            self.norm_mean.copy_(0.99 * self.norm_mean + 0.01 * renorm)
        else:
            renorm = self.norm_mean

        moves = moves * renorm
        # calculate the M value which the average of mean, and for the range (0,1)
        row_renorm = torch.max(moves.mean(dim=1), torch.tensor([1.0]).to(moves.device, moves.dtype))
        moves = moves / row_renorm.unsqueeze(1) # renormalise the length vector
        
        features = features.permute((0,2,1)).contiguous()
        x_evs, lens = self.big_pool(features, moves, weights, self.line)
        
        x_evs = x_evs.permute(0, 2, 1).contiguous()
        x_evs = F.pad(x_evs, (0, 15 - (x_evs.shape[2] % 15)))
        # x_evs = F.pad(x_evs, (0, 3 - (x_evs.shape[2] % 3)))
        return x_evs
##############################################################################################################################################################################################################################


def activation_function(activation):
    if activation == "mish":
        return Mish
    elif activation == "swish":
        return Swish
    elif activation == "relu":
        return torch.nn.ReLU
    elif activation == "gelu":
        return torch.nn.GELU
    elif activation == "glu":
        return torch.nn.GLU
    else:
        print("Unknown activation type:", activation)
        sys.exit(1)


class network(nn.Module):
    def __init__(self, config=None, arch=None, seqlen=4096, debug=False, dpblock_flag=True, replace=True):
        """
        Args:
            config (_type_, optional): _description_. Defaults to None.
            arch (_type_, optional): _description_. Defaults to None.
            seqlen (int, optional): _description_. Defaults to 4096.
            debug (bool, optional): _description_. Defaults to False.
            dpblock_flag (bool, optional): _description_. Defaults to True.
                IF the dpblock_flag sets to True:
                THEN we create the dynamic pooling layer and add to (or replace with) the specific <replace> layer.
                The recommand setting of dpblock index should be 0 or 1.
            dpblock_idx (int, optional): _description_. Defaults to 0.
            replace (bool, optional): _description_. Defaults to False.
        """
        # sets the dp_layer settings
        super().__init__()
#         self.dp_layer = [4, 768, -1, 9, 1, 0, 1, 0, 1, 3, 0.05, 32]
#         self.layer_dict = {0: 256, 1: 256, 2: 256, 3: 320, 4: 384, 5: 448, 6: 512, 7: 512, 8: 512, 9: 512, 10: 512, 11: 512, 12: 768, 13: 768, 14: 768, 15: 768, 16: 768, 17: 768, 
# 18: 768, 19: 768, 20: 768, 21: 768}
        if debug:
            print("Initializing network")
        self.seqlen = seqlen
        self.vocab = config.vocab
        # the data distribution may change after neural network training thus, after each network we feed each neural network with batchnorm layer.
        self.bn = nn.BatchNorm1d
        if arch is None:
            arch = rna_default  # the original RODAN is arch == None, we change to arch is None

        activation = activation_function(config.activation.lower())
        sqex_activation = activation_function(config.sqex_activation.lower())

        self.convlayers = nn.Sequential()
        in_channels = 1
        convsize = self.seqlen
        ############### Add dynamic pooling flags #######################################################
        # if dpblock_flag:  # If we apply dynamic pooling on the model
        #     dpblock_idx = config.layer
        #     try:
        #         if replace:
        #             # change the dp_layers value
        #             print(dpblock_idx)
        #             if isinstance(dpblock_idx, list): # try multiple layers
        #                 for idx in dpblock_idx:
        #                     self.dp_layer[1] = self.layer_dict[idx]
        #                     arch[idx] = self.dp_layer
        #             elif isinstance(dpblock_idx, int): #only replace with one layer
        #                 self.dp_layer[1] = self.layer_dict[dpblock_idx]
        #                 arch[dpblock_idx] = self.dp_layer
        #         else:
        #             arch.insert(dpblock_idx, self.dp_layer)
        #     except Exception:
        #         print(f"The index you type {dpblock_idx} is out of bound")
        ##################################################################################################
        for i, layer in enumerate(arch):
            # if len(layer) > 7:
            #     paddingarg = layer[0]
            #     out_channels =  layer[1]
            #     seperable = layer[2]
            #     kernel = layer[3]
            #     stride = layer[4]
            #     sqex = layer[5]
            #     dodropout = layer[6]
            #     bias = layer[7]
            #     dilation = layer[8]
            #     norm = layer[9]
            #     dropout = layer[10]
            #     prediction_size = layer[11]
            # else:
            paddingarg = layer[0]
            out_channels = layer[1]
            seperable = layer[2]
            kernel = layer[3]
            stride = layer[4]
            sqex = layer[5]
            dodropout = layer[6]
            expansion = True

            if dodropout:
                dropout = config.dropout
            else:
                dropout = 0
            if sqex:
                squeeze = config.sqex_reduction
            else:
                squeeze = 0

            if paddingarg == -1:
                padding = kernel // 2
            else:
                padding = paddingarg
            if i == 0:
                expansion = False

            convsize = (convsize + (padding*2) - (kernel-stride))//stride
            if debug:
                print("padding:", padding, "seperable:", seperable, "ch", out_channels, "k:",
                      kernel, "s:", stride, "sqex:", sqex, "drop:", dropout, "expansion:", expansion)
                print("convsize:", convsize)

            ###################### Dynamic Pooling  ###################################################################
            # add module add_module(name, convblock module)
            if len(layer) > 7:  # if the layer is dynamic pooling
                # activation = activation_function(config.dp_activation)
                # if dodropout:
                #     dropout = config.dp_dropout
                # else:
                #     dropout = 0
                # self.convlayers.add_module("conv"+str(i), dpool(
                #         in_channels,
                #         out_channels,
                #         kernel,
                #         stride,
                #         padding,
                #         dilation,
                #         bias,
                #         norm,
                #         dropout,
                #         activation,
                #         prediction_size))
                pass
            #########################################################################################################################################################################################################################################################################################
            else:
                activation = activation_function(config.activation)
                self.convlayers.add_module("conv"+str(i), convblock(in_channels, out_channels, kernel, stride=stride, padding=padding, seperable=seperable,
                                           activation=activation, expansion=expansion, dropout=dropout, squeeze=squeeze, sqex_activation=sqex_activation, residual=True))
            in_channels = out_channels
            self.final_size = out_channels

        self.final = nn.Linear(self.final_size, len(self.vocab))
        if debug:
            print("Finished init network")

    def forward(self, x):
        """
        Obtain the final results
        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # s = self.dynpool(x)
        x = self.convlayers(x)
        x = x.permute(0,2,1)
        x = self.final(x)
        x = torch.nn.functional.log_softmax(x, -1)
        return x.permute(1, 0, 2).contiguous()
    

counter = 0

def get_checkpoint(epoch, model, optimizer, scheduler):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    return checkpoint


def get_config(model, config):
    config = {
        "state_dict": model.state_dict(),
        "config": config
    }
    return config


def train(config=None, args=None, arch=None):
    graph = False
    modelfile = args.model
    trainloss = []
    validloss = []
    learningrate = []
    avg_lenses = []
    avg_moves = []
    torch.backends.cudnn.benchmark = True  # increase the speed of neural network
    if args.verbose:
        print("Using device:", device)
        print("Using training file:", config.trainfile)

    model = network(config=config, arch=arch, seqlen=config.seqlen).to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    if args.verbose:
        print(f"The total parameters: {params}") # print out the trainable parameters
    
    if modelfile != None:
        print("Loading pretrained model:", modelfile)
        model.load_state_dict(torch.load(modelfile))

    if args.verbose:
        print("Optimizer:", config.optimizer, "lr:",
              config.lr, "weightdecay", config.weightdecay)
        print("Scheduler:", config.scheduler, "patience:", config.scheduler_patience, "factor:", config.scheduler_factor,
              "threshold", config.scheduler_threshold, "minlr:", config.scheduler_minlr, "reduce:", config.scheduler_reduce)

    if config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weightdecay)
        
    elif config.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        
    elif config.optimizer.lower() == "ranger":
        from pytorch_ranger import Ranger
        optimizer = Ranger(model.parameters(), lr=config.lr, alpha = 0.5, weight_decay=config.weightdecay)
    if args.verbose:
        print(model)

    model.eval()
    with torch.no_grad():
        fakedata = torch.rand((1, 1, config.seqlen))
        fakeout = model.forward(fakedata.to(device))
        elen = fakeout.shape[0]

    data = dataloader(recfile=config.trainfile,
                      seq_len=config.seqlen, elen=elen)
    data_loader = DataLoader(dataset=data, batch_size=config.batchsize,
                             shuffle=True, num_workers=args.workers, pin_memory=True)

    if config.scheduler == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.scheduler_patience, factor=config.scheduler_factor, verbose=args.verbose, threshold=config.scheduler_threshold, min_lr=config.scheduler_minlr)

    count = 0
    last = None

    if args.statedict:
        checkpoint = torch.load(args.statedict)
        #checkpoint_optimizer = torch.load('runs-ext.torch')
        model.load_state_dict(checkpoint["state_dict"])
        #optimizer.load_state_dict(checkpoint_optimizer["optimizer"])
#         optimizer.param_groups[0]['lr'] = 0.002
        #scheduler.load_state_dict(checkpoint_optimizer["scheduler"])

    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)

    shutil.rmtree(args.savedir+"/"+config.name, True)

    global iterations

    for epoch in range(config.epochs):
        model.train()
        totalloss = 0
        loopcount = 0
        learningrate.append(optimizer.param_groups[0]['lr'])
        if args.verbose:
            print("Learning rate:", learningrate[-1])

    ##############################################################################
    # Compared to Dynamic pooling structure, they are using (x, y, inds, lens),
    # i.e. event, label, event_len, label_len
    ##############################################################################
        for i, (event, event_len, label, label_len) in enumerate(data_loader):
            # event =
            event = torch.unsqueeze(event, 1)
            if event.shape[0] < config.batchsize:
                continue

            label = label[:, :max(label_len)]
            event = event.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            event_len = event_len.to(device, non_blocking=True)
            label_len = label_len.to(device, non_blocking=True)

            optimizer.zero_grad()

            # TODO: WE NEED to focus on this part more times... 
            out = model.forward(event)
            losses = ont.ctc_label_smoothing_loss(out, label, label_len, ls_weights)
            loss = losses["loss"]
            loss.backward()
            # if loss.item() > 0.004:
                # print(loss.item())
            totalloss += loss.cpu().detach().numpy()
            if count % 1000 == 1:
                print("Loss", loss.data, "epoch:", epoch,
                    count, optimizer.param_groups[0]['lr'])

            if config.gradclip:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradclip)

            optimizer.step()
            loopcount += 1
            count += 1
            iterations += 1
            if loopcount >= config.train_loopcount:
                break
        if args.verbose:
            print("Train epoch loss", totalloss/loopcount)

        vl = validate(model, device, config=config,
                      args=args, epoch=epoch, elen=elen)

        if config.scheduler == "reducelronplateau":
            scheduler.step(vl)
        elif config.scheduler == "decay":
            if (epoch > 0) and (epoch % config.scheduler_reduce == 0):
                optimizer.param_groups[0]['lr'] *= config.scheduler_factor
                if optimizer.param_groups[0]['lr'] < config.scheduler_minlr:
                    optimizer.param_groups[0]['lr'] = config.scheduler_minlr

        trainloss.append(float(totalloss/loopcount))
        validloss.append(vl)

        torch.save(get_config(model, config.orig), args.savedir +
                   "/"+config.name+"-epoch"+str(epoch)+".torch")
        torch.save(get_checkpoint(epoch, model, optimizer, scheduler),
                   args.savedir+"/"+config.name+"-ext.torch")
        
        if args.verbose:
            print("Train losses:", trainloss)
            print("Valid losses:", validloss)
            print("Learning rate:", learningrate)

    return trainloss, validloss


def validate(model, device, config=None, args=None, epoch=-1, elen=34):
    if config.valid_loopcount < 1:
        return(np.float(0))
    modelfile = None
    if args != None:
        modelfile = args.model

    # NOTE: possibly move these into train
    valid_data = dataloader(recfile=config.validfile,
                            seq_len=config.seqlen, elen=elen)
    valid_loader = DataLoader(dataset=valid_data, batch_size=config.batchsize,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    total = 0
    totalloss = 0

    if model is None and modelfile:
        model = network(config=config).to(device)
        model.load_state_dict(torch.load(modelfile))

    model.eval()

    with torch.no_grad():
        for i, values in enumerate(valid_loader):
            event = values[0]
            event_len = values[1]
            label = values[2]
            label_len = values[3]
            event = torch.unsqueeze(event, 1)
            if event.shape[0] < config.batchsize:
                continue

            label = label[:, :max(label_len)]
            event = event.to(device)
            event_len = event_len.to(device)
            label = label.to(device)
            label_len = label_len.to(device)
            out = model.forward(event)
            losses = ont.ctc_label_smoothing_loss(out, label, label_len, ls_weights)
            loss = losses["loss"]
            totalloss += loss.cpu().detach().numpy()
            total += 1
            if total >= config.valid_loopcount:
                break
    if args.verbose:
        print("Validation loss:", totalloss / total)

    return np.float(totalloss / total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data.')
    parser.add_argument("-c", "--config", default=None, type=str)
    parser.add_argument("-f", "--func", default=None, type=str)
    parser.add_argument("-d", "--statedict", default=None, type=str)
    parser.add_argument("-a", "--arch", default=None,
                        type=str, help="Architecture file")
    parser.add_argument("-D", "--savedir", default="runs",
                        type=str, help="save directory (default: runs)")
    parser.add_argument("-n", "--name", default=None, type=str)
    parser.add_argument("-m", "--model", default=None, type=str)
    parser.add_argument("-l", "--labelsmoothing",
                        default=False, action="store_true")
    parser.add_argument("-w", "--workers", default=8,
                        type=int, help="num_workers (default: 8)")
    parser.add_argument("-v", "--verbose", default=True,
                        action="store_false", help="Turn verbose mode off")
    parser.add_argument("--rna", default=False,
                        action="store_true", help="Use default RNA model")
    parser.add_argument("--dna", default=False,
                        action="store_true", help="Use default DNA model")
    #parser.add_argument("-P", "--position", default=3, action = "store_true", help = "Set the position for Dynamic Pooling")
    args = parser.parse_args()
    continue_training = False
    if continue_training:
        args.statedict = 'runs-epoch29.torch'
    if args.name == None:
        args.name = "training"
        while args.name == "":
            args.name = input("Number: ")
    defaultconfig["name"] = args.name

    if args.config != None:
        import yaml
        import re

        # https://stackoverflow.com/questions/52412297/how-to-replace-environment-variable-value-in-yaml-file-to-be-parsed-using-python
        def path_constructor(loader, node):
            # print(node.value)
            return os.path.expandvars(node.value)

        class EnvVarLoader(yaml.SafeLoader):
            pass

        path_matcher = re.compile(r'.*\$\{([^}^{]+)\}.*')
        EnvVarLoader.add_implicit_resolver('!path', path_matcher, None)
        EnvVarLoader.add_constructor('!path', path_constructor)

        newconfig = yaml.load(open(args.config), Loader=EnvVarLoader)
        defaultconfig.update(newconfig)

    if args.arch != None:
        defaultconfig["arch"] = open(args.arch, "r").read()

    if args.arch != None:
        print("Loading architecture from:", args.arch)
        args.arch = eval(open(args.arch, "r").read())

    if args.rna:
        args.arch = rna_default

    if args.dna:
        args.arch = dna_default

    config = objectview(defaultconfig)
    # from Bonito but weighting for blank changed to 0.1 from 0.4
    # if args.labelsmoothing:
    # if using args.labelsmooething
    C = len(config.vocab)
    ls_weights = torch.cat([torch.tensor([0.1]), (0.1 / (C - 1)) * torch.ones(C - 1)]).to(device)
    # train the neural network with configs
    train(config=config, args=args, arch=args.arch)
