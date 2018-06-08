# -*- coding: utf-8 -*-
"""
Created on Thu Jun 8 10:55:42 2018

@author: anand mooga
"""
import torch 
import torchvision
import torch.nn
import numpy as np 
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader 
import torch.nn.functional as F
from utils import reduce_mask, sparse_gather, sparse_scatter

import keras
from keras.datasets import mnist

'''
Using MNIST dataset as the input, and the input image+thresolding as the mask.
SBNet will be tested 

Understanding: SBNet is good for segmentaion, classification not so much :/ . To be verified!

'''
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train /= 255
X_test /= 255

#hyperparameters
num_epochs = 10
num_classes = 10
batch_size = 32
learning_rate = 0.01

N = batch_size
H = W = 28
C = 1
h = w = 2

bsize = [h,w]
boffset = [0,0]
bstride = [h,w]
bcount = [int(H/h), int(W/w)]

'''
mask = [N,H,W]
x = [N, C, H, W]
activeBlockIndices = [B, 3]
'''

class sparse_conv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(sparse_conv, self).__init__()
		
		self.channel =  nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 1, padding = 0),
			nn.ReLU(),
            nn.BatchNorm2d(out_ch))

		self.dense = nn.Sequential(
            nn.Conv2d(out_ch, 2*out_ch, 1, padding = 0),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(2*out_ch, 2*out_ch, 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
        	nn.Conv2d(2*out_ch, out_ch, 1, padding = 0),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch)
            )
	def forward(self, x, activeBlockIndices, bsize, bcount, boffset, bstride):
		xsize = list(x.size())
		x = self.channel(x)
		gathered = sparse_gather(x, xsize, activeBlockIndices, bsize, bcount, boffset, bstride)
		gathered =  self.dense(gathered)
		gsize = list(gathered.size())
		sparse_scatter(x, xsize, gathered, gsize, activeBlockIndices, bsize, bcount, boffset, bstride, do_add = 1)
		return x










