# -*- coding: utf-8 -*-
"""
Created on Thu Jun 8 10:55:42 2018

@author: anand mooga
"""
import torch 
import torchvision
import torch.nn as nn
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

X_train = X_train/255
X_test = X_test/255

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
	def __init__(self, in_ch, out_ch, bsize, bcount, boffset, bstride):
		super(sparse_conv, self).__init__()
		self.bsize = bsize
		self.bcount = bcount
		self.boffset = boffset
		self.bstride = bstride
	
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
            nn.BatchNorm2d(out_ch), #can be made as deep as required!
        	nn.Conv2d(2*out_ch, out_ch, 1, padding = 0),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch)
            )
	def forward(self, x, activeBlockIndices):
		xsize = list(x.size())
		x = self.channel(x)
		gathered = sparse_gather(x, xsize, activeBlockIndices, self.bsize, self.bcount, self.boffset, self.bstride)
		gathered =  self.dense(gathered)
		gsize = list(gathered.size())
		sparse_scatter(x, xsize, gathered, gsize, activeBlockIndices, self.bsize, self.bcount, self.boffset, self.bstride)
		return x


class FastNN(nn.Module):
	def __init__(self, N, C, W, H, w, h):
		super(FastNN, self).__init__()
		
		self.bsize = [h,w]
		self.boffset = [0,0]
		self.bstride = [h,w]

		self.bcount1 = [int(H/h), int(W/w)]
		self.srb1 = sparse_conv(C, 16, self.bsize, self.bcount1, self.boffset, self.bstride)
		self.mp1 = nn.MaxPool2d(2)

		self.bcount2 = [int(H/2*h), int(W/2*w)]
		self.srb2 = sparse_conv(16, 8, self.bsize, self.bcount2, self.boffset, self.bstride)
		self.mp2 = nn.MaxPool2d(2)

		self.dense1 = nn.Linear(7*7*8, 10)
		self.act1 = nn.Softmax()

	def forward(self, x, mask):
		
		mask1 = mask
		msize1 = list(mask1.size())
		activeBlockIndices1 = reduce_mask(mask1, msize1, self.bsize, self.bcount1, self.boffset, self.bstride)
		x = self.srb1(x, activeBlockIndices1)
		x = self.mp1(x)

		mask2 = self.mp1(mask1)
		msize2 = list(msize2.size())
		activeBlockIndices2 = reduce_mask(mask2, msize2, self.bsize, self.bcount2, self.boffset, self.bstride)
		x = self.srb2(x, activeBlockIndices2)
		x = self.mp2(x)

		x = x.reshape(-1)
		x = self.dense1(x)
		x = self.act1(x)
		return x





















