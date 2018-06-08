# -*- coding: utf-8 -*-
"""
Created on Thu Jun 8 09:43:13 2018

@author: anand mooga
"""

import torch 
import torch.nn
import numpy as np 
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader 
import torch.nn.functional as F

from utils import reduce_mask, sparse_gather, sparse_scatter

'''
Test the reduce_mask function
reduce_mask(mask, msize, bsize, bcount, boffset, bstride, thresh = 0.5)
'''

N = 10
C= 3
H = W = 64
h = w = 4

mask = np.random.randint(2, size=(N,H,W))
mask = torch.from_numpy(mask).float()
msize = list(mask.size())
#print(mask, msize)
bsize = [h,w]
boffset = [0,0]
bstride = [h,w]
bcount = [int(H/h), int(W/w)]

activeBlockIndices = reduce_mask(mask, msize, bsize, bcount, boffset, bstride)
#print(indices)


'''
Test the sparse_gather function
sparse_gather(x, xsize, activeBlockIndices, bsize, bcount, boffset, bstride)

'''

x = np.random.rand(N,C,H,W)
#x = np.ones((N,C,H,W))
x = torch.from_numpy(x).float()
xsize = list(x.size())
#print(x, xsize)
gathered = sparse_gather(x, xsize, activeBlockIndices, bsize, bcount, boffset, bstride)
gathered = 99*gathered
#print(gathered)
#print(gathered.size())

'''
Test the sparse_scatter function
sparse_scatter(x, xsize, gathered, gsize, activeBlockIndices, bsize, bcount, boffset, bstride, do_add =1)

'''
gsize = list(gathered.size())

sparse_scatter(x, xsize, gathered, gsize, activeBlockIndices, bsize, bcount, boffset, bstride, do_add =1)
#print(x)