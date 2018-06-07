# -*- coding: utf-8 -*-
"""
Created on Thu Jun 7 13:57:13 2018

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


'''

Reduce mask operation:
Takes a binary mask as input and after performing avg pooling selects active blocks
and return indices of the active blocks

The indices coorespond to the centre of the 

Params:
	mask 						# torch tensor, mask is binary
	msize	 					# list [N,H,W] where N is batch diemension, W, H are width and height of mask
	bsize						# list [block height, block width]
	bcount						# list [grid height, grid width]
	boffset 					# list [offset height, offset width]
	bstride						# list [block stride h, block stride w]	
	thresh 						# threshhold for being active
	

Require_grad = Fasle

Output:
	activeBlockIndices of size [B, 3] where B is the number of active blocks 
	and 3 corresponds to [N, y, x] where N is batch size, and (y,x) is the 
	co-ordinates of the "centre" of the block.

Limitations and to be implemented:
	block height = block width
	msize should be divisibe by grid size, can be solved by padding
	manually input offsets and strides, write a func to calculate

'''




	



















































