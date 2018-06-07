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


def reduce_mask(mask, msize, bsize,	bcount, boffset, bstride, thresh = 0.5):
    '''
	Reduce mask operation:
	Takes a binary mask as input and after performing avg pooling selects active blocks
	and return indices of the active blocks

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

	Ideas to improve this:
		use Fig 4 logic, this will greatly speed it up : 

	'''
    assert len(mask.get_shape()) == 3, 'Expect mask rank = 3'
    assert type(bsize) in [list, tuple], '`bsize` needs to be a list or tuple.'
    assert type(bcount) in [list, tuple], '`bcount` needs to be a list or tuple.'
    assert type(boffset) in [list, tuple], '`boffset` needs to be a list or tuple.'
    assert type(bstride) in [list, tuple], '`bsize` needs to be a list or tuple.'

    assert bsize[0] == bsize[1], 'Expect block to be a square, bsize[0] == bsize[1]' 
    
    assert msize[1]%bcount[0] == 0 and msize[2]%bcount[1] == 0, 'Mask cannot be partioned into given grid shape'

    activeBlockIndices = []
    count_index
	for N in range(msize[0]):  #loop through batches 
		for h0 in range(bcount[0]): #loop through grid 
			for w0 in range(bcount[1]):
				bh_index = boffset[0] + h0*bstride[0]
				bw_index = boffset[1] + w0*bstride[1]
				active = False  #flag to indicate weather a box is active
				block = mask[N, bh_index:min(bh_index+bsize[0], msize[1]), bw_index:min(bw_index+bsize[1], msize[2])]
				val = block.mean()
				if val > thresh:
					active = True
				if active:
					activeBlockIndices[count_index] = [N, bh_index, bw_index]
					count_index += 1
	return activeBlockIndices



def sparse_gather(x, xsize, activeBlockIndices, bsize, bcount, boffset, bstride):
	'''
	Sparse gather operation:

	Takes a torch tensor as input and gathers blocks on basis of indices generated 
	from reduce_mask. Essentialy a slicing and concatenation along the batch dimension

	Params:
		x 							# torch tensor [N, C, H, W] 
		xsize						# list [N,C,H,W] giving the shapes 
		activeBlockIndices			# list [B, [N, y, x]] B,N,y,x has ususal defn, check reduce_mask
		bsize						# list [block height, block width]
		bcount						# list [grid height, grid width]
		boffset 					# list [offset height, offset width]
		bstride						# list [block stride h, block stride w]	
		thresh 						# threshhold for being active
		
	https://github.com/pytorch/pytorch/issues/822

	'''
	compressed = torch.empty((activeBlockIndices.get_shape()[0], xsize, bsize[0], bsize[1]))
	for B, bh_index, bw_index in activeBlockIndices:
		block = x[N, bh_index:min(bh_index+bsize[0], xsize[2]), bw_index:min(bw_index+bsize[1], xsize[3])]
		compressed[B] = block
	return compressed




















	



















































