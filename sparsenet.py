# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:12:57 2018

@author: anand mooga
"""


import torch 
import torch.nn as nn
import numpy as np 
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader 
import torch.nn.functional as F


'''
Version 2
torch version: 0.4.0


Rough Usage:
x = tensor of [N,C,H,W]
mask = tensor of [N,H,W]
ksize = [h,w]
kstride = [h-1, w-1]

mask = pad_input(mask, ksize, kstride)
mask.require_grad = False
x = pad_input(x, ksize, kstride)
x.require_grad = True

indices = reduce_mask_pool2d(mask, ksize, kstride, thresh = 0.2, avg= True)
gathered = gather2d(x, indices, ksize, kstride)

## do conv operations on gathred

scatter2d(x, gathered, indices, ksize, kstride, add = True)

## do conv operations on x 

'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def reduce_mask_pool2d(mask, ksize, kstride, thresh = 0.2, avg= True):
	'''
	Reduce mask operation:Takes a binary mask as input and after performing 
	avg pooling or max pooling selects active blocksand return indices of 
	the active blocks
	
	Inputs:
	:param mask:		torch tensor	#[N,H,W] binary mask, where N is batch diemension, W, H are width and height of mask
	:param kszie:		[list, tuple]	#[h,w] Size of kernel to perform pooling
	:param kstride:		[list, tuple] 	#[h_stride,w_stride] Stride of Kernel
	:param thresh:		int 			#applicable for avg pooling
	:param avg:			bool 			#Avg pooling or max pooling 

	Require_grad = Fasle

	Output:
		indicies of size [B, 3] where B is the number of active blocks 
		and 3 corresponds to [N, y, x] where N is batch size, and (y,x) is the 
		co-ordinates of the "centre" of the block.
		indicies is a torch tensor
	Note:
		padding is done automatically
	'''

	assert torch.is_tensor(mask) == True, 'Expect mask to be a pytorch tensor'
	isize = list(mask.size())
	assert len(mask.size()) == 3 , 'Expect input rank = 3'
	assert type(ksize) in [list, tuple], 'Expect `ksize` to be list or tuple'
	assert type(kstride) in [list, tuple], 'Expect `kstride` to be list or tuple'
	assert len(kstride) == 2 and len(ksize) == 2, 'Expect length of kstride and ksize to be 2'
	assert type(thresh) in [int, float], 'Expect `thresh` to be int or float'

	mask = mask.unsqueeze(1)

	if avg:
		temp = F.avg_pool2d(input = mask, kernel_size = ksize, stride = kstride, padding = 0).to(device).squeeze()
		
		indicesm = torch.where(temp > thresh, torch.ones_like(temp).to(device), torch.zeros_like(temp).to(device)).int()
		indices = (indicesm != 0).nonzero().to(device)
		return indices
	else:
		temp = F.max_pool2d(input = mask, kernel_size = ksize, stride = kstride, padding = 0).to(device).squeeze()
		indicesm = torch.where(temp > thresh, torch.ones_like(temp).to(device), torch.zeros_like(temp).to(device)).int()
		indices = (indicesm != 0).nonzero().to(device)
		return indices


def pad_input(input, ksize, kstride):
	'''
	Pads the input or mask according to the required kernel to perform sparse convolution

	Inputs:
	:param input:		torch tensor	#[N,H,W] binary mask or [N,C,H,W] input, where N is batch diemension, W, H are width and height of mask and C is channels
	:param kszie:		[list, tuple]	#[h,w] Size of kernel to perform pooling
	:param kstride:		[list, tuple] 	#[h_stride,w_stride] Stride of Kernel

	Output: Zero padded torch tensor
	'''
	assert torch.is_tensor(input) == True, 'Expect input to be a pytorch tensor'
	isize = list(input.size())
	assert len(input.size()) == 3 or len(input.size()) == 4, 'Expect input rank = 3(mask) or 4(input)'
	assert type(ksize) in [list, tuple], 'Expect `ksize` to be list or tuple'
	assert type(kstride) in [list, tuple], 'Expect `kstride` to be list or tuple'
	assert len(kstride) == 2 and len(ksize) == 2, 'Expect length of kstride and ksize to be 2'

	#padding along width!
	pad_w = kstride[-1] - ((isize[-1]-ksize[-1])%kstride[-1])
	pad_w1 = pad_w2 = pad_w//2
	if pad_w%2 == 1:
		pad_w2 += 1

	#padding along height
	pad_h = kstride[-2] - ((isize[-2]-ksize[-2])%kstride[-2])
	pad_h1 = pad_h2 = pad_h//2
	if pad_h%2 == 1:
		pad_h2 += 1
	
	pad = (pad_w1, pad_w2, pad_h1, pad_h2)
	return F.pad(input, pad, "constant", 0).to(device)

def gather2d(input, indices, ksize, kstride):
	'''
	Sparse gather operation:

	Takes a torch tensor as input and gathers blocks on basis of indices generated 
	from reduce_mask_pool2d. Essentialy a slicing and concatenation along the batch dimension

	Params:
	:param input:		torch tensor	#[N,C,H,W] size tensor, where N is batch diemension, W, H are width and height of mask 
	:param indices:		torch tensor 	#[B,3] where B is the number of active blocks and 3 corresponds to [N,y,x]
	:param kszie:		[list, tuple]	#[h,w] Size of kernel to perform pooling
	:param kstride:		[list, tuple] 	#[h_stride,w_stride] Stride of Kernel

	Output:
		Returns a [B,C,h,w] shape tensor which is the active blocks stacked in the 
		batch diemension
	gradient flow is fine: https://github.com/pytorch/pytorch/issues/822
	
	Problems? : How to remove the for loop and vectorize it?
	idea: falatten and multiply and reshape !
	'''
	assert torch.is_tensor(input) == True, 'Expect input to be a pytorch tensor'
	isize = list(input.size())
	assert len(input.size()) == 4, 'Expect input rank = 4 , [N,C,H,W]'

	assert torch.is_tensor(indices) == True, 'Expect indicies to be a pytorch tensor'
	asize = list(indices.size()) 

	assert type(ksize) in [list, tuple], 'Expect `ksize` to be list or tuple'
	assert type(kstride) in [list, tuple], 'Expect `kstride` to be list or tuple'
	assert len(kstride) == 2 and len(ksize) == 2, 'Expect length of kstride and ksize to be 2'

	#gathered = torch.empty((asize[0], isize[1], ksize[0], ksize[1])).to(device)
	gathered = input[indices[0][0]:indices[0][0]+1, :, indices[0][1]*kstride[0]:indices[0][1]*kstride[0]+ksize[0], indices[0][2]*kstride[1]: indices[0][2]*kstride[1]+ksize[1]]
	for B, h0, w0 in indices[1:]:
		gathered = torch.cat((gathered, input[B:B+1, :, h0*kstride[0]:h0*kstride[0]+ksize[0], w0*kstride[1]: w0*kstride[1]+ksize[1]]), 0)
	return gathered

def scatter2d(input, gathered, indices, ksize, kstride, add = True):
	'''
	Sparse scatter operation:

	Takes a gathered torch tensor as input and scatters it back on the input on basis of
	indices generated from reduce_mask. Essentialy a slicing and addition/write operation

	Params:
		:param input:		torch tensor	#[N,C,H,W] size tensor, where N is batch diemension, W, H are width and height of mask 
		:param gathered:	torch tensor 	#[B,C,h1,w1] B,C are same as the input, h1 and w1 are deetermined on the type of convolutions used
		:param indices:		torch tensor 	#[B,3] where B is the number of active blocks and 3 corresponds to [N,y,x]
		:param kszie:		[list, tuple]	#[h,w] Size of kernel to perform pooling
		:param kstride:		[list, tuple] 	#[h_stride,w_stride] Stride of Kernel
		:param add:			bool 			#Decides weather to add the values or replace them while scattering
	Output:
		A tensor of same shape as input, but it has been updatd with the scattered values
	'''
	assert torch.is_tensor(input) == True, 'Expect input to be a pytorch tensor'
	assert len(input.size()) == 4, 'Expect input rank = 4 , [N,C,H,W]'

	assert torch.is_tensor(indices) == True, 'Expect indicies to be a pytorch tensor'

	assert torch.is_tensor(gathered) == True, 'Expect gathered to be a pytorch tensor'
	gsize = list(gathered.size())

	assert type(ksize) in [list, tuple], 'Expect `ksize` to be list or tuple'
	assert type(kstride) in [list, tuple], 'Expect `kstride` to be list or tuple'
	assert len(kstride) == 2 and len(ksize) == 2, 'Expect length of kstride and ksize to be 2'

	count_index = 0
	for B, h0, w0 in indices:
		if add:
			input[B, :, h0*kstride[0]:h0*kstride[0]+gsize[2], w0*kstride[1]: w0*kstride[1]+gsize[3]] += gathered[count_index]
		else:
			input[B, :, h0*kstride[0]:h0*kstride[0]+gsize[2], w0*kstride[1]: w0*kstride[1]+gsize[3]] = gathered[count_index]
		count_index += 1



def mask_pool2d(mask):
	'''
	mask downsample operation:Takes a binary mask as input and performs max
	pooling to reduce its size 
	
	Inputs:
	:param mask:		torch tensor	#[N,H,W] binary mask, where N is batch diemension, W, H are width and height of mask
	
	Outputs: A binary torch tensor of half the input mask size
	'''
	assert torch.is_tensor(mask) == True, 'Expect mask to be a pytorch tensor'
	isize = list(mask.size())
	assert len(mask.size()) == 3 , 'Expect input rank = 3'

	mask = mask.unsqueeze(1)
	temp = F.max_pool2d(input = mask, kernel_size = [2,2], stride = [2,2], padding = 0).to(device).squeeze().float()
	temp.require_grad = False
	return temp



class gather2dc(nn.Module):
	def __init__(self, ksize, kstride):
		super(gather2dc, self).__init__()
		self.ksize = ksize
		self.kstride = kstride

	def forward(self, input, indices):
		gathered = input[indices[0][0]:indices[0][0]+1, :, indices[0][1]*self.kstride[0]:indices[0][1]*self.kstride[0]+self.ksize[0], indices[0][2]*self.kstride[1]: indices[0][2]*self.kstride[1]+self.ksize[1]]
		for B, h0, w0 in indices[1:]:
			gathered = torch.cat((gathered, input[B:B+1, :, h0*self.kstride[0]:h0*self.kstride[0]+self.ksize[0], w0*self.kstride[1]: w0*self.kstride[1]+self.ksize[1]]), 0)
		return gathered


class scatter2dc(nn.Module):
	def __init__(self, kstride):
		super(scatter2dc, self).__init__()
		self.kstride = kstride

	def forward(self, input, gathered, indices):
		gsize = list(gathered.size())
		count_index = 0
		for B, h0, w0 in indices:
				input[B, :, h0*self.kstride[0]:h0*self.kstride[0]+gsize[2], w0*self.kstride[1]: w0*self.kstride[1]+gsize[3]] += gathered[count_index]
				count_index +=1
		return input




#from sparsenet import *
class sparse_block(nn.Module):
	def __init__(self, inp_ch, out_ch, ksize, kstride, block_layers = None, thresh = 0.2):
		super(sparse_block, self).__init__()

		assert type(block_layers) in [type(None), torch.nn.modules.container.Sequential], 'Expect block_layers as None or torch.nn.Sequential() object' 
		assert type(ksize) in [list, tuple], 'Expect `ksize` to be list or tuple'
		assert type(kstride) in [list, tuple], 'Expect `kstride` to be list or tuple'
		assert len(kstride) == 2 and len(ksize) == 2, 'Expect length of kstride and ksize to be 2'
		assert type(inp_ch) in [int, float], 'inp_ch should be int or float'
		assert type(out_ch) in [int, float], 'out_ch should be int or float'
		self.ksize = ksize
		self.kstride = kstride
		self.thresh = thresh
		self.g = gather2dc(ksize, kstride)
		self.c = scatter2dc(kstride)
		self.channel_control =  nn.Sequential(
			nn.Conv2d(inp_ch, out_ch, 1, padding = 0),
			nn.ReLU(),
            nn.BatchNorm2d(out_ch)).to(device)

		if block_layers == None:
			self.operation = nn.Sequential(
				nn.Conv2d(out_ch, 2*out_ch, 1, padding = 0),
				nn.ReLU(),
				nn.BatchNorm2d(2*out_ch),
				nn.Conv2d(2*out_ch, 2*out_ch, 3, padding = 1),
				nn.ReLU(),
				nn.BatchNorm2d(2*out_ch), #can be made as deep as required!
				nn.Conv2d(2*out_ch, out_ch, 1, padding = 0),
				nn.ReLU(),
				nn.BatchNorm2d(out_ch)
				).to(device)
		else:
			self.operation = block_layers.to(device)

	def forward(self, x, mask):
		x = self.channel_control(x)
		mask = pad_input(mask, self.ksize, self.kstride)
		mask.require_grad = False
		x = pad_input(x, self.ksize, self.kstride)
		x.require_grad = True

		indices = reduce_mask_pool2d(mask, self.ksize, self.kstride, self.thresh, avg= True)
		#gathered = gather2d(x, indices, self.ksize, self.kstride)
		gathered = self.g(x, indices)
		### Here the custom conv operations are done
		gathered = self.operation(gathered)
		x = self.c(x, gathered, indices)
		return x



############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################


def gather2dnew(input, indices, ksize, kstride):
    '''
    Sparse gather operation:

    Takes a torch tensor as input and gathers blocks on basis of indices generated 
    from reduce_mask_pool2d. Essentialy a slicing and concatenation along the batch dimension

    Params:
    :param input:		torch tensor	#[N,C,H,W] size tensor, where N is batch diemension, W, H are width and height of mask 
    :param indices:		torch tensor 	#[B,3] where B is the number of active blocks and 3 corresponds to [N,y,x]
    :param kszie:		[list, tuple]	#[h,w] Size of kernel to perform pooling
    :param kstride:		[list, tuple] 	#[h_stride,w_stride] Stride of Kernel

    Output:
        Returns a [B,C,h,w] shape tensor which is the active blocks stacked in the 
        batch diemension
    gradient flow is fine: https://github.com/pytorch/pytorch/issues/822

    Problems? : How to remove the for loop and vectorize it?
    idea: falatten and multiply and reshape !
    '''
    assert torch.is_tensor(input) == True, 'Expect input to be a pytorch tensor'
    isize = list(input.size())
    assert len(input.size()) == 4, 'Expect input rank = 4 , [N,C,H,W]'

    assert torch.is_tensor(indices) == True, 'Expect indicies to be a pytorch tensor'
    asize = list(indices.size()) 

    assert type(ksize) in [list, tuple], 'Expect `ksize` to be list or tuple'
    assert type(kstride) in [list, tuple], 'Expect `kstride` to be list or tuple'
    assert len(kstride) == 2 and len(ksize) == 2, 'Expect length of kstride and ksize to be 2'

    x = np.array(torch.chunk(input, chunks = isize[0], dim = 0), dtype = torch.Tensor)
    x = np.array([np.array(torch.chunk(i, chunks = isize[2]//ksize[0], dim = 2), dtype = torch.Tensor) for i in x], dtype= torch.Tensor)
    x = np.array([np.array([np.array(torch.chunk(j, chunks = isize[3]//ksize[1], dim =3), dtype= torch.Tensor) for j in i], dtype= torch.Tensor)for i in x], dtype= torch.Tensor)

    ind = indices.cpu().detach().numpy()
    temp = x[ind[:,0],ind[:,1],ind[:,2]]
    return torch.cat(tuple(test), dim = 0)

'''
          279 function calls in 0.014 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      113    0.006    0.000    0.006    0.000 {built-in method chunk}
        1    0.005    0.005    0.005    0.005 {built-in method cat}
      131    0.001    0.000    0.001    0.000 {built-in method numpy.core.multiarray.array}
       16    0.000    0.000    0.006    0.000 <ipython-input-192-a8d8e554d238>:35(<listcomp>)
        1    0.000    0.000    0.014    0.014 <string>:1(<module>)
        1    0.000    0.000    0.014    0.014 <ipython-input-192-a8d8e554d238>:1(gather2dnew)
        1    0.000    0.000    0.000    0.000 {method 'cpu' of 'torch._C._TensorBase' objects}
        1    0.000    0.000    0.001    0.001 <ipython-input-192-a8d8e554d238>:34(<listcomp>)
        1    0.000    0.000    0.014    0.014 {built-in method builtins.exec}
        3    0.000    0.000    0.000    0.000 {method 'size' of 'torch._C._TensorBase' objects}
        1    0.000    0.000    0.000    0.000 {method 'numpy' of 'torch._C._TensorBase' objects}
        2    0.000    0.000    0.000    0.000 __init__.py:113(is_tensor)
        1    0.000    0.000    0.000    0.000 {method 'detach' of 'torch._C._TensorBase' objects}
        2    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method builtins.len}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
'''






def scatter2dnew(input, gathered, indices, ksize, kstride, add = True):
    '''
    Sparse scatter operation:

    Takes a gathered torch tensor as input and scatters it back on the input on basis of
    indices generated from reduce_mask. Essentialy a slicing and addition/write operation

    Params:
        :param input:		torch tensor	#[N,C,H,W] size tensor, where N is batch diemension, W, H are width and height of mask 
        :param gathered:	torch tensor 	#[B,C,h1,w1] B,C are same as the input, h1 and w1 are deetermined on the type of convolutions used
        :param indices:		torch tensor 	#[B,3] where B is the number of active blocks and 3 corresponds to [N,y,x]
        :param kszie:		[list, tuple]	#[h,w] Size of kernel to perform pooling
        :param kstride:		[list, tuple] 	#[h_stride,w_stride] Stride of Kernel
        :param add:			bool 			#Decides weather to add the values or replace them while scattering
    Output:
        A tensor of same shape as input, but it has been updatd with the scattered values
    '''
    assert torch.is_tensor(input) == True, 'Expect input to be a pytorch tensor'
    assert len(input.size()) == 4, 'Expect input rank = 4 , [N,C,H,W]'
    isize = list(input.size())

    assert torch.is_tensor(indices) == True, 'Expect indicies to be a pytorch tensor'

    assert torch.is_tensor(gathered) == True, 'Expect gathered to be a pytorch tensor'
    gsize = list(gathered.size())

    assert type(ksize) in [list, tuple], 'Expect `ksize` to be list or tuple'
    assert type(kstride) in [list, tuple], 'Expect `kstride` to be list or tuple'
    assert len(kstride) == 2 and len(ksize) == 2, 'Expect length of kstride and ksize to be 2'
    
    x = np.array(torch.chunk(input, chunks = isize[0], dim = 0), dtype = torch.Tensor)
    x = np.array([np.array(torch.chunk(i, chunks = isize[2]//ksize[0], dim = 2), dtype = torch.Tensor) for i in x], dtype= torch.Tensor)
    x = np.array([np.array([np.array(torch.chunk(j, chunks = isize[3]//ksize[1], dim =3), dtype= torch.Tensor) for j in i], dtype= torch.Tensor)for i in x], dtype= torch.Tensor)
    
    g = np.array(torch.chunk(gathered, chunks = gsize[0], dim =0), dtype = torch.Tensor)
    ind = indices.cpu().detach().numpy()
    
    x[ind[:,0],ind[:,1],ind[:,2]] = g
    
    x = [[torch.cat(tuple(j) , dim = 3) for j in i] for i in x]
    x = [torch.cat(tuple(i), dim =2) for i in x]
    return torch.cat(tuple(x), dim =0)

'''
397 function calls in 0.029 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      113    0.020    0.000    0.020    0.000 {built-in method cat}
      114    0.008    0.000    0.008    0.000 {built-in method chunk}
      132    0.001    0.000    0.001    0.000 {built-in method numpy.core.multiarray.array}
        1    0.000    0.000    0.030    0.030 <ipython-input-215-4bd89e09b3ef>:1(scatter2dnew)
       16    0.000    0.000    0.005    0.000 <ipython-input-215-4bd89e09b3ef>:33(<listcomp>)
        1    0.000    0.000    0.000    0.000 {method 'cpu' of 'torch._C._TensorBase' objects}
        1    0.000    0.000    0.030    0.030 <string>:1(<module>)
        1    0.000    0.000    0.019    0.019 <ipython-input-215-4bd89e09b3ef>:40(<listcomp>)
        1    0.000    0.000    0.001    0.001 <ipython-input-215-4bd89e09b3ef>:32(<listcomp>)
        1    0.000    0.000    0.001    0.001 <ipython-input-215-4bd89e09b3ef>:41(<listcomp>)
        1    0.000    0.000    0.030    0.030 {built-in method builtins.exec}
        1    0.000    0.000    0.000    0.000 {method 'numpy' of 'torch._C._TensorBase' objects}
        3    0.000    0.000    0.000    0.000 {method 'size' of 'torch._C._TensorBase' objects}
        3    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 {method 'detach' of 'torch._C._TensorBase' objects}
        3    0.000    0.000    0.000    0.000 __init__.py:113(is_tensor)
        3    0.000    0.000    0.000    0.000 {built-in method builtins.len}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

'''


class gather2dcf(nn.Module):
	def __init__(self, ksize, kstride):
		super(gather2dc, self).__init__()
		self.ksize = ksize
		self.kstride = kstride

	def forward(self, input, indices):
		asize = list(indices.size()) 
		isize = list(input.size())
		ksize = self.ksize
		kstride = self.kstride
		x = np.array(torch.chunk(input, chunks = isize[0], dim = 0), dtype = torch.Tensor)
	    x = np.array([np.array(torch.chunk(i, chunks = isize[2]//ksize[0], dim = 2), dtype = torch.Tensor) for i in x], dtype= torch.Tensor)
	    x = np.array([np.array([np.array(torch.chunk(j, chunks = isize[3]//ksize[1], dim =3), dtype= torch.Tensor) for j in i], dtype= torch.Tensor)for i in x], dtype= torch.Tensor)

	    ind = indices.cpu().detach().numpy()
	    temp = x[ind[:,0],ind[:,1],ind[:,2]]
	    return torch.cat(tuple(temp), dim = 0)
		


class scatter2dcf(nn.Module):
	def __init__(self, ksize, kstride):
		super(scatter2dc, self).__init__()
		self.ksize = ksize
		self.kstride = kstride

	def forward(self, input, gathered, indices):
		isize = list(input.size())
		gsize = list(gathered.size())
		ksize = self.ksize
		kstride = self.kstride
		
	    x = np.array(torch.chunk(input, chunks = isize[0], dim = 0), dtype = torch.Tensor)
	    x = np.array([np.array(torch.chunk(i, chunks = isize[2]//ksize[0], dim = 2), dtype = torch.Tensor) for i in x], dtype= torch.Tensor)
	    x = np.array([np.array([np.array(torch.chunk(j, chunks = isize[3]//ksize[1], dim =3), dtype= torch.Tensor) for j in i], dtype= torch.Tensor)for i in x], dtype= torch.Tensor)
	    
	    g = np.array(torch.chunk(gathered, chunks = gsize[0], dim =0), dtype = torch.Tensor)
	    ind = indices.cpu().detach().numpy()
	    
	    x[ind[:,0],ind[:,1],ind[:,2]] = g
	    
	    x = [[torch.cat(tuple(j) , dim = 3) for j in i] for i in x]
	    x = [torch.cat(tuple(i), dim =2) for i in x]
	    return torch.cat(tuple(x), dim =0)