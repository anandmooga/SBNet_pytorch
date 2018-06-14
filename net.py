# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:24:11 2018

@author: anand mooga
"""
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np 
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader 
import torch.nn.functional as F
import keras
from keras.datasets import mnist

from sparsenet import sparse_block, reduce_mask_pool2d, pad_input, sparse_gather2d, sparse_scatter2d, mask_pool2d

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
num_classes = 10
batch_size = 32
learning_rate = 0.01

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train/255
X_test = X_test/255

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categorical_features = [0])
y_train1 = enc.fit_transform(y_train.reshape(-1,1)).toarray()

X_train_t =  torch.FloatTensor(X_train[:, np.newaxis, :, :]).to(device)
y_train_t =  torch.FloatTensor(y_train1).to(device)

mask_t = X_train_t.clone().to(device)
mask_t = (mask_t > (10/255)).float()
mask_t.squeeze()
mask_t.require_grad = False

img_channel, img_height, img_width = X_train.shape[0], X_train.shape[1], X_train.shape[2]


class net(nn.Module):
	def __init__(self, inp_ch, num_classes):
		super(net, self).__init__()
		self.sparse1 = sparse_block(inp_ch, 8, [5,5], [4,4])
		self.mp1 = nn.MaxPool2d(2)
		self.sparse2 = sparse_block(8, 16, [5,5], [4,4])
		self.mp2 = nn.MaxPool2d(2)
		self.fc = nn.Sequential(
			nn.Linear(16*8*8, num_classes),
			nn.Softmax()
			)
	def forward(self, x, mask1):
		x = self.sparse1(x, mask1)
		x = self.mp1(x)
		mask2 = mask_pool2d(mask1)
		x = self.sparse2(x, mask2)
		x = self.mp2(x)
		x = x.view(batch_size, -1)
		x = self.fc(x)
		return x

model = net(img_channel, num_classes).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


#train
model.train()
for epoch in range(1, num_epochs+1):
	print("Epoch = ", epoch)
	z = 0
	for i in range(len(X_train)//batch_size):
		x = X_train_t[z:z+batch_size, :, :, :]
		y = y_train_t[z:z+batch_size]
		mask = mask_t[z:z+batch_size, :, :]
		z += batch_size
		labels = labels_t[z:z+batch_size]

		#Forward pass
		output = model(x, mask)
		loss = criterion(output, y)

		#Backward pass
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if z%10 == 0:
			_, predicted = torch.max(output.data, 1)
			total = batch_size
			correct = (predicted == labels ).sum().item()
			print('\rStep [{}/{}], Loss: {:.4f} , acc = {:.2f}%'.format(i, len(X_train)//batch_size, loss.data[0], correct*100/total), end = ' ')



