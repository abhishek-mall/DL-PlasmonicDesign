import torch
import os
import numpy as np
import pandas as pd
import torch.utils.data
from models import Encoder, Decoder
from create_dataset import readData
import torch.nn as nn

#Range of values for rectangle-square pair geometrical class

# Lx -- 30 - 70 nm
# Ly -- 100 - 200 nm
# L -- 20 - 60 nm
# t -- 50 -150nm
# GPU
use_cuda = 1
device = torch.device("cuda" if use_cuda else "cpu")

maxVals = torch.tensor([30.00, 100.00, 20.00, 50.00]).to(device)   # Error : Divided my minimum values instead of maximum values


# Helper functions

def writeInFiles(pred, gt, folder, epoch):
	pred = pred.detach().cpu().numpy()
	gt = gt.detach().cpu().numpy()
	if(folder == "Geometry_3000/"):
		pred = pred * maxVals
		gt = gt * maxVals

	for i in range(0, pred.shape[0]):
		f = folder + str(epoch) + "_" + str(i) + ".txt"
		p = pred[i] 
		np.savetxt(f, p,  delimiter='\t')

	for i in range(0, gt.shape[0]):
		f = folder + str(epoch) + "_" + str(i) + "original.txt"
		p = gt[i]
		np.savetxt(f, p,  delimiter='\t')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.eval()
    return model

def exp_loss(a,b):
    diff = torch.abs(a-b)
    exp_diff = torch.exp(diff) -1
    sum_exp_diff = torch.mean(exp_diff)
    return sum_exp_diff


# Hyperparameters
batchSize = 25
num_epochs = 10000
lr = 0.00001
weight_decay = 0.1

# Initialzing encoder model and optimizers
encoder = Encoder().to(device)
decoder = Decoder().to(device)
# encoder = load_checkpoint("models/encoder1.pth").to(device)
# decoder = load_checkpoint("models/decoder1.pth").to(device)
encOptim = torch.optim.Adam(encoder.parameters(), lr=0.0005)
decOptim = torch.optim.Adam(decoder.parameters(), lr=0.0005)
l1Loss = nn.L1Loss()
mse = nn.MSELoss()

def trainEncoder(epoch, trainData):
	strLoss, spctrLoss = 0.0, 0.0
	for (iter, (geometry, spctr)) in enumerate(trainData):
		encoder.zero_grad()
		decoder.zero_grad()
		spctr = spctr.to(device)
		geometry = geometry.to(device)
		spctr = spctr.view(spctr.shape[0], -1)
		out = encoder(spctr)
		spctrPred = decoder(out)
		strLoss += l1Loss(out*maxVals, geometry*maxVals)
		spctrLoss += mse(spctrPred, spctr)
		loss = mse(out, geometry) 
		loss.backward()
		encOptim.step()	

	print("Epoch number:", epoch)
	print("Geometry Loss:", strLoss.item()/(iter+1))
	print("Spectrum Loss:", spctrLoss.item()/(iter+1))

def trainDecoder(epoch, trainData):
	strLoss, spctrLoss = 0.0, 0.0
	for (iter, (geometry, spctr)) in enumerate(trainData):
		encoder.zero_grad()
		decoder.zero_grad()
		spctr = spctr.to(device)
		geometry = geometry.to(device)
		spctr = spctr.view(spctr.shape[0], -1)
		spctrPred = decoder(geometry)
		spctrLoss += mse(spctrPred, spctr)
		loss = mse(spctrPred, spctr) + exp_loss(spctrPred, spctr)
		loss.backward()
		decOptim.step()

	print("Epoch number:", epoch)
	# print("Geometry Loss:", strLoss.item()/(iter+1))
	print("Spectrum Loss:", spctrLoss.item()/(iter+1))

# defining function to train 
def train(epoch, trainData):
	strLoss, spctrLoss = 0.0, 0.0
	for (iter, (geometry, spctr)) in enumerate(trainData):
		encoder.zero_grad()
		decoder.zero_grad()
		spctr = spctr.to(device)
		geometry = geometry.to(device)
		spctr = spctr.view(spctr.shape[0], -1)
		out = encoder(spctr)
		spctrPred = decoder(out)
		strLoss += mse(out, geometry)
		spctrLoss += mse(spctrPred, spctr)
		loss = mse(out, geometry) + exp_loss(spctrPred, spctr) + mse(spctrPred, spctr)
		loss.backward()
		encOptim.step()	
		decOptim.step()

	print("Epoch number:", epoch)
	print("Geometry Loss:", strLoss.item()/(iter+1))
	print("Spectrum Loss:", spctrLoss.item()/(iter+1))

def validate(epoch, valData):
	strLoss, spctrLoss = 0.0, 0.0
	for (iter, (geometry, spctr)) in enumerate(valData):
		encoder.zero_grad()
		decoder.zero_grad()
		spctr = spctr.to(device)
		geometry = geometry.to(device)
		spctr = spctr.view(spctr.shape[0], -1)
		out = encoder(spctr)
		spctrPred = decoder(out)
		strLoss += l1Loss(out*maxVals, geometry*maxVals)
		spctrLoss += mse(spctrPred, spctr)
		# writeInFiles(out, geometry, "Geometry_3000/", epoch)
		# writeInFiles(spctrPred, spctr, "Spectrum_3000/", epoch)

	print("Epoch number:", epoch)
	print("Geometry Loss:", strLoss.item()/(iter+1))
	print("Spectrum Loss:", spctrLoss.item()/(iter+1))

if __name__ == '__main__': 
	data = readData("E:\\Abhishek\\AE\\PairTwo\\PairTwo\\")
	np.random.shuffle(data)
	trainData = data[0:int(len(data)*0.9)]
	valData = data[int(len(data)*0.9): len(data)]
	trainData =  torch.utils.data.DataLoader(trainData, batchSize, True)
	valData =  torch.utils.data.DataLoader(valData, batchSize , True)

	for epoch in range(0, 2000):
		# train(epoch, trainData)
		trainEncoder(epoch, trainData)
		trainDecoder(epoch, trainData)
		# checkpoint = {'model': Encoder().to(device) ,'state_dict': encoder.state_dict(), 'optimizer' : encOptim.state_dict()}
		# torch.save(checkpoint, 'models/encoder1.pth')
		# checkpoint = {'model': Decoder().to(device) ,'state_dict': decoder.state_dict(), 'optimizer' : decOptim.state_dict()}
		# torch.save(checkpoint, 'models/decoder1.pth')
		# validate(epoch, valData)


	