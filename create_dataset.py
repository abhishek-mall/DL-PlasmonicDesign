import torch
import os
import numpy as np
import pandas as pd
import torch.utils.data
from models import Encoder

#Range of values  rectangle-square pair geometrical class

# Lx -- 30 - 70 nm      41 values
# Ly -- 100 - 200 nm    101 values
# L -- 20 - 60 nm       41 values
# t -- 50 -150nm        101 values

# Total output layer 284

maxVals = np.array([70.00, 200.00, 60.00, 150.00])
def adjustIndex(geometry):
	return geometry / maxVals

def readData(path):

	Data = []

	csvPath = path + "PairTwoSpectrum\\"
	file = path + "PairTwoGeometry.xls"
	data = pd.ExcelFile(file)
	geometryData = data.parse(0)
	geometryData = geometryData.values[:, 1:]

	for csvFile in os.listdir(csvPath):
		index = int(csvFile.split(".")[0]) 
		geometry = geometryData[index-1]
		geometry = adjustIndex(geometry)     # Scaling the structural parameters values
		geometry = torch.tensor(geometry).float() 
		spctr = pd.read_csv(csvPath + csvFile, header=None)
		spctr = torch.tensor(spctr.values).float()
		dataPoint = (geometry, spctr)
		Data.append(dataPoint)

	return Data

