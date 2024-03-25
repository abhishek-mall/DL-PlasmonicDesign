import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
data =  pd.read_csv("Pair1.csv")

data = data.values
geometry = data[0:data.shape[0], 0:4]
data = data[0:data.shape[0], 4:105]

a = 60
b = 70
c = 80

minDiff = 100
index = 900


def desire():
    array = np.zeros(101)
    for f in range(0, 101):
        if (f <= a):
            val = 0.6
        if (b>=f>=a):
            val = 0.1
        if (c>=f>=b):
            val = 0.6
        if (f>=c):
            val = 0.1
            
        array[f] = val
    return array

def MAE(real, generated):
    diff = abs(real - generated)
    diff = np.mean(diff)   
    return diff

sample = desire()
for i in range(0, data.shape[0]):
     diff = MAE(data[i], sample)
     if( diff  <= minDiff):
         minDiff = diff
         index = i

print(minDiff, index)
print(geometry[index])

plt.figure("Comparison")
plt.plot(sample)
plt.plot(data[index])



concat= np.vstack((sample, data[index]))
print(concat.shape)
np.savetxt("desired_Pair1.csv", concat, delimiter=",")


