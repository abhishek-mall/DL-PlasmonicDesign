import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
data =  pd.read_csv("DoubleArc.csv")

data = data.values
geometry = data[0:data.shape[0], 0:4]
data = data[0:data.shape[0], 4:105]

f0 =42
f1 =56
f2 = 81
sigma0 = 10
sigma1 = 7
sigma2 = 8
minDiff = 100
index = 900


def desire():
    array = np.zeros(101)
    for f in range(0, 101):
        val = 0.7* math.exp(-((f-f0)**2)/(2*(sigma0**2))) + 0.4* math.exp(-((f-f1)**2)/(2*(sigma1**2))) + 0.6* math.exp(-((f-f2)**2)/(2*(sigma2**2)))
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
np.savetxt("Gaussian_npj_DoubleArc.csv", concat, delimiter=",")


