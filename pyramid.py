from averageSeries import * 
from math import *
from scipy.optimize import basinhopping
import sys
import time
import matplotlib.pyplot as plt


start = time.time()
train_filename = 'Beef_TRAIN'
test_filename = 'Beef_TEST'
f = open('ClassificationClusteringDatasets/' + train_filename)
data_train = []
train_lower_b = []
train_upper_b = []
window_size = 10
r = 5

for line in f:
    floats = [float(x) for x in line.strip().split(' ')]
    data_train.append(floats)
    train_upper_b.append(upper_keogh(floats[1:]))
    train_lower_b.append(lower_keogh(floats[1:]))
f.close()

data_test = []
f = open('ClassificationClusteringDatasets/' + test_filename)
for line in f:
    floats = [float(x) for x in line.strip().split(' ')]
    data_test.append(floats)
f.close()

def pyramid(serie):
    out=[]
    for idx,value in enumerate(serie):
        if idx==len(serie)-1: break
        else : out.append(average_ts(value,serie[idx+1]))
    if len(out)==1: return out
    else :
        print(len(out)) 
        return pyramid(out)

def pisa(serie,iteration):
    out=[]
    serie.append(serie[0])
    iteration-=1
    for idx,value in enumerate(serie):
        if idx==len(serie)-1: break
        else : out.append(average_ts(value,serie[idx+1]))
    if iteration==0: return out
    else: return pisa(out,iteration)


class_a = []
counting = 0
first_label = data_train[0][0]
for d in data_train:
    if first_label == d[0]:
        class_a.append(d[1:])
        # plt.plot(d[1:])
        counting += 1

        

memPath=[]
# avgSerie = average_ts(class_a[2],class_a[1])
print("class_a size: {}".format(len(class_a)))
# result = pyramid(class_a)
result = pisa(class_a,10)
result = pyramid(result)
plt.plot(result[0])
plt.show()