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

class_a = []
counting = 0
first_label = data_train[0][0]
for d in data_train:
    if first_label == d[0]:
        class_a.append(d[1:])
        # plt.plot(d[1:])
        counting += 1
memPath=[]
avgSerie = average_ts(class_a[2],class_a[1])
plt.plot(avgSerie)
plt.show()