from math import *
from scipy.optimize import basinhopping
import sys
import time
import statistics
import matplotlib.pyplot as plt
from computeAccuracy import *

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

def dba(D, I):
    # D - the set of sequence to average
    # I - number of iteration
    T = D[0] # get the medoid? from the set D
    for i in range(I):
        T = dba_update(T,D)
    return T

def dba_update(T, D):
    # T - the average sequence to refine ( lenght = L )
    # D - the set of sequence to average
    #STEP 1 : compute the multiple alignment for T
    L = len(T)
    alignment = [set() for _ in range(L)]
    for s in D:
        alignment_for_s = dtw_multiple_alignment(T, s)
        for i in range(L):
            alignment[i] = alignment[i].union(alignment_for_s[i])
    #STEP2 : compute the multiple alignment for the alignment
    T = [0]*L
    for i in range(L):
        if len(alignment[i]) == 0:
            T[i] = 0
        else:
            T[i] = statistics.mean(alignment[i])
    return T

def dtw_multiple_alignment(s_ref, s):
    # STEP 1 : Compute the accumulated cost matrix of DTW
    cost, path =  DTWCostMatrix(s_ref, s, 20)
    # print(cost)
    # sys.exit(0)
    # STEP 2 : Store element associate to s_ref
    L = len(s_ref)
    alignment = [set() for _ in range(L)]
    i = len(s_ref) - 1
    j = len(s) - 1 
    while i > 1 and j > 1 :
        alignment[i] = alignment[i].union(set([s[j]]))
        if i == 1:  j = j -1
        elif j ==1: i = i-1
        else:
            score = min(cost[(i-1, j-1)], cost[(i, j-1)], cost[(i-1, j)])
            if score == cost[(i-1, j-1)]:
                i = i - 1
                j = j - 1
            elif score== cost[(i-1, j)]:
                    i = i - 1
            else: j = j -1
    return alignment

class_a = []
counting = 0
first_label = data_train[0][0]
for d in data_train:
    if first_label == d[0]:
        class_a.append(d[1:])
        # plt.plot(d[1:])
        counting += 1
# print(class_a)
# plt.show()
mean = dba(class_a, 10)
distance = avgMeanErrorEuclideanDistance(mean, class_a)
print("AVD DISTANCE: {}".format(distance))
plt.plot(mean)
plt.show()
