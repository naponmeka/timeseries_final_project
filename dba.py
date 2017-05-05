from math import *
from scipy.optimize import basinhopping
import sys
import time
import statistics
import matplotlib.pyplot as plt
from computeAccuracy import *

start = time.time()
train_filename = 'Trace_ALL'
test_filename = 'Beef_TEST'
f = open('ClassificationClusteringDatasets/' + train_filename)
data_train = []
train_lower_b = []
train_upper_b = []
window_size = 10
r = 5
data_train_dict = {}

for line in f:
    floats = [float(x) for x in line.strip().split()]
    ts = floats[1:]
    ts = normalizeSeries(ts)

    if floats[0] in data_train_dict:
        data_train_dict[floats[0]].append(ts)
    else:
        data_train_dict[floats[0]] = [ts]
    # data_train.append(floats)
    train_upper_b.append(upper_keogh(ts))
    train_lower_b.append(lower_keogh(ts))
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

the_mean = 0
sum_weight = 0
meanDistances = []
weights_to_mean = []
for key, one_class_data in data_train_dict.items():
    mean = dba(one_class_data, 10)
    distance = avgMeanErrorEuclideanDistance(mean, one_class_data)
    meanDistances.append(distance)
    weights_to_mean.append(len(one_class_data))

print("MEAN DIS")
print(meanDistances)
print("AVG MEAN DIS")
the_mean = 0
sum_weight = 0
for idx,m in enumerate(meanDistances):
    the_mean += m * weights_to_mean[idx]
    sum_weight += weights_to_mean[idx]
the_mean = the_mean/sum_weight
print(the_mean)