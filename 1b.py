from math import *
import sys
import time
from joblib import Parallel, delayed
import multiprocessing
from averageSeries import *

def upper_keogh(seq):
    upper_seq = []
    for i in range(len(seq)):
        max_value = seq[i]
        for j in range(r):
            if i - j < 0: continue
            if seq[i-j] > max_value: max_value = seq[i-j]
        for j in range(r):
            if i + j >= len(seq): continue
            if seq[i+j] > max_value: max_value = seq[i-j]
        upper_seq.append(max_value)
    return upper_seq

def lower_keogh(seq):
    lower_seq = []
    for i in range(len(seq)):
        min_value = seq[i]
        for j in range(r):
            if i - j < 0: continue
            if seq[i-j] < min_value: min_value = seq[i-j]
        for j in range(r):
            if i + j >= len(seq): continue
            if seq[i+j] < min_value: min_value = seq[i-j]
        lower_seq.append(min_value)
    return lower_seq

start = time.time()
train_filename = 'synthetic_control_TRAIN'
test_filename = 'synthetic_control_TEST'
f = open('ClassificationClusteringDatasets/' + train_filename)
data_train = []
for line in f:
    floats = [float(x) for x in line.strip().split()]
    data_train.append(floats)
f.close()

data_test = []
train_lower_b = []
train_upper_b = []
f = open('ClassificationClusteringDatasets/' + test_filename)
for line in f:
    floats = [float(x) for x in line.strip().split()]
    data_test.append(floats)
    train_upper_b.append(upper_keogh(floats[1:]))
    train_lower_b.append(lower_keogh(floats[1:]))
f.close()

# weight1 = 1
# weight2 = 1
# weight3 = 1
def DTWDistanceCustom(s1, s2, w):
    DTW={}
    w = max(w, abs(len(s1)-len(s2)))
    for i in range(-3, len(s1)):
        for j in range(-3, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            # DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            ### number 1
            # DTW[(i, j)] = min(
            #                 DTW[(i-1, j-1)] + 2 * dist,
            #                 DTW[(i-1, j)] + 1 * dist,
            #                 DTW[(i, j-1)] + 1 * dist)
            ### number 2
            # DTW[(i, j)] = min(
            #                 DTW[(i-1, j-3)] + 2 * ( s1[i] - s2[j-2] )**2 + ( s1[i] - s2[j-1] )**2 + dist,
            #                 DTW[(i-1, j-2)] + 2 * ( s1[i] - s2[j-1] )**2 + dist,
            #                 DTW[(i-1, j-1)] + 2 * dist,
            #                 DTW[(i-2, j-1)] + 2 * ( s1[i-1] - s2[j] )**2 + dist,
            #                 DTW[(i-3, j-1)] + 2 * ( s1[i-2] - s2[j] )**2 + ( s1[i-1] - s2[j] )**2 + dist)
                            
            ### number 3
            # DTW[(i, j)] = min(
            #                 DTW[(i-1, j-2)] + 2 * ( s1[i] - s2[j] )**2 + dist ,
            #                 DTW[(i-1, j-1)] + 2 * dist,
            #                 DTW[(i-2, j-1)] + 2 * (s1[i-1] - s2[j])**2 + dist)
            ### number 4
            # DTW[(i, j)] = min(
            #                 DTW[(i-2, j-3)] + 2 * ( s1[i-1] - s2[j-2] )**2 + 2 * ( s1[i] - s2[j-1] )**2 + dist,
            #                 DTW[(i-1, j-1)] + 2 * dist,
            #                 DTW[(i-3, j-2)] + 2 * ( s1[i-2] - s2[j-1] )**2 + 2 * ( s1[i-1] - s2[j] )**2 + dist)
            ### ARROW SHAPE
            # DTW[(i, j)] = min(
            #                 DTW[(i-1, j-2)] + 3 * dist,
            #                 DTW[(i-1, j-1)] + 2 * dist,
            #                 DTW[(i-2, j-1)] + 3 * dist)
            ### OUR SHAPE
            DTW[(i, j)] = min(
                            DTW[(i-1, j)] + 1 * dist,
                            DTW[(i-1, j-1)] + 2 * dist,
                            DTW[(i, j-2)] + 2 * dist)
    return sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(C, idx):
    sum_weight = 0
    for i in range(len(C)):
        if C[i] > train_upper_b[idx][i]:
            sum_weight += (C[i] - train_upper_b[idx][i]) ** 2
        elif C[i] < train_lower_b[idx][i]:
            sum_weight += (C[i] - train_lower_b[idx][i]) ** 2
    return sum_weight

correct_predictions = 0

# inputs = data_test
def process(idx,d_test):
    correct_predictions = 0
    min_dist = float('inf')
    the_label = 0
    for idx2, d_train in enumerate(data_train):
        LB_dist = LB_Keogh(d_test[1:], idx2)
        if LB_dist < min_dist:
            distance = DTWDistanceCustom(d_test[1:],d_train[1:],30)
            if distance < min_dist:
                min_dist = distance
                the_label = d_train[0]
    if the_label == d_test[0]:
        return 1
    else:
        return 0
    # end = time.time()
    # return correct_predictions
print("START")
num_cores = multiprocessing.cpu_count()
correct_predictions = Parallel(8)(delayed(process)(idx,d_test) for idx,d_test in enumerate(data_test) )

cc = 0
for i in correct_predictions:
    cc+=i
print(cc/len(data_test))
# print(correct_predictions/len(data_test))
