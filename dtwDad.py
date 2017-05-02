from math import *
from scipy.optimize import basinhopping
import sys
import time
import statistics
import matplotlib.pyplot as plt

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

# weight1 = 1
# weight2 = 1
# weight3 = 1
def DTWDistance(s1, s2, w, weight1, weight2, weight3):
    DTW={}
    w = max(w, abs(len(s1)-len(s2)))
    for i in range(-2, len(s1)):
        for j in range(-2, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist = (s1[i]-s2[j])**2
            # DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            DTW[(i, j)] = min(DTW[(i-1, j-1)] + weight1 * dist,
                            DTW[(i-1, j)] + weight2 * dist,
                            DTW[(i, j-1)] + weight3 * dist)
    return sqrt(DTW[len(s1)-1, len(s2)-1])

def DTWCostMatrix(s1, s2, w):
    DTW={}
    w = max(w, abs(len(s1)-len(s2)))
    for i in range(-2, len(s1)):
        for j in range(-2, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist = (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
    # return sqrt(DTW[len(s1)-1, len(s2)-1])
    return DTW

correct_predictions = 0

def LB_Keogh(C, idx):
    sum_weight = 0
    for i in range(len(C)):
        if C[i] > train_upper_b[idx][i]:
            sum_weight += (C[i] - train_upper_b[idx][i]) ** 2
        elif C[i] < train_lower_b[idx][i]:
            sum_weight += (C[i] - train_lower_b[idx][i]) ** 2
    return sum_weight
'''
for idx,d_test in enumerate(data_test):
    min_dist = float('inf')
    the_label = 0
    for idx2, d_train in enumerate(data_train):
        LB_dist = LB_Keogh(d_test[1:], idx2)
        if LB_dist < min_dist:
            weight1 = 1
            weight2 = 1
            weight3 = 1
            distance = DTWDistance(d_test[1:],d_train[1:],window_size,weight1,weight2,weight3)
            if distance < min_dist:
                min_dist = distance
                the_label = d_train[0]
    if the_label == d_test[0]:
        correct_predictions += 1
    end = time.time()
    print("index:{}, time:{}".format(idx, end - start))
print(correct_predictions/len(data_test))
'''
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
    cost =  DTWCostMatrix(s_ref, s, 20)
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
mean = dba(class_a, 1)
# print(mean)
# print(counting)
plt.plot(mean)
# x = DTWCostMatrix(class_a[0],class_a[1],10)
# print(x)
# plt.ylabel('some numbers')
plt.show()
# data_train = []