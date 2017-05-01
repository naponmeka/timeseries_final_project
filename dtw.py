from math import *
from scipy.optimize import basinhopping
import sys
import time

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
    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            # DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            DTW[(i, j)] = min(DTW[(i-1, j-1)] + weight1 * dist,
                            DTW[(i-1, j)] + weight2 * dist,
                            DTW[(i, j-1)] + weight3 * dist)
    return sqrt(DTW[len(s1)-1, len(s2)-1])

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
def func3d(x):
    correct_predictions = 0
    for idx,d_test in enumerate(data_test):
        min_dist = float('inf')
        the_label = 0
        for idx2, d_train in enumerate(data_train):
            LB_dist = LB_Keogh(d_test[1:], idx2)
            if LB_dist < min_dist:
                distance = DTWDistance(d_test[1:],d_train[1:],window_size,x[0],x[1],x[2])
                if distance < min_dist:
                    min_dist = distance
                    the_label = d_train[0]
        if the_label == d_test[0]:
            correct_predictions += 1
        end = time.time()
        # print("index:{}, time:{}".format(idx, end - start))
    # print(correct_predictions/len(data_test))
    return -1*correct_predictions/len(data_test)



x0 = [1.0, 1.0, 1.0]
# the bounds
xmin = [0., 0., 0.]
xmax = [3., 3., 3.]

# rewrite the bounds in the way required by L-BFGS-B
bounds = [(low, high) for low, high in zip(xmin, xmax)]
minimizer_kwargs = {"method": "L-BFGS-B", "bounds":bounds}
print('start basinhopping')
ret = basinhopping(func3d, x0, minimizer_kwargs=minimizer_kwargs, niter=100)
print("global minimum: x = [%.4f, %.4f, %.4f], f(x0) = %.4f" % (ret.x[0],ret.x[1],ret.x[2],ret.fun))
print('done')
end = time.time()
print(end - start)