from math import *
# from scipy.optimize import basinhopping
import sys
import time
# import matplotlib.pyplot as plt

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

# start = time.time()
# train_filename = 'Beef_TRAIN'
# test_filename = 'Beef_TEST'
# f = open('ClassificationClusteringDatasets/' + train_filename)
# data_train = []
# train_lower_b = []
# train_upper_b = []
window_size = 10
r = 5

# for line in f:
#     floats = [float(x) for x in line.strip().split(' ')]
#     data_train.append(floats)
#     train_upper_b.append(upper_keogh(floats[1:]))
#     train_lower_b.append(lower_keogh(floats[1:]))
# f.close()

# data_test = []
# f = open('ClassificationClusteringDatasets/' + test_filename)
# for line in f:
#     floats = [float(x) for x in line.strip().split(' ')]
#     data_test.append(floats)
# f.close()

# # weight1 = 1
# # weight2 = 1
# # weight3 = 1

# correct_predictions = 0

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

# def noobmean(s1):
#     if(len(s1)==1) return s1
#     out = []
#     for s in s1:
#         out.append()


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
    path={}
    w = max(w, abs(len(s1)-len(s2)))
    for i in range(-2, len(s1)):
        for j in range(-2, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist = (s1[i]-s2[j])**2
            minVar = min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            DTW[(i, j)] = dist + minVar
            if minVar == DTW[(i-1, j)]: path[(i,j)] = (i-1,j)
            elif minVar == DTW[(i, j-1)]: path[(i,j)] = (i,j-1)
            else : path[(i,j)] = (i-1,j-1)
            
    # return sqrt(DTW[len(s1)-1, len(s2)-1])
    return DTW,path

def genVectorBase(value, base_v):
    vec = []
    while(value > 0):
        current_value = value % base_v
        vec.append(current_value)
        value = value // base_v
    return vec

def DTWCostNDimMatrix(seqs):
    nDim = len(seqs)
    DTW = {}
    path = {}
    for i in range((len(seqs[0])+2) ** nDim):
        n_vec = genVectorBase(i, len(seqs[0])+2)
        for j in range(len(n_vec)):
            n_vec[j] -= 2
        DTW[n_vec] = float('inf')
    n_vec = genVectorBase(0, len(seqs[0])+2)
    for j in range(len(n_vec)):
        n_vec[j] -= 1
    DTW[n_vec] = 0
    for item in range((len(seqs[0])) ** nDim):
        index = genVectorBase(item, len(seqs[0]))
        dist = 0
        for i in range(nDim):
            for j in range(i+1,nDim):
                dist += (seqs[i][index[i]] - seqs[j][index[j]]) ** 2
        min_value = float('inf')
        min_path = [0]*nDim
        for i in range(1, 2 ** nDim):
            neg_vec = genVectorBase(i, 2)
            new_vec = index.clone()
            for j in range(len(neg_vec)):
                new_vec[j] -= neg_vec[j]
            if min_value > DTW[new_vec]:
                 min_value = DTW[new_vec]
                 min_path = new_vec
        path[index] = min_path
    return DTW, path

def CalPath(path, next):
    if next[0] < 0 or next[1] < 0 : return 
    # print(next)
    global memPath
    memPath = [next] + memPath
    CalPath(path, path[(next)])

def CalNDimPath(path, next):
    # if next[0] < 0 or next[1] < 0 : return
    for i in next:
        if i < 0:
            return
    # print(next)
    global memPath
    memPath = [next] + memPath
    CalPath(path, path[(next)])

def uniScaling(s1,to_len):
    current_len = len(s1)
    ratio = current_len/to_len  
    result=[]
    for i in range(1,to_len+1):
        # print(i)
        # print(ceil(i*(ratio)))
        result.append( s1[ceil(i*(ratio))-1] )
        
    return result

def toMatrix(input,_mSize):
    Matrix = [[0 for x in range(_mSize)] for y in range(_mSize)]
    for key,value in input.items():
        Matrix[key[0]][key[1]] = value
    return Matrix

def getSeriefromPath(_memPath,_s1,_s2):
    new_serie= []
    for point in _memPath:
        new_serie.append((_s1[point[0]] + _s2[point[1]])*0.5)
    return new_serie

def getSeriefromPathNDim(_memPath,series):
    new_serie= []
    for mapped_points in _memPath:
        sum_value = 0
        for idx,point in enumerate(mapped_points):
            sum_value += series[idx][point]
        new_serie.append(sum_value / len(series))
        #new_serie.append((_s1[point[0]] + _s2[point[1]])*0.5)
    return new_serie

def average_ts(_s1, _s2):
    _s2 = uniScaling(_s2,len(_s1))
    costMap,path = DTWCostMatrix(_s1, _s2,10)
    costMatrix = toMatrix(costMap,len(_s1))
    global memPath
    memPath=[]
    CalPath(path,path[(len(_s1)-1,len(_s1)-1)])
    unScaledSeries = getSeriefromPath(memPath,_s1,_s2)
    avgSerie = uniScaling(unScaledSeries,len(_s1))
    return avgSerie

def average_n_ts(series):
    for i in range(1, len(seires)):
        series[i] = uniScaling(series[i],len(series[0]))
    costMap,path = DTWCostNDimMatrix(series)
    global memPath2
    memPath2=[]
    last_index = genVectorBase(len(series[0]) ** len(series) - 1, len(series[0])))
    CalNDimPath(path,path[last_index])
    unScaledSeries = getSeriefromPathNDim(memPath2, series)
    avgSerie = uniScaling(unScaledSeries, len(_s1))
    return avgSerie
#
# x,path = DTWCostMatrix(class_a[0],class_a[1],10)

# w,h = 470,470
# Matrix = [[0 for x in range(w)] for y in range(h)]

# for key,value in x.items():
#     Matrix[key[0]][key[1]] = value
# plt.figure(1)
# plt.plot(Matrix)

# MatrixP = [[0 for x in range(w)] for y in range(h)]


# for key,value in path.items():
#     MatrixP[key[0]][key[1]] = value
# print(MatrixP[469][469])
# # plt.figure(2)
# # plt.plot(MatrixP)
# plt.show()
memPath=[]
memPath2=[]
# CalPath(path, path[(469,469)])
# new_serie= []
# for point in memPath:
#     new_serie.append((class_a[0][point[0]] + class_a[1][point[1]])*0.5)
# print(len(new_serie))
# plt.plot(new_serie)
# plt.show()
print(genVectorBase(14237,100))
# print('Hello')
# scaled = uniScaling(new_serie,400)
# plt.plot(scaled)
# plt.show();