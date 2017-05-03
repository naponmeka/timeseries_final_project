import math
from averageSeries import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import itertools
import statistics

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import sys
from computeAccuracy import *

train_filename = 'Beef_TRAIN'
test_filename = 'Beef_TEST'
f = open('ClassificationClusteringDatasets/' + train_filename)
data_train = []
train_lower_b = []
train_upper_b = []
window_size = 30
r = 5
lookupData = []

data_train_dict = {}
lookupData_dict = {}
for idx,line in enumerate(f):
    floats = [float(x) for x in line.strip().split()]
    ts = floats[1:]
    ts = normalizeSeries(ts)
    # if floats[0] == 1.000000:
    #     data_train.append(np.array(ts))
    #     lookupData.append(ts)
    if floats[0] in data_train_dict:
        data_train_dict[floats[0]].append(np.array(ts))
    else:
        data_train_dict[floats[0]] = [np.array(ts)]
    
    if floats[0] in lookupData_dict:
        lookupData_dict[floats[0]].append(ts)
    else:
        lookupData_dict[floats[0]] = [ts]
    
f.close()
def psa_average(c, num_of_data, class_name):
    if c < num_of_data:
        return [lookupData_dict[class_name][c], 1]
    else:
        [z1, r1] = psa_average(dict_tree[c][0], num_of_data, class_name)
        [z2, r2] = psa_average(dict_tree[c][1], num_of_data, class_name)
        z = average_ts_weight(z1, z2, r1, r2)
        return [z, r1+r2]

meanDistances = []
for key, one_class_data in data_train_dict.items():
    class_name = key
    number_of_data = len(one_class_data)
    X = one_class_data
    X = np.array(X)
    # print("NP size : {}".format(X.shape[0]))
    n_clusters = 3
    # for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
    metric = "cityblock"
    model = AgglomerativeClustering(n_clusters=n_clusters,
                                    linkage="average", affinity=metric)
    model.fit(X)

    ii = itertools.count(X.shape[0])
    my_h_tree = [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in model.children_]
    dict_tree = {}
    # print(my_h_tree)
    nodes = {}
    for elem in my_h_tree:
        nodes[elem['node_id']] = True
        dict_tree[elem['node_id']] = (elem['left'],elem['right'])
    print(nodes)
    for elem in my_h_tree:
        nodes[elem['left']] = False
        nodes[elem['right']] = False
    print("---")
    print(nodes)
    root = 0
    for elem in my_h_tree:
        if nodes[elem['node_id']] == True:
            root = elem['node_id']
    #### PSA ####
    Z,r = psa_average(root, number_of_data, class_name)
    result = Z
    m = avgMeanErrorEuclideanDistance(result, lookupData_dict[class_name])
    meanDistances.append(m)

print("MEAN DIS")
print(meanDistances)
print("AVG MEAN DIS")
print(statistics.mean(meanDistances))
# TODO: weighted mean