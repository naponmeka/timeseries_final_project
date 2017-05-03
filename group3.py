from averageSeries import *
import sys
from collections import deque
import matplotlib.pyplot as plt
train_filename = 'Beef_TRAIN'
test_filename = 'Beef_TEST'
f = open('ClassificationClusteringDatasets/' + train_filename)
data_train = []
train_lower_b = []
train_upper_b = []
window_size = 30
r = 5
queue_train = deque([])

data_train_by_class = {}

for idx,line in enumerate(f):
    floats = [float(x) for x in line.strip().split()]
    if floats[0] in data_train_by_class:
        data_train_by_class[floats[0]].append(uniScaling(floats[1:],100))
    else:
        data_train_by_class[floats[0]] = [uniScaling(floats[1:],100)]
    # queue_train.append([idx]+floats)
    # data_train.append(floats)
f.close()
# print(data_train_by_class)
# sys.exit(0)
def find_closest_three(arr_of_seq):
    usedIndex = []
    cluster = {}
    cluster_no = 0
    result = []
    # newGroup = True
    # print("INPUT")
    # print(arr_of_seq)
    group = []
    for idx, value in enumerate(arr_of_seq):
        distance_and_index = []
        # print("FIRST LOOP:{}".format(idx))
        group = []
        if idx in usedIndex: continue
        group.append(arr_of_seq[idx])
        usedIndex.append(idx)
        for idx2, value2 in enumerate(arr_of_seq):
            if idx2 in usedIndex: continue
            distance = DTWDistance(value, value2,30)
            distance_and_index.append((distance,idx2))
        distance_and_index = sorted(distance_and_index)
        for idx3, p in enumerate(distance_and_index):
            if idx3 == 2 : break
            group.append(arr_of_seq[p[1]])
            usedIndex.append(p[1])
        result.append(group)
        newGroup = True

    return result

# avg_results = []
graphs_to_plot = []
for key_classname, value in data_train_by_class.items():
    print("DOING CLASS: {}".format(key_classname))
    cluster_of_three = find_closest_three(value)

    weights = []
    for group in cluster_of_three:
        weight=[]
        for one in group:
            weight.append(1)
        weights.append(weight)
    # print(cluster_of_three)
    # print(weights)
    # sys.exit(0)
    results = []
    weight_results = []
    while True:
        if len(cluster_of_three) == 1:break
        print('start while')
        print(cluster_of_three)
        counting = 0
        for i,c in enumerate(cluster_of_three):
            counting +=1
            # weights = []
            # for i in range(len(c)):
            #     weights.append(1)
            result = average_n_ts(c,weights[i])
            new_weight = sum(weights[i])
            weight_results.append(new_weight)
            results.append(result)
        print("LEN:{}".format(len(results)))
        weights = [weight_results]
        # print(weights)
        # sys.exit(0)
        cluster_of_three = [results]

    graphs_to_plot.append(cluster_of_three[0][0])


for avg_result in graphs_to_plot:
    plt.figure()
    plt.plot(avg_result)
    plt.show()