from averageSeries import *
import sys
from collections import deque
# import matplotlib.pyplot as plt
from computeAccuracy import *
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

print("Hello")
for idx,line in enumerate(f):
    floats = [float(x) for x in line.strip().split()]
    if floats[0] in data_train_by_class:
        data_train_by_class[floats[0]].append(uniScaling(floats[1:],10))
    else:
        data_train_by_class[floats[0]] = [uniScaling(floats[1:],10)]
f.close()

def find_closest_three(arr_of_seq):
    usedIndex = []
    cluster = {}
    cluster_no = 0
    result = []
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
            distance = DTWDistance(value['serie'], value2['serie'],30)
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
meanDistances = []
weights_to_mean = []
for key_classname, one_class_data in data_train_by_class.items():
    print("DOING CLASS: {}".format(key_classname))
    arr_dict_one_data = []
    for one in one_class_data:
        one_dict = {
            'serie': one,
            'weight': 1
        }
        arr_dict_one_data.append(one_dict)
    cluster_of_three = find_closest_three(arr_dict_one_data)

    # weights = []
    # for group in cluster_of_three:
    #     weight=[]
    #     for one in group:
    #         weight.append(1)
    #     weights.append(weight)
    results = []
    
    while True:
        if len(cluster_of_three) == 1:break
        print(len(cluster_of_three))
        counting = 0
        # weight_results = []
        for i,c in enumerate(cluster_of_three):
            counting +=1
            result = average_n_ts_band_tuple(c)
            # new_weight = sum(weights[i])
            # weight_results.append(new_weight)
            results.append(result)
        # weights = weight_results
        # weights
        cluster_of_three = find_closest_three(results)
        #group weight item 
    sys.exit(0)
    the_mean_seq = cluster_of_three[0][0]
    distance = avgMeanErrorEuclideanDistance(the_mean_seq, one_class_data)
    meanDistances.append(distance)
    weights_to_mean.append(len(one_class_data))
    graphs_to_plot.append(the_mean_seq)

the_mean = 0
sum_weight = 0
for idx,m in enumerate(meanDistances):
    the_mean += m * weights_to_mean[idx]
    sum_weight += weights_to_mean[idx]
the_mean = the_mean/sum_weight
print(the_mean)

# for avg_result in graphs_to_plot:
#     plt.figure()
#     plt.plot(avg_result)
#     plt.show()