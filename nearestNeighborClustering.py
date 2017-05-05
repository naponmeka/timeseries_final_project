from averageSeries import *
import sys
from collections import deque
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
        data_train_by_class[floats[0]].append(floats[1:])
    else:
        data_train_by_class[floats[0]] = [floats[1:]]
    # queue_train.append([idx]+floats)
    # data_train.append(floats)
f.close()
for key_classname, value in data_train_by_class.items():
    print("DOING CLASS: {}".format(key_classname))
    queue_train.append(float('inf'))
    for v in value:
        queue_train.append(v)
    # do the clustering
    clusters = {}
    cluster_no = 0
    group_leader = []
    while queue_train:
        current = queue_train.popleft()
        if current == float('inf'):
            cluster_no += 1
            if queue_train:
                print("new group: {}".format(cluster_no))
                group_leader = queue_train.popleft()
                clusters[cluster_no] = [group_leader[0]]
            else:
                break
            queue_train.append(float('inf'))
        else:
            if cluster_no in clusters and len(clusters[cluster_no]) == 3:
                cluster_no += 1
            distance = DTWDistance(group_leader[1:], current[1:], 30)
            # print(distance)
            if distance < 0.5:
                if cluster_no in clusters:
                    clusters[cluster_no].append(current[0])
                else:
                    clusters[cluster_no] = [current[0]]
            else:
                queue_train.append(current)
    print(clusters)

sys.exit(0)
data_test = []
# f = open('ClassificationClusteringDatasets/' + test_filename)
# for line in f:
#     floats = [float(x) for x in line.strip().split()]
#     data_test.append(floats)
# f.close()



# print(clusters)
sys.exit(0)