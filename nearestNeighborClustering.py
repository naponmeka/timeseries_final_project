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
queue_train.append(float('inf'))
for idx,line in enumerate(f):
    floats = [float(x) for x in line.strip().split()]
    queue_train.append([idx]+floats)
    data_train.append(floats)
f.close()

data_test = []
# f = open('ClassificationClusteringDatasets/' + test_filename)
# for line in f:
#     floats = [float(x) for x in line.strip().split()]
#     data_test.append(floats)
# f.close()


clusters = {}
cluster_no = 0
group_leader = []
while queue_train:
    current = queue_train.popleft()
    if current == float('inf'):
        cluster_no += 1
        print("new group: {}".format(cluster_no))
        if queue_train:
            group_leader = queue_train.popleft()
        else:
            break
        queue_train.append(float('inf'))
    else:
        distance = DTWDistance(group_leader[2:], current[2:], 30)
        # print(distance)
        if distance < 0.5:
            if cluster_no in clusters:
                clusters[cluster_no].append((current[0],current[1]))
            else:
                clusters[cluster_no] = [(current[0],current[1])]
        else:
            queue_train.append(current)
print(clusters)
sys.exit(0)