import math
from averageSeries import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import itertools

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import sys

train_filename = 'Beef_TRAIN'
test_filename = 'Beef_TEST'
f = open('ClassificationClusteringDatasets/' + train_filename)
data_train = []
train_lower_b = []
train_upper_b = []
window_size = 30
r = 5
lookupData = []
for idx,line in enumerate(f):
    floats = [float(x) for x in line.strip().split()]
    if floats[0] == 1.000000:
        data_train.append(np.array(floats[1:]))
        lookupData.append(floats[1:])
f.close()




# Generate waveform data
n_features = 2000
t = np.pi * np.linspace(0, 1, n_features)
# print(t)

def sqr(x):
    return np.sign(np.cos(x))

X = list()
y = list()
for i, (phi, a) in enumerate([(.5, .15), (.5, .6), (.3, .2)]):
    for _ in range(30):
        phase_noise = .01 * np.random.normal()
        amplitude_noise = .04 * np.random.normal()
        additional_noise = 1 - 2 * np.random.rand(n_features)
        # Make the noise sparse
        additional_noise[np.abs(additional_noise) < .997] = 0

        X.append(12 * ((a + amplitude_noise)
                 * (sqr(6 * (t + phi + phase_noise)))
                 + additional_noise))
        y.append(i)
number_of_data = len(data_train)
print("number of train: {}".format(number_of_data))
X = data_train
X = np.array(X)
y = np.array(y)
print("NP size : {}".format(X.shape[0]))

n_clusters = 3

labels = ('Waveform 1', 'Waveform 2', 'Waveform 3')

# Plot the ground-truth labelling
# plt.figure()
# plt.axes([0, 0, 1, 1])
# for l, c, n in zip(range(n_clusters), 'rgb',
#                    labels):
#     lines = plt.plot(X[y == l].T, c=c, alpha=.5)
#     lines[0].set_label(n)

# plt.legend(loc='best')

# plt.axis('tight')
# plt.axis('off')
# plt.suptitle("Ground truth", size=20)


# Plot the distances
# for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
#     avg_dist = np.zeros((n_clusters, n_clusters))
#     plt.figure(figsize=(5, 4.5))
#     for i in range(n_clusters):
#         for j in range(n_clusters):
#             avg_dist[i, j] = pairwise_distances(X[y == i], X[y == j],
#                                                 metric=metric).mean()
#     avg_dist /= avg_dist.max()
#     for i in range(n_clusters):
#         for j in range(n_clusters):
#             plt.text(i, j, '%5.3f' % avg_dist[i, j],
#                      verticalalignment='center',
#                      horizontalalignment='center')

#     plt.imshow(avg_dist, interpolation='nearest', cmap=plt.cm.gnuplot2,
#                vmin=0)
#     plt.xticks(range(n_clusters), labels, rotation=45)
#     plt.yticks(range(n_clusters), labels)
#     plt.colorbar()
#     plt.suptitle("Interclass %s distances" % metric, size=18)
#     plt.tight_layout()


# Plot clustering results

# for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
metric = "cityblock"
model = AgglomerativeClustering(n_clusters=n_clusters,
                                linkage="average", affinity=metric)
model.fit(X)
# plt.figure()
# plt.axes([0, 0, 1, 1])
# print("model child: {}".format(type(model.children_[0][0])))
# dendrogram(model.children_)  

ii = itertools.count(X.shape[0])
my_h_tree = [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in model.children_]
dict_tree = {}
print(my_h_tree)
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
# print(root)
# sys.exit(0)


# for l, c in zip(np.arange(model.n_clusters), 'rgbk'):
#     print("model label : {}".format(model.labels_))
#     plt.plot(X[model.labels_ == l].T, c=c, alpha=.5)
# plt.axis('tight')
# plt.axis('off')
# plt.suptitle("AgglomerativeClustering(affinity=%s)" % metric, size=20)


# plt.show()
##### SON OF ANNE ZONE #####


def psa():
    # C = AGGLOMERATIVECLUSTERING(T)
    # c = the root leaf of the tree C

    Z,r = psa_average(root)
    
    return Z

def psa_average(c):
    if c < number_of_data:
        return [lookupData[c], 1]
    else:
        [z1, r1] = psa_average(dict_tree[c][0])
        [z2, r2] = psa_average(dict_tree[c][1])
        z = average_ts_weight(z1, z2, r1, r2)
        return [z, r1+r2]
result = psa()
for a in lookupData:
    plt.plot(a[1:])
plt.plot(result,'black')
plt.show()

