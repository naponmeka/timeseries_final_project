from averageSeries import * 
from math import *
from scipy.optimize import basinhopping
import sys
import time
import matplotlib.pyplot as plt
from computeAccuracy import *

sys.setrecursionlimit(1500)
start = time.time()
train_filename = 'OliveOil_ALL'
test_filename = 'Beef_TEST'
f = open('ClassificationClusteringDatasets/' + train_filename)
data_train = []
train_lower_b = []
train_upper_b = []
window_size = 10
r = 5
data_train_dict = {}
for line in f:
    floats = [float(x) for x in line.strip().split()]
    ts = floats[1:]
    ts = normalizeSeries(ts)
    if floats[0] in data_train_dict:
        data_train_dict[floats[0]].append(ts)
    else:
        data_train_dict[floats[0]] = [ts]
    # data_train.append(floats)
    train_upper_b.append(upper_keogh(ts))
    train_lower_b.append(lower_keogh(ts))
f.close()

# data_test = []
# f = open('ClassificationClusteringDatasets/' + test_filename)
# for line in f:
#     floats = [float(x) for x in line.strip().split(' ')]
#     data_test.append(floats)
# f.close()

def pyramid(serie):
    out=[]
    for idx,value in enumerate(serie):
        if idx==len(serie)-1: break
        else : out.append(average_ts(value,serie[idx+1]))
    if len(out)==1: return out
    else :
        print(len(out))
        return pyramid(out)

def pisa(serie,iteration):
    out=[]
    if iteration%2==0: serie.append(serie[0])
    else:
        serie = [serie[len(serie)-1]] + serie
    iteration-=1
    for idx,value in enumerate(serie):
        if idx==len(serie)-1: break
        else : out.append(average_ts(value,serie[idx+1]))
    if iteration==0: return out
    else:
        # for r in out:
        #     plt.plot(r)
        # plt.show()

        return pisa(list(out),iteration)

meanDistances = []
weights_to_mean = []
for key, one_class_data in data_train_dict.items():
    mean = pyramid(one_class_data)[0]
    distance = avgMeanErrorEuclideanDistance(mean, one_class_data)
    meanDistances.append(distance)
    weights_to_mean.append(len(one_class_data))

the_mean = 0
sum_weight = 0
for idx,m in enumerate(meanDistances):
    the_mean += m * weights_to_mean[idx]
    sum_weight += weights_to_mean[idx]
the_mean = the_mean/sum_weight
print(the_mean)


sys.exit(0)
##########

class_a = []
counting = 0
first_label = data_train[0][0]
for d in data_train:
    if first_label == d[0]:
        class_a.append(d[1:])
        # plt.plot(d[1:])
        counting += 1
size = 50
first = class_a[1]
first = uniScaling(first , size)
second = class_a[2]
second = uniScaling(second , size)
third = class_a[3]
third = uniScaling(third,size)
# result = average_ts(first,second)
# result = average_ts(result,second)
result2 = average_n_ts([first,second,third],[1,0,0])

# result = pyramid(class_a)
# result = pisa([first,first,second],50)

# print(result)
# plt.plot(average_ts(first,second), 'purple')
plt.plot(first)
plt.plot(second)
plt.plot(third)

# plt.plot(result2,'red')
plt.plot(result2,'black')
plt.show()
sys.exit(0)

        

memPath=[]
# avgSerie = average_ts(class_a[2],class_a[1])
print("class_a size: {}".format(len(class_a)))
# l1 = range(0,10)
# l2 = range(0,10)
# print(xxx)
# result = pyramid(class_a)
result = pisa(class_a,20)
# result = pyramid(result)
# plt.plot(result[0])
for r in result:
    plt.plot(r)
plt.show()
