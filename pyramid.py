from averageSeries import * 
from math import *
from scipy.optimize import basinhopping
import sys
import time
import matplotlib.pyplot as plt


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
