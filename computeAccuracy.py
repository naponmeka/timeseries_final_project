from averageSeries import *
import math
import statistics
# test_list = []
# for i in range(10):
#     test_list.append(range(10))

# the_mean_seq = range(1,11)


def avgMeanErrorDistance(mean_seq, list_of_sequence):
    sum_distance = 0
    for seq in list_of_sequence:
        sum_distance += DTWDistance(mean_seq, seq, 30)
    return sum_distance/len(list_of_sequence)

def euclideanDistance(seq1, seq2):
    seq2 = uniScaling(seq2,len(seq1))
    distance = 0
    for idx,value in enumerate(seq1):
        distance += (seq1[idx] - seq2[idx]) ** 2
    distance = sqrt(distance)
    print("DISTANCE: {}".format(distance))
    return distance

# l1 = [1,3,2,3,4,5]
# l2 = [3,1,2,3,4,5]

# print(euclideanDistance(l1, l2))

def avgMeanErrorEuclideanDistance(mean_seq, list_of_sequence):
    sum_distance = 0
    distances = []
    for seq in list_of_sequence:
        distances.append(euclideanDistance(mean_seq, seq))
        sum_distance += euclideanDistance(mean_seq, seq)
    print("-----ANOVA-----")
    print(distances)
    print("MEAN: {}".format(statistics.mean(distances)))
    print("SD: {}".format(statistics.stdev(distances)))
    return sum_distance/len(list_of_sequence)
# print(avgMeanErrorDistance(the_mean_seq,test_list))