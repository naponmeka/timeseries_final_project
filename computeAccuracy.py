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
        distance = (DTWDistance(mean_seq, seq, 1000))**2
        distances.append(distance)
        sum_distance += distance
    print("-----ANOVA-----")
    print(distances)
    print("MEAN: {}".format(statistics.mean(distances)))
    print("SD: {}".format(statistics.stdev(distances)))
    return sum_distance/len(list_of_sequence)

def avgMeanErrorEuclideanDistanceTuple(mean_seq_tuple, list_of_sequence_tuple):
    mean_seq = mean_seq_tuple['serie']
    list_of_sequence = list_of_sequence_tuple
    # for idx, item in list_of_sequence_tuple.items():
    #     list_of_sequence.append(item['serie'])
    
    
    sum_distance = 0
    distances = []
    for seq in list_of_sequence:
        distance = (DTWDistance(mean_seq, seq, 1000))**2
        distances.append(distance)
        sum_distance += distance
    print("-----ANOVA-----")
    print(distances)
    print("MEAN: {}".format(statistics.mean(distances)))
    print("SD: {}".format(statistics.stdev(distances)))
    return sum_distance/len(list_of_sequence)
