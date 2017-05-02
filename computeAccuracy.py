from averageSeries import *

# test_list = []
# for i in range(10):
#     test_list.append(range(10))

# the_mean_seq = range(1,11)


def avgMeanErrorDistance(mean_seq, list_of_sequence):
    sum_distance = 0
    for seq in list_of_sequence:
        sum_distance += DTWDistance(mean_seq, seq, 30)
    return sum_distance/len(list_of_sequence)

# print(avgMeanErrorDistance(the_mean_seq,test_list))