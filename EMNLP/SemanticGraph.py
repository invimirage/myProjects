import pickle
import time
from math import *

from TestParser import test_parser
from TrainDataParser import data_parser


# build a batch for training, size denotes the size of data set under each tag
# length is the total length of a tag's data set
def build_batch(data, size, length, indexes, sel_count):
    batch = []
    for i in range(sel_count):
        batch.append([])
        ind = indexes[i]
        if ind + size > length[i]:
            batch[i] = data[i][ind : length[i]]
            batch[i].extend(data[i][0 : ind + size - length[i]])
            indexes[i] = ind + size - length[i]
        else:
            batch[i] = data[i][ind : ind + size]
            indexes[i] = ind + size
    return batch


# calculate the LL for a batch
alpha = 0.5


def logloss(batch, weights, sel_count):
    sum1 = 0
    sum2 = 0
    total_sum = []
    for i in range(sel_count):
        total_sum.append([])
        for text in batch[i]:
            logsum = 0
            for j in range(sel_count):
                expsum = 0
                for feat_num, kind in text.items():
                    if not kind == j:
                        continue
                    if i == kind:
                        sum1 += weights[feat_num]
                    expsum += weights[feat_num]
                logsum += exp(expsum)
            sum2 += log(logsum)
            total_sum[i].append(logsum)
    sum3 = sum([pow(weights[i], 2) for i in range(len(weights))]) * (alpha / 2)
    return sum1 - sum2 - sum3, total_sum


# caluculate the gradient
def gradient(batch, total_sum, weights, sel_count):
    sum1 = []
    sum2 = []
    for i in range(len(weights)):
        sum1.append(0)
        sum2.append(0)
    for i in range(sel_count):
        for num, text in enumerate(batch[i]):
            for j in range(sel_count):
                expsum = 0
                index_i = []
                for feat_num, kind in text.items():
                    if not kind == j:
                        continue
                    if i == kind:
                        sum1[feat_num] += 1
                    index_i.append(feat_num)
                    expsum += weights[feat_num]
                for ind in index_i:
                    sum2[ind] += exp(expsum) / total_sum[i][num]
    res = [sum1[i] - sum2[i] - alpha * weights[i] for i in range(len(weights))]
    return res


def my_optimizer(use_lable, train_path, test_path):
    clock_begin = time.perf_counter()
    learning_rate = 0.002
    if use_lable:
        batch_size = 100
    else:
        batch_size = 500
    iternum = 1000
    # Use the stored results from train data parser
    # with open('save_feature_results_label:{}.txt'.format(use_lable), 'rb') as savedfile:
    #     data_featured = pickle.load(savedfile)
    #     test_data = pickle.load(savedfile)
    #     weights = pickle.load(savedfile)
    #     sel_count = pickle.load(savedfile)
    # Use train data parser directly
    data_featured, test_data, weights, sel_count = data_parser(train_path, use_lable)
    clock_dataparse = time.perf_counter()
    indexes = []
    length = []
    for i in range(sel_count):
        indexes.append(0)
        length.append(len(data_featured[i]))
    for i in range(iternum):
        batch = build_batch(data_featured, batch_size, length, indexes, sel_count)
        ll, total_sum = logloss(batch, weights, sel_count)
        grad = gradient(batch, total_sum, weights, sel_count)
        if i % 50 == 0:
            print("-" * 20)
            print("step " + str(i))
            classify(test_data, weights, sel_count)
        for j in range(len(grad)):
            weights[j] += grad[j] * learning_rate
        # Save coefficients for test data
        if i % 200 == 0:
            with open("save_model_label:{}.txt".format(use_lable), "wb") as savefile:
                pickle.dump(weights, savefile)
    clock_end = time.perf_counter()
    print("-" * 30)
    print("Making result")
    print("Training:")
    classify(data_featured, weights, sel_count)
    print("Testing:")
    classify(test_data, weights, sel_count)
    print("Time spent:")
    print("Data parsing: " + str(clock_dataparse - clock_begin) + " seconds")
    print("Iterations: " + str(clock_end - clock_dataparse) + " seconds")
    print(
        "Every iteration on average: "
        + str((clock_end - clock_dataparse) / iternum)
        + " seconds"
    )
    # Use test data parser directly
    print("Generating output")
    # This is generated from train data parser, the file isn't large
    with open("save_dictions_label:{}.txt".format(use_lable), "rb") as savefile:
        words_dict = pickle.load(savefile)
        POSdict = pickle.load(savefile)
        check_pos = pickle.load(savefile)
        trans_names = pickle.load(savefile)
        myfeature = pickle.load(savefile)
    test_parser(
        test_path,
        words_dict=words_dict,
        pos_dict=POSdict,
        check_pos=check_pos,
        trans=trans_names,
        weights=weights,
        myfeature=myfeature,
    )


# data: featured data, containing dicts for each text
# dict has format key: feat_number value: kind
def classify(data, weights, sel_count):
    correct_items = 0
    total_items = 0
    total_kinds = []
    predict_kinds = []
    correct_kinds = []
    F1s = []
    for i in range(sel_count):
        predict_kinds.append(0)
        total_kinds.append(0)
        F1s.append(0)
        correct_kinds.append(0)
    for i in range(sel_count):
        for text in data[i]:
            total_items += 1
            total_kinds[i] += 1
            kind_score = []
            for j in range(sel_count):
                kind_score.append(0)
            for feat_num, kind in text.items():
                kind_score[kind] += weights[feat_num]
            predict_kinds[kind_score.index(max(kind_score))] += 1
            if i == kind_score.index(max(kind_score)):
                correct_items += 1
                correct_kinds[i] += 1
    print("Accuracy: " + str(correct_items / total_items))
    return correct_items / total_items, sum(F1s) / sel_count


if __name__ == "__main__":
    my_optimizer(True, train_path="dm.sdp", test_path="esl.input")
