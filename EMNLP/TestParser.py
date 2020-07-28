import pickle

import numpy as np


def test_parser(file_path, weights, myfeature, words_dict, pos_dict, trans, check_pos):
    paras = []
    print("Reading and parsing file")
    with open(file_path, encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if len(line.strip("\n")) == 0:
                continue
            if line[0] == "#":
                paras.append([line])
                continue
            paras[-1].append(line.strip("\n").split("\t"))

    print("Paragraph number: " + str(len(paras)))
    file_output = open("result_labeled.sdp", encoding="utf-8", mode="w")

    # main parsing process
    new_para = []
    for num, para in enumerate(paras):
        if num % 100 == 0:
            print(num)
        new_para.append([para[0]])
        words = [("root", "root")]
        stack_stat = [0]
        queue_stat = []
        arcs = {}
        arc_lable = {}
        for word in para[1:]:
            new_para[-1].append(word)
            word_actual = word[1]
            words.append((word_actual, word[3]))
            queue_stat.append(int(word[0]))
        trans_history = []
        stage = 0
        # do the transitions
        while len(queue_stat) > 0:
            next_tran = np.zeros(len(trans))
            word_and_pos = []
            # find features
            try:
                word_and_pos.append((check_pos["s_t"], words[int(stack_stat[-1])]))
            except:
                pass
            try:
                word_and_pos.append((check_pos["q_t"], words[int(queue_stat[0])]))
            except:
                pass
            try:
                word_and_pos.append((check_pos["s_n"], words[int(stack_stat[-2])]))
            except:
                pass
            try:
                word_and_pos.append((check_pos["q_n"], words[int(queue_stat[1])]))
            except:
                pass
            try:
                word_and_pos.append((check_pos["s_b"], words[int(stack_stat[-1]) - 1]))
            except:
                pass
            try:
                word_and_pos.append((check_pos["s_a"], words[int(stack_stat[-1]) + 1]))
            except:
                pass
            try:
                word_and_pos.append((check_pos["q_b"], words[int(queue_stat[0]) - 1]))
            except:
                pass
            try:
                word_and_pos.append((check_pos["q_a"], words[int(queue_stat[0]) + 1]))
            except:
                pass
            possible_feats = []
            for where, (word, pos) in word_and_pos:
                try:
                    w_n = words_dict[word]
                    possible_feats.append((where, 0, w_n))
                    p_n = pos_dict[pos]
                    possible_feats.append((where, 1, p_n))
                except:
                    pass
            try:
                feature = (check_pos["op-1"], 2, trans_history[stage - 1])
                possible_feats.append(feature)
            except:
                pass
            try:
                feature = (check_pos["op-2"], 2, trans_history[stage - 2])
                possible_feats.append(feature)
            except:
                pass
            successful_feature = []
            for feature in possible_feats:
                try:
                    for lable, number in myfeature[feature]:
                        next_tran[lable] += weights[number]
                    successful_feature.append(feature)
                except:
                    pass
            for f1 in range(len(successful_feature)):
                for f2 in range(f1 + 1, len(successful_feature)):
                    if (
                        successful_feature[f1][1] == 0
                        and successful_feature[f2][1] == 1
                    ):
                        continue
                    feature = (successful_feature[f1], successful_feature[f2])
                    try:
                        for lable, number in myfeature[feature]:
                            next_tran[lable] += weights[number]
                    except:
                        pass
            # decide the best move
            next_tran = np.argmax(next_tran)
            tran_name = trans[next_tran]
            # judge whether it's legal
            if len(trans_history) >= 1:
                last_tran = trans[trans_history[-1]]
                if (last_tran == "sw" and tran_name == "sw") or (
                    (last_tran[0] == "l" or last_tran[0] == "r")
                    and (tran_name[0] == "l" or tran_name[0] == "r")
                ):
                    next_tran = 1
                    tran_name = "sh"
            if len(stack_stat) == 0 and tran_name == "d":
                next_tran = 1
                tran_name = "sh"
            if tran_name[0] == "l" or tran_name[0] == "r":
                try:
                    if (queue_stat[0], stack_stat[-1]) in arc_lable.keys() or (
                        stack_stat[-1],
                        queue_stat[0],
                    ) in arc_lable.keys():
                        next_tran = 1
                        tran_name = "sh"
                except:
                    pass
            stage += 1
            try:
                if tran_name == "sh":
                    stack_stat.append(queue_stat[0])
                    del queue_stat[0]
                elif tran_name == "d":
                    stack_stat.pop()
                elif tran_name == "sw":
                    temp = stack_stat[-1]
                    stack_stat[-1] = stack_stat[-2]
                    stack_stat[-2] = temp
                elif tran_name[0] == "l":
                    if len(tran_name) == 1:
                        tran_name = tran_name + "ARG"
                    try:
                        arcs[queue_stat[0]].append(stack_stat[-1])
                    except:
                        arcs[queue_stat[0]] = [stack_stat[-1]]
                    finally:
                        arc_lable[(queue_stat[0], stack_stat[-1])] = tran_name[1:]
                else:
                    if len(tran_name) == 1:
                        tran_name = tran_name + "ARG"
                    try:
                        arcs[stack_stat[-1]].append(queue_stat[0])
                    except:
                        arcs[stack_stat[-1]] = [queue_stat[0]]
                    finally:
                        arc_lable[(stack_stat[-1], queue_stat[0])] = tran_name[1:]
            except:
                stack_stat.append(queue_stat[0])
                del queue_stat[0]
                next_tran = 1
            trans_history.append(next_tran)
        arcs = list(sorted(arcs.items(), key=lambda x: x[0]))
        froms = len(arcs)
        has_root = 0
        try:
            if arcs[0][0] == 0:
                has_root = 1
        except:
            pass
        froms -= has_root
        extension = "-|" * 2 + "_|" * froms
        extension = extension.strip("|").split("|")
        for each_word in range(1, len(new_para[-1])):
            new_para[-1][each_word].extend(extension)
        for num, (f, ts) in enumerate(arcs):
            if f == 0:
                new_para[-1][ts[-1]][4] = "+"
            else:
                new_para[-1][f][5] = "+"
                for t in ts:
                    new_para[-1][t][6 + num - has_root] = arc_lable[(f, t)]
        file_output.write(new_para[-1][0])
        for line in new_para[-1][1:]:
            file_output.write("\t".join(line) + "\n")
        file_output.write("\n")
    file_output.close()


if __name__ == "__main__":
    use_label = True
    with open("save_dictions_label:{}.txt".format(use_label), "rb") as savefile:
        words_dict = pickle.load(savefile)
        POSdict = pickle.load(savefile)
        check_pos = pickle.load(savefile)
        trans_names = pickle.load(savefile)
        myfeature = pickle.load(savefile)
    with open("save_model_label:{}.txt".format(use_label), "rb") as savefile:
        weights = pickle.load(savefile)
    print(trans_names)
    print(len(trans_names))
    test_parser(
        "esl.input",
        words_dict=words_dict,
        pos_dict=POSdict,
        check_pos=check_pos,
        trans=trans_names,
        weights=weights,
        myfeature=myfeature,
    )
