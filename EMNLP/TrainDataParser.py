import pickle
from math import *


def data_parser(file_path, use_lable):
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
                paras.append([])
                continue
            paras[-1].append(line.strip("\n").split("\t"))

    print("Paragraph number: " + str(len(paras)))
    # Part-Of-Speech
    POSset = []
    # All links' kinds
    LinkLable = []

    # links in sentences
    links_in_paras = []
    plain_links = []
    words_in_paras = []

    # Word importance
    word_importance = {}
    top_importance = 10
    predicate_importance = 5
    noraml_link = 3
    show_up = 1

    # lemma (unused)
    lemma_for_words = {}

    total_arcs = 0
    for para in paras:
        links_in_paras.append({})
        plain_link = {}
        words_in_paras.append([("root", "root")])
        for word in para:
            word_id = int(word[0])
            word_actual = word[1]
            try:
                word_importance[word_actual] += show_up
            except:
                word_importance[word_actual] = show_up
            words_in_paras[-1].append((word_actual, word[3]))
            lemma = word[2]
            if word_actual not in lemma_for_words.keys():
                lemma_for_words[word_actual] = [lemma]
            else:
                lemma_for_words[word_actual].append(lemma)
            POSset.append(word[3])
            # top
            if word[4] == "+":
                total_arcs += 1
                links_in_paras[-1][0] = {word_id: "root"}
                plain_link[0] = [word_id]
                if word_actual not in word_importance.keys():
                    word_importance[word_actual] = top_importance
                else:
                    word_importance[word_actual] += top_importance
            # from
            if word[5] == "+":
                links_in_paras[-1][word_id] = {}
                if word_actual not in word_importance.keys():
                    word_importance[word_actual] = predicate_importance
                else:
                    word_importance[word_actual] += predicate_importance
        plain_links.append(plain_link)

    # Mark down the links
    for para_num, para in enumerate(paras):
        predicates = list(links_in_paras[para_num].keys())
        predicates = sorted(predicates)
        has_root = 0
        if len(predicates) == 0:
            continue
        if predicates[0] == 0:
            has_root = 1
        for word in para:
            wid = int(word[0])
            total_inlinks = 0
            for i in range(6, len(word)):
                if word[i] != "_":
                    total_arcs += 1
                    total_inlinks += 1
                    LinkLable.append(i)
                    # has a virtual root
                    this_predicate = predicates[i - 6 + has_root]
                    links_in_paras[para_num][this_predicate][wid] = word[i]
                    if this_predicate < wid:
                        try:
                            plain_links[para_num][this_predicate].append(wid)
                        except:
                            plain_links[para_num][this_predicate] = [wid]
                    else:
                        try:
                            plain_links[para_num][wid].append(this_predicate)
                        except:
                            plain_links[para_num][wid] = [this_predicate]

            word_actual = word[1]
            if total_inlinks > 0:
                if word_actual not in word_importance.keys():
                    word_importance[word_actual] = noraml_link * total_inlinks
                else:
                    word_importance[word_actual] += noraml_link * total_inlinks

    POSset.append("root")
    POSset = set(POSset)
    POSdict = {}
    for id, i in enumerate(POSset):
        POSdict[i] = id

    all_words = list(word_importance.keys())
    all_words_dict = {}
    for id, word in enumerate(all_words):
        all_words_dict[word] = id
    print("There are in total {} words".format(len(all_words)))

    # Judge gold transitions
    ops = {"swap": 0, "shift": 0, "reduce": 0, "larc": 0, "rarc": 0}
    Transformations = {"sw": 0, "sh": 0, "d": 0}
    Transforms_in_para = []
    Status_in_para = []
    for para_num, para in enumerate(words_in_paras):
        trans = []
        status = {"S": [], "Q": []}
        word_num = len(para)
        links = links_in_paras[para_num]
        plain_link = plain_links[para_num]
        for i in plain_link.keys():
            plain_link[i] = sorted(plain_link[i])
        stack = [0]
        queue = []
        for i in range(1, word_num):
            queue.append(i)
        last_arc = False
        last_swap = False
        while len(queue) > 0:
            status["S"].append("|".join(list(map(lambda x: str(x), stack))))
            status["Q"].append("|".join(list(map(lambda x: str(x), queue))))
            # shift
            if len(stack) == 0:
                trans.append("sh")
                ops["shift"] += 1
                Transformations["sh"] += 1
                stack.append(queue[0])
                del queue[0]
                last_arc = False
                last_swap = False
                continue
            word1, word2 = stack[-1], queue[0]
            # Reduce
            if (
                word1 not in plain_link.keys()
                or len(plain_link[word1]) == 0
                or plain_link[word1][-1] < word2
                or (plain_link[word1][-1] == word2 and last_arc)
            ):
                stack.pop()
                trans.append("d")
                Transformations["d"] += 1
                ops["reduce"] += 1
                last_arc = False
                last_swap = False
                continue
            # Arc
            if last_arc == False and word2 in plain_link[word1]:
                last_arc = True
                try:
                    link_lable = links[word1][word2]
                    ops["rarc"] += 1
                    try:
                        Transformations["r" + link_lable] += 1
                    except:
                        Transformations["r" + link_lable] = 1
                    trans.append("r" + link_lable)
                    continue
                except:
                    try:
                        link_lable = links[word2][word1]
                        ops["larc"] += 1
                        try:
                            Transformations["l" + link_lable] += 1
                        except:
                            Transformations["l" + link_lable] = 1
                        trans.append("l" + link_lable)
                        continue
                    except:
                        print("Something's wrong")
            last_arc = False
            # Swap
            if last_swap == False and len(stack) > 1:
                word0 = stack[-2]
                for i in range(len(plain_link[word0])):
                    if plain_link[word0][i] >= word2:
                        plain_link[word0] = plain_link[word0][i:]
                        break
                for i in range(len(plain_link[word1])):
                    if plain_link[word1][i] > word2:
                        plain_link[word1] = plain_link[word1][i:]
                        break
                if plain_link[word0] < plain_link[word1]:
                    last_swap = True
                    temp = stack[-1]
                    stack[-1] = stack[-2]
                    stack[-2] = temp
                    trans.append("sw")
                    ops["swap"] += 1
                    Transformations["sw"] += 1
                    continue
            # Shift
            last_swap = False
            ops["shift"] += 1
            trans.append("sh")
            Transformations["sh"] += 1
            stack.append(queue[0])
            del queue[0]
        Transforms_in_para.append(trans)
        status["S"].append(list(map(lambda x: str(x), stack)))
        status["Q"].append([])
        Status_in_para.append(status)
    print(ops)
    print(
        "There are in total {} arcs, and the model generates {} arcs, {}%".format(
            total_arcs,
            ops["larc"] + ops["rarc"],
            (ops["larc"] + ops["rarc"]) * 100 / total_arcs,
        )
    )
    print("Generating status vectors and operation vectors")
    check_pos = {
        "s_t": 0,
        "q_t": 1,
        "s_n": 2,
        "q_n": 3,
        "s_b": 4,
        "s_a": 5,
        "q_b": 6,
        "q_a": 7,
        "op-1": 8,
        "op-2": 9,
    }

    # feature's set
    myfeature = {}
    para_number = len(Transforms_in_para)
    Trans_selected = {}
    sel_count = 0
    # only transitions appearing more than this number will be taken in
    sel_shrehold = 100
    trans_numbers = []
    # judge how frequent features should be
    feature_shrehold = 0.5
    trans_names = []
    for key, val in Transformations.items():
        if val < sel_shrehold:
            continue
        trans_numbers.append(val)
        Trans_selected[key] = sel_count
        trans_names.append(key)
        sel_count += 1
    if not use_lable:
        Trans_selected = {"sw": 0, "sh": 1, "d": 2, "l": 3, "r": 4}
        trans_names = list(Trans_selected.keys())
        sel_count = 5
    print("Transition number: {}, which are {}".format(sel_count, Trans_selected))
    data_classified = []

    # Get the stack and queue status of each stage, then produce features
    for i in range(sel_count):
        data_classified.append([])
    for para in range(para_number):
        if para % 1000 == 0:
            print("{}/{}".format(para, para_number))
            del_keys = []
            for key, show_up_count in myfeature.items():
                lable_kind = key[-1]
                if show_up_count < log(trans_numbers[lable_kind]) * feature_shrehold:
                    del_keys.append(key)
            print(
                "Now key number is {}, and {} will be deleted".format(
                    len(myfeature.keys()), len(del_keys)
                )
            )
            for key in del_keys:
                del myfeature[key]
        trans = Transforms_in_para[para]
        stat = Status_in_para[para]
        words = words_in_paras[para]
        stage_num = len(trans)
        for stage in range(stage_num):
            gold_trans = trans[stage]
            if not use_lable:
                if gold_trans[0] == "s":
                    gold_trans = gold_trans[0:2]
                else:
                    gold_trans = gold_trans[0]
            trans[stage] = gold_trans
            try:
                lable_num = Trans_selected[gold_trans]
                stack_stat = stat["S"][stage].split("|")
                queue_stat = stat["Q"][stage].split("|")
                word_and_pos = []
                successful_feature = []
                # Try to get all feature positions
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
                    word_and_pos.append(
                        (check_pos["s_b"], words[int(stack_stat[-1]) - 1])
                    )
                except:
                    pass
                try:
                    word_and_pos.append(
                        (check_pos["s_a"], words[int(stack_stat[-1]) + 1])
                    )
                except:
                    pass
                try:
                    word_and_pos.append(
                        (check_pos["q_b"], words[int(queue_stat[0]) - 1])
                    )
                except:
                    pass
                try:
                    word_and_pos.append(
                        (check_pos["q_a"], words[int(queue_stat[0]) + 1])
                    )
                except:
                    pass
                word_pos_rec = []
                for where, (word, pos) in word_and_pos:
                    w_n = all_words_dict[word]
                    feature = (where, 0, w_n, lable_num)
                    successful_feature.append(feature)
                    try:
                        myfeature[feature] += 1
                    except:
                        myfeature[feature] = 1
                    p_n = POSdict[pos]
                    feature = (where, 1, p_n, lable_num)
                    successful_feature.append(feature)
                    try:
                        myfeature[feature] += 1
                    except:
                        myfeature[feature] = 1
                    word_pos_rec.append((where, w_n, p_n))
                trans_history = []
                try:
                    feature = (
                        check_pos["op-1"],
                        2,
                        Trans_selected[trans[stage - 1]],
                        lable_num,
                    )
                    successful_feature.append(feature)
                    trans_history.append(
                        (check_pos["op-1"], Trans_selected[trans[stage - 1]])
                    )
                    try:
                        myfeature[feature] += 1
                    except:
                        myfeature[feature] = 1
                except:
                    pass
                try:
                    feature = (
                        check_pos["op-2"],
                        2,
                        Trans_selected[trans[stage - 2]],
                        lable_num,
                    )
                    successful_feature.append(feature)
                    trans_history.append(
                        (check_pos["op-1"], Trans_selected[trans[stage - 2]])
                    )
                    try:
                        myfeature[feature] += 1
                    except:
                        myfeature[feature] = 1
                except:
                    pass
                # Crossing features are also important
                data_classified[lable_num].append(
                    {"wp": word_pos_rec, "th": trans_history}
                )
                for f1 in range(len(successful_feature)):
                    for f2 in range(f1 + 1, len(successful_feature)):
                        if (
                            successful_feature[f1][1] == 0
                            and successful_feature[f2][1] == 1
                        ):
                            continue
                        feature = (
                            successful_feature[f1][0:3],
                            successful_feature[f2][0:3],
                            lable_num,
                        )
                        try:
                            myfeature[feature] += 1
                        except:
                            myfeature[feature] = 1
            except:
                pass

    feat_count = 0
    weights = []
    lable_dict = {}
    for key, val in myfeature.items():
        lable_kind = key[-1]
        if val < log(trans_numbers[lable_kind]) * feature_shrehold:
            continue
        if len(key) == 3:
            try:
                lable_dict[(key[0][0:3], key[1][0:3])].append((key[2], feat_count))
            except:
                lable_dict[(key[0][0:3], key[1][0:3])] = [(key[2], feat_count)]
        else:
            try:
                lable_dict[key[0:3]].append((key[3], feat_count))
            except:
                lable_dict[key[0:3]] = [(key[3], feat_count)]
        weights.append(0)
        feat_count += 1
    print(feat_count)
    print("Generating training data...")
    data_featured_all = []
    cross_sum = 0
    for i in range(sel_count):
        data_featured_all.append([])
        for num, record in enumerate(data_classified[i]):
            if num > 200000:
                break
            data_featured_all[i].append({})
            feat_succ = []
            for where, word, pos in record["wp"]:
                try:
                    for j, id in lable_dict[(where, 0, word)]:
                        data_featured_all[i][num][id] = j
                    feat_succ.append((where, 0, word))
                except:
                    pass
                try:
                    for j, id in lable_dict[(where, 1, pos)]:
                        data_featured_all[i][num][id] = j
                    feat_succ.append((where, 1, pos))
                except:
                    pass
            for where, trans in record["th"]:
                try:
                    for j, id in lable_dict[(where, 2, trans)]:
                        data_featured_all[i][num][id] = j
                    feat_succ.append((where, 2, trans))
                except:
                    pass
            for feat1 in range(len(feat_succ)):
                for feat2 in range(feat1 + 1, len(feat_succ)):
                    try:
                        for j, id in lable_dict[(feat_succ[feat1], feat_succ[feat2])]:
                            data_featured_all[i][num][id] = j
                            cross_sum += 1
                    except:
                        pass

    print(cross_sum)
    test_gap = 5
    test_data = []
    data_featured = []
    for i in range(sel_count):
        test_data.append([])
        data_featured.append([])
        for j in range(len(data_featured_all[i])):
            if j % test_gap == 1:
                test_data[-1].append(data_featured_all[i][j])
            else:
                data_featured[-1].append(data_featured_all[i][j])
    print("Training data generated")
    # Saving data for the optimizer
    print("Saving featured results")
    # with open('save_feature_results_label:{}.txt'.format(use_lable), 'wb') as savefile:
    #     pickle.dump(data_featured, savefile)
    #     pickle.dump(test_data, savefile)
    #     pickle.dump(weights, savefile)
    #     pickle.dump(lable_dict, savefile)
    #     pickle.dump(sel_count, savefile)
    # Saving data for test parser, this must be executed
    with open("save_dictions_label:{}.txt".format(use_lable), "wb") as savefile:
        pickle.dump(all_words_dict, savefile)
        pickle.dump(POSdict, savefile)
        pickle.dump(check_pos, savefile)
        pickle.dump(trans_names, savefile)
        pickle.dump(lable_dict, savefile)
    print("done")
    return data_featured, test_data, weights, sel_count


if __name__ == "__main__":
    data_parser("result_labeled.sdp", True)
    data_parser("dm.sdp", False)
