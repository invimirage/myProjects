# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 20:44:26 2019

@author: haozhjiang
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import actoolkit.lgtplus as lgtplus
from procedural_update_20190827 import settings
from matplotlib import pyplot as plt
import numpy as np

def model(col, fname, maxrank=4):
    ## Load Data
    adprofile = pd.read_csv(fname)
    # 原始场景是ctr的，因为文件中ctr太多就没有改
    adprofile.rename(columns={col: 'CTR'}, inplace=True)
    adprofile.rename(columns={settings.cols[-1]: 'weightcount'})
    # adprofile.drop(["xdatetime", "age", "ds", "importds", "updatedtime", "id"], axis=1, inplace=True)
    print(len(list(set(adprofile['adid']))))
    print(len(adprofile))

    ## Specify Continuous Features
    continuous_feats = settings.parse_feats(settings.continuous_feats)

    # Specify Features
    features = settings.parse_feats(settings.ad_feats)

    # New ID to original feature value
    relations = {}

    ## Dropping all-NULL Columns
    adprofile.dropna(axis=1, how="all", inplace=True)

    # Identifying categorical features
    cat_cols = [col for col in features if col not in continuous_feats]

    # Re-ID feature categories from 0
    for col in cat_cols:
        col_id = col.strip(' \n') + '_ID'
        adprofile[col_id] = adprofile.groupby([col]).grouper.group_info[0]
        relation = adprofile[[col_id, col]].drop_duplicates()
        relations[col_id] = {}
        relations[col_id] = dict(relation)
        print(col, max(adprofile[col_id]))
    adprofile.drop(cat_cols, axis=1, inplace=True)

    ## Specify Features to use
    x_feature_list = list(relations.keys()) + continuous_feats

    ## Creating Test/Training Set
    x_train, x_test, y_train, y_test = train_test_split(adprofile.drop('CTR', axis=1)[x_feature_list], adprofile['CTR'],
                                                        test_size=0.2, random_state=1)

    ## Train LGBMRanker

    lgbr_multi = LGBMRegressor(objective="regression",
                               learning_rate=0.148,
                               num_leaves=40,
                               num_trees=20,
                               max_depth=10,
                               n_jobs=8,
                               importance_type='gain')

    lgbr_multi.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
                   categorical_feature=list(relations.keys()))
    feature_names = lgbr_multi._Booster.dump_model()["feature_names"]
    feature_importance = dict(zip(feature_names, lgbr_multi.feature_importances_))
    feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print(feature_importance)
    feat_imp_dict = dict(feature_importance)
    feature_importance = pd.DataFrame(feature_importance)
    feature_importance.columns = ['feature', 'importance']
    feature_importance.to_csv('feat_imp_{}.csv'.format(col), index=False)

    ## Getting All Paths
    adprofile_allwx_all_path = lgtplus.get_all_paths(lgbr_multi)

    ## Getting High Frequence Patterns
    feature_names = lgbr_multi._Booster.dump_model()["feature_names"]
    frequent_paths = []
    for rank in range(2, maxrank + 1):
        frequent_paths.append(lgtplus.get_hi_freq_pattern(adprofile_allwx_all_path, rank, 50, True, feature_names))
        print(frequent_paths[-1])

    frequent_crosses = pd.concat([pd.DataFrame(fq) for fq in frequent_paths])
    frequent_crosses.columns = ['cross_name', 'times']
    frequent_crosses.to_csv('frequent_crosses_{}.csv'.format(col), index=False)

    # Score the feature combinations of rank 2, 3 and 4
    feature_score = []
    for rank in range(2, maxrank + 1):
        feature_score.append({})
        for features, times in frequent_paths[rank - 2]:
            feats = list(sorted([feature for feature in features]))
            feats_imp = sum(feat_imp_dict[feat] for feat in feats) / rank
            feat_key = ','.join(settings.parse_key(feat) for feat in feats)
            try:
                feature_score[-1][feat_key][0] += times
                feature_score[-1][feat_key][1] += feats_imp
            except KeyError:
                feature_score[-1][feat_key] = [times, feats_imp]
        print(feature_score[-1])

    # ## Getting Feature Combination Bias
    # L2_ft_bias = lgtplus.get_all_feature_bias(adprofile[adprofile.CTR != 0][adprofile.weightcount > 5000],
    #                                           lgtplus.remove_duplicate_combinations(lgtplus.replace_varname_to_id(L2)),
    #                                           multi_ft_cutoff=20)
    # L3_ft_bias = lgtplus.get_all_feature_bias(adprofile[adprofile.CTR != 0][adprofile.weightcount > 5000],
    #                                           lgtplus.remove_duplicate_combinations(lgtplus.replace_varname_to_id(L3)),
    #                                           multi_ft_cutoff=20)
    # L4_ft_bias = lgtplus.get_all_feature_bias(adprofile[adprofile.CTR != 0][adprofile.weightcount > 5000],
    #                                           lgtplus.remove_duplicate_combinations(lgtplus.replace_varname_to_id(L4)),
    #                                           multi_ft_cutoff=20)
    #
    # ## Getting Single Feature Bias
    # L1_ft_bias = lgtplus.get_all_feature_bias(adprofile[adprofile.CTR != 0][adprofile.weightcount > 5000],
    #                                           ['xspace', 'xad_crtsize', 'gender', 'bid_id',
    #                                            'quality_id', 'kadpage_image_profile_tone',
    #                                            'aesthetics_id',
    #                                            'kadpage_image_profile_similarclassid',
    #                                            'kadpage_image_profile_saturation', 'kadpage_image_profile_light',
    #                                            'kadpage_image_profile_colorful', 'kadpage_image_profile_colorlayout',
    #                                            'kadpage_image_profile_coloremotion',
    #                                            'kadpage_image_profile_scene_interest1',
    #                                            'kadpage_image_profile_scene_interest2',
    #                                            'kadpage_image_profile_scene_interest3',
    #                                            'kadpage_image_profile_style_interest1',
    #                                            'kadpage_image_profile_style_interest2',
    #                                            'kadpage_image_profile_style_interest3',
    #                                            'kadpage_image_profile_color_interest1',
    #                                            'kadpage_image_profile_color_interest2',
    #                                            'kadpage_image_profile_color_interest3',
    #                                            'kadpage_image_profile_object_interest1',
    #                                            'kadpage_image_profile_object_interest2',
    #                                            'kadpage_image_profile_object_interest3',
    #                                            'kadpage_image_profile_logo_interest1',
    #                                            'kadpage_image_profile_logo_interest2',
    #                                            'kadpage_image_profile_logo_interest3',
    #                                            'kadpage_image_profile_starface_interest1',
    #                                            'kadpage_image_profile_starface_interest2',
    #                                            'kadpage_image_profile_starface_interest3',
    #                                            'kadpage_image_profile_clothestagging_interest1',
    #                                            'kadpage_image_profile_clothestagging_interest2',
    #                                            'kadpage_image_profile_clothestagging_interest3',
    #                                            'kadpage_image_profile_bgcolor_interest1',
    #                                            'kadpage_image_profile_bgcolor_interest2',
    #                                            'kadpage_image_profile_bgcolor_interest3',
    #                                            'kadpage_image_profile_fgcolor_interest1',
    #                                            'kadpage_image_profile_fgcolor_interest2',
    #                                            'kadpage_image_profile_fgcolor_interest3', 'CAT1_ID', 'CAT2_ID'],
    #                                           multi_ft_cutoff=20)
    return feature_score

def visulize(cxr_score, bias_score, bias_weight=0.3, maxrank=4, topn=10):
    feat_score = []

    for rank in range(2, maxrank + 1):
        feat_score.append({})
        for feat, scores in cxr_score[rank - 2].items():
            feat_score[-1][feat] = [score * (1 - bias_weight) for score in scores]
        for feat, scores in bias_score[rank - 2].items():
            try:
                feat_score[-1][feat][0] += bias_weight * scores[0]
                feat_score[-1][feat][0] += bias_weight * scores[0]
            except KeyError:
                feat_score[-1][feat] = [score * bias_weight for score in scores]

    feat_names = []
    feat_imp = []
    feat_freq = []
    for rank in range(2, maxrank + 1):
        feat_names.append([])
        feat_imp.append([])
        feat_freq.append([])
        feat_score[rank - 2] = list(sorted(feat_score[rank - 2].items(), key=lambda x: x[1][0], reverse=True))
        for feat, scores in feat_score[rank - 2]:
            feat_names[-1].append(feat)
            feat_freq[-1].append(scores[0])
            feat_imp[-1].append(scores[1])
        # feat_names[-1] = np.array(feat_names[-1])
        # feat_imp[-1] = np.array(feat_imp[-1])
        # feat_freq[-1] = np.array(feat_freq[-1])

    def scaling(xlist):
        xmax = max(xlist)
        xmin = min(xlist)
        if xmax == xmin:
            return 1
        return [(x - xmin) / (xmax - xmin) for x in xlist]
    figure, ax = plt.subplots(maxrank - 1, 1, figsize=(20, 20 * (maxrank - 1)), dpi=80, constrained_layout=True)
    for i in range(maxrank - 1):
        names = feat_names[i]
        length = len(names)
        if length >= topn:
            length = topn
        names = names[:length]
        imps = feat_imp[i][:length]
        imps_sc = scaling(imps)
        freq = feat_freq[i][:length]
        print(names)
        print(freq)
        freq_sc = scaling(freq)
        x = np.arange(length)
        thisfig = ax[i]
        width = 0.3
        thisfig.set_xticks(x)
        thisfig.bar(x - 0.5 * width, freq_sc, width=width, label='frequent_path')
        thisfig.bar(x + 0.5 * width, imps_sc, width=width, color='crimson', label='feature_importance')
        # thisfig.set_xlabel('transition range')
        # thisfig.set_ylabel('pcvrbias')
        thisfig.set_xticklabels(names, rotation=20, ha='right', va='top', fontsize=15)
        for xl, imp in zip(x, imps):
            thisfig.text(xl - width, imp, '{:.2}'.format(imp), ha='center', va='bottom')
        for xl, fre in zip(x, freq):
            thisfig.text(xl, fre, '{}'.format(fre), ha='center', va='bottom')
        thisfig.set_title(settings.parse_key('Rank {}'.format(i + 2)), fontsize=15)

    plt.savefig('res.png')
    plt.show()

if __name__ == "__main__":
    trainning_file = settings.LGBM_train + '.csv'
    cxr, pcxrbias = settings.cols[2], settings.cols[3]
    cxrscore = model(cxr, trainning_file)
    biasscore = model(pcxrbias, trainning_file)
    visulize(cxrscore, biasscore)
