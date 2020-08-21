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

def model(col, fname):
    ## Load Data
    adprofile = pd.read_csv(fname)
    # 原始场景是ctr的，因为文件中ctr太多就没有改
    adprofile.rename(columns={col: 'CTR'}, inplace=True)
    adprofile.rename(columns={settings.cols[-1]: 'weightcount'})
    # adprofile.drop(["xdatetime", "age", "ds", "importds", "updatedtime", "id"], axis=1, inplace=True)
    print(len(list(set(adprofile['adid']))))
    print(len(adprofile))

    ## Specify Continuous Features
    continuous_feats = settings.continuous_feats

    # Specify Features
    features = settings.ad_feats

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
    feature_importance = pd.DataFrame(feature_importance)
    feature_importance.columns = ['feature', 'importance']
    feature_importance.to_csv('feat_imp_{}.csv'.format(col), index=False)
    ## Getting All Paths
    adprofile_allwx_all_path = lgtplus.get_all_paths(lgbr_multi)

    ## Getting High Frequence Patterns
    feature_names = lgbr_multi._Booster.dump_model()["feature_names"]

    L2 = lgtplus.get_hi_freq_pattern(adprofile_allwx_all_path, 2, 50, True, feature_names)
    print(L2)
    L3 = lgtplus.get_hi_freq_pattern(adprofile_allwx_all_path, 3, 50, True, feature_names)
    print(L3)
    L4 = lgtplus.get_hi_freq_pattern(adprofile_allwx_all_path, 4, 50, True, feature_names)
    print(L4)
    frequent_crosses = pd.concat([pd.DataFrame(L2), pd.DataFrame(L3), pd.DataFrame(L4)])
    frequent_crosses.columns = ['cross_name', 'times']
    frequent_crosses.to_csv('frequent_crosses_{}.csv'.format(col), index=False)

    ## Getting Feature Combination Bias
    L2_ft_bias = lgtplus.get_all_feature_bias(adprofile[adprofile.CTR != 0][adprofile.weightcount > 5000],
                                              lgtplus.remove_duplicate_combinations(lgtplus.replace_varname_to_id(L2)),
                                              multi_ft_cutoff=20)
    L3_ft_bias = lgtplus.get_all_feature_bias(adprofile[adprofile.CTR != 0][adprofile.weightcount > 5000],
                                              lgtplus.remove_duplicate_combinations(lgtplus.replace_varname_to_id(L3)),
                                              multi_ft_cutoff=20)
    L4_ft_bias = lgtplus.get_all_feature_bias(adprofile[adprofile.CTR != 0][adprofile.weightcount > 5000],
                                              lgtplus.remove_duplicate_combinations(lgtplus.replace_varname_to_id(L4)),
                                              multi_ft_cutoff=20)

    ## Getting Single Feature Bias
    L1_ft_bias = lgtplus.get_all_feature_bias(adprofile[adprofile.CTR != 0][adprofile.weightcount > 5000],
                                              ['xspace', 'xad_crtsize', 'gender', 'bid_id',
                                               'quality_id', 'kadpage_image_profile_tone',
                                               'aesthetics_id',
                                               'kadpage_image_profile_similarclassid',
                                               'kadpage_image_profile_saturation', 'kadpage_image_profile_light',
                                               'kadpage_image_profile_colorful', 'kadpage_image_profile_colorlayout',
                                               'kadpage_image_profile_coloremotion',
                                               'kadpage_image_profile_scene_interest1',
                                               'kadpage_image_profile_scene_interest2',
                                               'kadpage_image_profile_scene_interest3',
                                               'kadpage_image_profile_style_interest1',
                                               'kadpage_image_profile_style_interest2',
                                               'kadpage_image_profile_style_interest3',
                                               'kadpage_image_profile_color_interest1',
                                               'kadpage_image_profile_color_interest2',
                                               'kadpage_image_profile_color_interest3',
                                               'kadpage_image_profile_object_interest1',
                                               'kadpage_image_profile_object_interest2',
                                               'kadpage_image_profile_object_interest3',
                                               'kadpage_image_profile_logo_interest1',
                                               'kadpage_image_profile_logo_interest2',
                                               'kadpage_image_profile_logo_interest3',
                                               'kadpage_image_profile_starface_interest1',
                                               'kadpage_image_profile_starface_interest2',
                                               'kadpage_image_profile_starface_interest3',
                                               'kadpage_image_profile_clothestagging_interest1',
                                               'kadpage_image_profile_clothestagging_interest2',
                                               'kadpage_image_profile_clothestagging_interest3',
                                               'kadpage_image_profile_bgcolor_interest1',
                                               'kadpage_image_profile_bgcolor_interest2',
                                               'kadpage_image_profile_bgcolor_interest3',
                                               'kadpage_image_profile_fgcolor_interest1',
                                               'kadpage_image_profile_fgcolor_interest2',
                                               'kadpage_image_profile_fgcolor_interest3', 'CAT1_ID', 'CAT2_ID'],
                                              multi_ft_cutoff=20)
    return feature_importance, frequent_crosses


if __name__ == "__main__":
    trainning_file = 'lgbmtrain.csv'
    feat_imp, fre_path = model('cvr', trainning_file)
    model('pcvrbias', trainning_file)
