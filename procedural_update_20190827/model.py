# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 20:44:26 2019

@author: haozhjiang
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy
from math import ceil
from lightgbm import LGBMRanker
import numpy as np
import actoolkit.lgtplus as lgtplus
import shap
import pickle


## Two Helper Functions
def binning_prices(input_dat, num_bins):
    '''
    Binning bid_price into user specified number of bins. Outputs a new data frame with bid_id, along with the bins used to create this id. 
    '''
    ret_dat = deepcopy(input_dat)
    
    bid_price = input_dat[["CTR","bid_price"]]
    bin_num = num_bins
    bid_price.sort_values(by='bid_price',axis = 0)['bid_price']
    
    bid_bins = []
    n = len(input_dat)
    current_bound = 0
    for i in range(bin_num):
        if i != (bin_num - 1):
            bid_bins.append((current_bound, bid_price.sort_values(by='bid_price',axis = 0)['bid_price'].iloc[int(i*ceil(n / bin_num))]))
            current_bound = bid_price.sort_values(by='bid_price',axis = 0)['bid_price'].iloc[int(i*ceil(n / bin_num))]
        else:
            bid_bins.append((current_bound, int(bid_price['bid_price'].max())+1))
    
    bins_bid = pd.IntervalIndex.from_tuples(bid_bins,closed="left")
    ret_dat["bid_price_bin"] = pd.cut(ret_dat["bid_price"], bins_bid)
    ret_dat["bid_id"] = ret_dat.groupby(["bid_price_bin"]).grouper.group_info[0]
    ret_dat.drop("bid_price_bin",axis = 1,inplace=True)
    
    return ret_dat, bid_bins


def binning_image_profile(input_dat):
    '''
    Binning image aesthetics and image quality into pre-specified bins. Outputs a new data frame with the new IDs, along with the mapping obtained (original value to ID)
    '''
    ret_dat = deepcopy(input_dat)
    
    quality_bins = pd.IntervalIndex.from_tuples([(0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),
                                                (0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1)],closed="left")
    aesthetics_bins = pd.IntervalIndex.from_tuples([(1,2),(2,3),(3,4),(4,5),(5,6),
                                                (6,7),(7,8),(8,9),(9,10)],closed="left")
    ret_dat["aesthetics_bin"] = pd.cut(ret_dat["kadpage_image_profile_aesthetics"], aesthetics_bins)
    ret_dat["quality_bin"] = pd.cut(ret_dat["kadpage_image_profile_quality"], quality_bins)
    ret_dat["aesthetics_id"] = ret_dat.groupby(["aesthetics_bin"]).grouper.group_info[0]
    ret_dat["quality_id"] = ret_dat.groupby(["quality_bin"]).grouper.group_info[0]
    qid_relations = ret_dat[['kadpage_image_profile_aesthetics','aesthetics_id']].drop_duplicates()
    aid_relations = ret_dat[['kadpage_image_profile_quality','quality_id']].drop_duplicates()

    ret_dat.drop("aesthetics_bin",axis = 1,inplace=True)
    ret_dat.drop("quality_bin",axis = 1,inplace=True)
    
    return ret_dat,qid_relations,aid_relations

def get_pairs(input_data,granularity = 100,cutoff_ratio = 1.5,MAX_ITER = 10000,MAX_GROUP = 10000,verbose=True):
    '''
    Returning a data frame of paired data, given required CTR ratio. Supports early stopping based on max iteration, max amount of groups and granularity.
    
    granularity: record every [granularity] pairs
    cutoff_ratio: required ratio of CTRs in a pair
    '''
    ## create copy of data
    dat_copy = deepcopy(input_data)
    dat_copy["new_group"] = 0
    
    dat_copy = binning_prices(dat_copy,10)[0]
    dat_copy["GroupId"] = dat_copy.groupby(["xspace","CAT1_ID","CAT2_ID","gender","bid_id"]).grouper.group_info[0]
    
    ## initialize group count
    grp_count = 0
    iter_count = 0
    
    ## create template return data frame
    ret_df = pd.DataFrame(columns=dat_copy.columns)
    
    ## start loop
    for i in range(len(np.unique(dat_copy["GroupId"]))):
        current = dat_copy[dat_copy["GroupId"] == i]
        
        ## exit current iteration if observation count = 1
        if len(current) == 1:
            continue
        else:
            for j in range(len(current)-1):
                for k in range(1,len(current)):                                       
                    ## exiting loop according to given early_stop
                    if iter_count >= MAX_ITER:
                        break
                    if grp_count >= MAX_GROUP:
                        break
                    ## filtering according to given cutoff
                    if current['CTR'].iloc[j] == 0 and current['CTR'].iloc[k] == 0:
                        continue
                    elif current['CTR'].iloc[j] == 0 or current['CTR'].iloc[k] == 0:
                        if iter_count % granularity == 0:
                            current.at[current.index[j],"new_group"] = grp_count
                            current.at[current.index[k],"new_group"] = grp_count
                            ret_df = ret_df.append(current.iloc[j,:])
                            ret_df = ret_df.append(current.iloc[k,:])
                            grp_count += 1
                             ## if verbose, print progress so far
                            if verbose: 
                                if grp_count % 500 == 0:
                                    print("Completed: "+str(grp_count))
                        iter_count += 1
                    elif current["CTR"].iloc[j]/current["CTR"].iloc[k] < 1/cutoff_ratio or current["CTR"].iloc[j]/current["CTR"].iloc[k] > cutoff_ratio:
                        if iter_count % granularity == 0:
                            current.at[current.index[j],"new_group"] = grp_count
                            current.at[current.index[k],"new_group"] = grp_count
                            ret_df = ret_df.append(current.iloc[j,:])
                            ret_df = ret_df.append(current.iloc[k,:])
                            grp_count += 1
                             ## if verbose, print progress so far
                            if verbose: 
                                if grp_count % 500 == 0:
                                    print("Completed: "+str(grp_count))
                        iter_count += 1
            
    xtrain_new = ret_df.iloc[:,1:-3]
    ytrain_new = ret_df.iloc[:,0]
    
    xtrain_new = xtrain_new.apply(pd.to_numeric)
    ytrain_new = pd.to_numeric(ytrain_new)
    
    
    return xtrain_new, ytrain_new


if __name__ == "__main__":
    
    ## Load Data
    adprofile = pd.read_csv("adprofile_allwx_20190820.csv")
    adprofile.rename(columns={'ctr':'CTR'}, inplace=True)
    adprofile.drop(["xdatetime","age","ds","importds","updatedtime","id"],axis = 1,inplace=True)
    print(len(list(set(adprofile['adid']))))
    print(len(list(set(adprofile['tid']))))
    print(len(adprofile))
    
    ## Re-IDing CAT1/CAT2 ID 
    adprofile["CAT1_ID"] = adprofile.groupby(["xad_cagegory1id"]).grouper.group_info[0]
    adprofile["CAT2_ID"] = adprofile.groupby(["xad_cagegory2id"]).grouper.group_info[0]
    
    ## Create CAT1/CAT2 Mapping
    cat1_relations = adprofile[['xad_cagegory1id','CAT1_ID']].drop_duplicates()
    cat2_relations = adprofile[['xad_cagegory2id','CAT2_ID']].drop_duplicates()
    
    ## Drop Original ID
    adprofile.drop(["xad_cagegory1id","xad_cagegory2id"],axis = 1, inplace=True)
    
    ## Binning Continuous Variables and Keep Mapping
    adprofile,bid_bins = binning_prices(adprofile,10)
    adprofile,qid_relations, aid_relations = binning_image_profile(adprofile)
    
    ## Create Categorical Mapping
    cat1_map = {pairs[0]:pairs[1] for pairs in np.array(cat1_relations)}
    cat2_map = {pairs[0]:pairs[1] for pairs in np.array(cat2_relations)}
    qid_map = {pairs[0]:pairs[1] for pairs in np.array(qid_relations)}
    aid_map = {pairs[0]:pairs[1] for pairs in np.array(aid_relations)}
    
    cat_idname_mapping = {'xad_cagegory1id':"CAT1_ID",
                          'xad_cagegory2id':"CAT2_ID",
                          'kadpage_image_profile_quality':"quality_id",
                          'kadpage_image_profile_aesthetics':"aesthetics_id"}
    
    cat_new_id_mapping = {'xad_cagegory1id':cat1_map,
                          'xad_cagegory2id':cat2_map,
                          'kadpage_image_profile_quality':qid_map,
                          'kadpage_image_profile_aesthetics':aid_map}
    
    ## Dropping all-NULL Columns
    adprofile.dropna(axis = 1,how = "all",inplace = True)
    
    ## Creating Test/Training Set
    
    ## Specify Features to use
    x_feature_list = ['xspace', 'xad_crtsize', 'gender', 'bid_price',
           'kadpage_image_profile_quality', 'kadpage_image_profile_tone',
           'kadpage_image_profile_aesthetics',
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
           'kadpage_image_profile_fgcolor_interest3', 'CAT1_ID', 'CAT2_ID']
    
    x_train,x_test,y_train,y_test = train_test_split(adprofile.drop('CTR',axis = 1)[x_feature_list],adprofile['CTR'],test_size = 0.2,random_state=1)
    
    ## Specify Categorical Features
    cat_ft_list = [0, 1, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
                   25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    
    ## Create Pairs, see comment in original code
    # x_pairs, y_pairs = get_pairs(pd.DataFrame(y_train).join(x_train),granularity = 10,cutoff_ratio = 2,MAX_ITER = np.inf ,MAX_GROUP = 50000,verbose=True)
    # x_pairs.to_csv('newx.csv', index=False)
    # y_pairs.to_csv('newy.csv', index=False)
#   Load Pairs from File
    x_pairs = pd.read_csv("x_pairs.csv")
    x_pairs.set_index('Unnamed: 0', inplace=True)
    y_pairs = pd.read_csv("y_pairs.csv",header=None)
    y_pairs.set_index(0, inplace=True)
    y_pairs.columns = ['CTR']
    
    ## Train LGBMRanker
    lgbr_multi = LGBMRanker(objective = "regression",
                      learning_rate = 0.248,
                       num_leaves = 300,
                       # num_trees = 325,
                       num_trees = 10,
                       max_depth=20)
    
    lgbr_multi.fit(x_pairs,y_pairs, group = [2 for i in range(int(len(x_pairs)/2))],categorical_feature = cat_ft_list)    
    
    ## Getting All Paths
    adprofile_allwx_all_path = lgtplus.get_all_paths(lgbr_multi)    
    
    ## Getting High Frequence Patterns
    feature_names = lgbr_multi._Booster.dump_model()["feature_names"]
    print(adprofile_allwx_all_path[9])
    L2 = lgtplus.get_hi_freq_pattern(adprofile_allwx_all_path,2,50,True,feature_names)
    L3 = lgtplus.get_hi_freq_pattern(adprofile_allwx_all_path,3,50,True,feature_names)
    L4 = lgtplus.get_hi_freq_pattern(adprofile_allwx_all_path,4,50,True,feature_names)
    
    ## Getting Feature Combination Bias
    L2_ft_bias = lgtplus.get_all_feature_bias(adprofile[adprofile.CTR != 0][adprofile.imp > 5000],lgtplus.remove_duplicate_combinations(lgtplus.replace_varname_to_id(L2)),multi_ft_cutoff = 20)
    L3_ft_bias = lgtplus.get_all_feature_bias(adprofile[adprofile.CTR != 0][adprofile.imp > 5000],lgtplus.remove_duplicate_combinations(lgtplus.replace_varname_to_id(L3)),multi_ft_cutoff = 20)
    L4_ft_bias = lgtplus.get_all_feature_bias(adprofile[adprofile.CTR != 0][adprofile.imp > 5000],lgtplus.remove_duplicate_combinations(lgtplus.replace_varname_to_id(L4)),multi_ft_cutoff = 20)

    ## Getting Single Feature Bias
    L1_ft_bias = lgtplus.get_all_feature_bias(adprofile[adprofile.CTR != 0][adprofile.imp > 5000],['xspace', 'xad_crtsize', 'gender', 'bid_id',
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
           'kadpage_image_profile_fgcolor_interest3', 'CAT1_ID', 'CAT2_ID'], multi_ft_cutoff = 20)
    
    ## SHAP Value
    lm_ex = shap.TreeExplainer(lgbr_multi)
    lm_ex_shap  = lm_ex.shap_values(adprofile[x_feature_list],adprofile['CTR'])
    shap.summary_plot(lm_ex_shap,adprofile[x_feature_list] ,feature_names,max_display = 42,plot_type="bar") ## importance plot
    
    ## Save Model to File
    pickle.dump([lgbr_multi,cat_idname_mapping,cat_new_id_mapping,bid_bins], open("lgt_model.p", "wb"))
