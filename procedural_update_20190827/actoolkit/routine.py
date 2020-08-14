# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:30:45 2019

@author: haozhjiang

A set of functions meant for routine updates.
"""

import numpy as np
from pandas import IntervalIndex, cut, DataFrame
from copy import deepcopy

def get_new_ID(input_dat, mapping,id_type):
    '''
    Given value to ID mapping from the original model, transform the values of incoming data to IDs.
    
    Exception handling:
        - if value is nan, ID is also nan
        - if value non-existent in original mapping, encode the value as max_ID+1
    '''
    ret = np.zeros(len(input_dat))
    original_val = np.array(input_dat[id_type])
    for i in range(len(original_val)):
        if np.isnan(original_val[i]): ## handle nan
            ret[i] = np.nan
        elif original_val[i] in mapping.keys(): ## regular mapping
            ret[i] = mapping[original_val[i]]
        else: ## create new id if value non-existent in original mapping
            mapping[original_val[i]] = max(mapping.values())+1
            ret[i] = mapping[original_val[i]]
        
    return ret
    
def get_bid_id(input_dat, original_buckets):
    '''
    Take the original bins for bid_prices, and make sure it contains the max/min value of bid_price in the incoming data. 
    Then transform bid_price to bid_id accordingly.
    '''
    working_dat = deepcopy(input_dat)
    max_price = np.max(input_dat['bid_price'])
    min_price = np.min(input_dat['bid_price'])
    
    if min_price < original_buckets[0][0]:
        original_buckets[0]=(min_price, original_buckets[0][1])
    if max_price > original_buckets[-1][1]:
        original_buckets[-1]=(original_buckets[-1][0],max_price)
        
    bins_bid = IntervalIndex.from_tuples(original_buckets,closed="left")
    working_dat["bid_price_bin"] = cut(working_dat["bid_price"], bins_bid)
    
    return working_dat.groupby(["bid_price_bin"]).grouper.group_info[0]
        
def get_shap_ranking(shap_value, feature_names):
    '''
    Returns a (# samples x # features) data frame; each value is the ranking of the feature (column) in a given sample (row)
    '''
    df_shap = DataFrame(shap_value).T
    ret_array = np.zeros((len(df_shap.columns),len(df_shap)),dtype=np.int)
    for i in range(len(df_shap.columns)):
        current_ranking = df_shap[i].rank(method = 'min',ascending=False)
        ret_array[i] = current_ranking
        
    ret_df = DataFrame(ret_array)
    ret_df.columns = feature_names
    
    return ret_df
        
def get_top_k_shap(shap_value, feature_names, k = 5):
    '''
    Returns two data frames (# samples x k), namely the top-K negative SHAP value features, and the top-K positive SHAP value features. 
    '''
    df_shap = DataFrame(shap_value).T
    pos_ret = np.zeros((len(df_shap.columns),k),dtype=np.object)
    neg_ret = np.zeros((len(df_shap.columns),k),dtype=np.object)
    for i in range(len(df_shap.columns)):
        current_sorted = df_shap[i].sort_values(ascending=True)
        neg_ret[i] = [(feature_names[k],current_sorted[k]) for k in current_sorted[0:5].index]
        pos_ret[i] = [(feature_names[j],current_sorted[j]) for j in current_sorted.iloc[[-1,-2,-3,-4,-5]].index]
        
    neg_df = DataFrame(neg_ret)
    pos_df = DataFrame(pos_ret)
    
    return neg_df, pos_df
    