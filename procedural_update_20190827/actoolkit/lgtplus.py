# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:01:05 2019

@author: haozhjiang

A customized set of functions and tools developed for the LightGBM data structure.

High Frequency patterns is built using pyspark.
Plotting requires GRAPHVIZ. 

"""


#######################################
####### SETTING UP DEPENDENCIES #######
#######################################

from lightgbm.compat import (GRAPHVIZ_INSTALLED, string_type)
import numpy as np
import pandas as pd
from copy import deepcopy

import findspark
import os
os.environ['JAVA_HOME']='C:\Program Files\Java\jdk-11.0.8'
findspark.init('D:\spark')
from pyspark import SparkConf, SparkContext
#from continuous import get_h ## for info_gain only


################################
####### HELPER FUNCTIONS #######
################################

def rmse(predictions, targets):
    '''
    Calculate the rmse of the predictions.
    Need predictions and imnputs of same length, in array-like format.
    '''
    return np.sqrt(((predictions - targets) ** 2).mean())/predictions[0]

def _float2str(value, precision=None):
    return ("{0:.{1}f}".format(value, precision)
            if precision is not None and not isinstance(value, string_type)
            else str(value))
    
def node_count(node,current_count = None):
    '''
    Return the node count of the full (sub)tree, using the supplied node as the root. 
    '''
    if current_count is None:
        current_count = 0
    
    if is_leaf(node):
        current_count += 1
    else:
        current_count = current_count + 1 + node_count(node["right_child"],current_count) + node_count(node["left_child"],current_count)
        
    return current_count

def is_leaf(node):
    '''
    Returns True if current node is a leaf node. 
    '''
    return not (("right_child" in node.keys()) & ("left_child" in node.keys()))

def total_ctr(dat):
    '''
    Returns the average CTR of the dat; by dividing sum of clicks with sum of impressions.
    
    dat: Pandas DataFrame. Must have columns called "clk" and "imp".
    '''
    if sum(dat["weightcount"]) > 0:
        return sum(dat["weightcount"] * dat['CTR'])/sum(dat["weightcount"])
    else:
        return "None"
    
def dcg_at_k(r, k):
    '''
    Calculate the DCG@k score.
    '''
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    '''
    Calculate the NDCG@k score.
    '''
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

###############################################
####### TREE STRUCTURE UPDATE FUNCTIONS #######
###############################################
        
def update_metrics(current_node,current_threshold,current_feature,current_data,feature_names,branch,global_mean=None):
    '''
    Update Tree structure with customized metrics and information like CTR and URL of ads belonging to the current node. 
    '''
    parent_pred = total_ctr(current_data)
    if branch == "L":
        if isinstance(current_threshold, list):
            current_data = current_data[current_data[current_feature].isin(current_threshold)]
        elif isinstance(current_threshold, float):
            current_data = current_data[current_data[current_feature] <= current_threshold]
    if branch == "R":
        if isinstance(current_threshold, list):
            current_data = current_data[~current_data[current_feature].isin(current_threshold)]    
        elif isinstance(current_threshold, float):
            current_data = current_data[current_data[current_feature] > current_threshold]
            
    if branch == "root":
        current_data = current_data
        if global_mean is None:
            global_mean = total_ctr(current_data)
        
    current_node["average_CTR"] = total_ctr(current_data)
    current_node["abs_mean_bias"] = np.mean(abs((total_ctr(current_data) - current_data['CTR']) / current_data['CTR'])) * 100
    current_node["abs_mean_bias_root"] = np.mean(abs((global_mean - current_data['CTR']) / current_data['CTR'])) * 100
    current_node["n_samples"] = len(current_data)
    current_node["bias_ratio_parent"] = 100*np.mean(np.mean(abs((total_ctr(current_data) - current_data['CTR']) / current_data['CTR'])) - np.mean(abs((parent_pred - current_data['CTR']) / current_data['CTR'])))
    current_node["bias_ratio_root"] = 100*np.mean(np.mean(abs((total_ctr(current_data) - current_data['CTR']) / current_data['CTR'])) - np.mean(abs((global_mean - current_data['CTR']) / current_data['CTR'])))
#    current_node["ad_url"] = current_data["ad_img_url"]
    current_node["SD_CTR"] = np.var(current_data["CTR"]) ** 0.5
    current_node["CV_CTR"] = (np.var(current_data["CTR"]) ** 0.5)/(np.mean(current_data["CTR"])) * 100
    current_node["SD_Ratio"] = 100 * sum((current_data["CTR"] - np.mean(current_data["CTR"]))**2) / (sum((current_data["CTR"] - parent_pred)**2)+0.0000001)
#    current_node["Positive_Rate"] = np.sum(current_data["response"]) / len(current_data) * 100
#    current_node["rmse"] = rmse(np.ones(len(current_data))*np.mean(current_data['CTR']),current_data["CTR"])
    current_node["CTR"] = np.array(current_data["CTR"])
    current_node['total_imp'] = np.sum(current_data['imp'])
        
    if is_leaf(current_node):
        return current_node
    else:
        
        if isinstance(current_node["threshold"],str):
            current_threshold = [int(item) for item in current_node["threshold"].split("||")]
        else:
            current_threshold = current_node["threshold"]
            
        current_feature = feature_names[current_node["split_feature"]]
        ## Left Child
        current_node.update({"left_child": update_metrics(current_node["left_child"],current_threshold,current_feature,current_data,feature_names,"L",global_mean)})
        ## Right Child
        current_node.update({"right_child": update_metrics(current_node["right_child"],current_threshold,current_feature,current_data,feature_names,"R",global_mean)})
        return current_node

def update_node_id(current_node,current_count,branch):
    '''
    Update the tree structure with node_id
    '''
    if branch == "root":
        current_count = 0
    
    current_node['node_id'] = current_count

    if is_leaf(current_node):
        return current_node
    
    else:           
        ## Left Child
        current_node.update({"left_child": update_node_id(current_node["left_child"],current_count+1,"L")})
        ## Right Child
        current_node.update({"right_child": update_node_id(current_node["right_child"],current_count+1+node_count(current_node['left_child']),"R")})
        return current_node

def translate_node(current_node,ft_dict,feature_names):
    '''
    Translate feature names into their physical meaning (in Chinese)
    '''
    if is_leaf(current_node):
        return current_node
    else:
        
        current_feature = feature_names[current_node["split_feature"]]
        current_threshold = ", ".join([ ft_dict[current_feature][int(threshold)] for threshold in current_node["threshold"].split("||")])
        current_node["threshold"] = current_threshold
        
        ## Left Child
        current_node.update({"left_child": translate_node(current_node["left_child"],ft_dict,feature_names)})
        ## Right Child
        current_node.update({"right_child": translate_node(current_node["right_child"],ft_dict,feature_names)})
        
        return current_node
    
def update_tree(X,lgbm_classifier,ft_dict = None,metrics = True,node_id=True,translate = False,index=0):
    '''
    Update tree structure with custom metrics, node id and or physical meaning of features. 
    '''
    dt = lgbm_classifier._Booster.dump_model()["tree_info"][index].copy()
    feature_names = lgbm_classifier._Booster.dump_model()["feature_names"]
    
    parent_node = parent_node = dt["tree_structure"]
        
    if isinstance(parent_node["threshold"],str):
        current_threshold = [int(item) for item in parent_node["threshold"].split("||")]
    else:
        current_threshold = parent_node["threshold"]
        
    current_feature = feature_names[parent_node["split_feature"]]
    current_data = X 
    
    if metrics:
        dt.update({"tree_structure": update_metrics(parent_node,current_threshold,current_feature,current_data,feature_names,"root")})
    if node_id:
        dt.update({"tree_structure": update_node_id(parent_node,0,"root")})
    if translate:
        if ft_dict is None:
            raise ValueError("Need to supply ft_dict to translate nodes.")
        dt.update({"tree_structure": translate_node(parent_node,ft_dict,feature_names)})
        
    return dt

#################################################
####### TREE STRUCTURE PLOTTING FUNCTIONS #######
#################################################
    
def _to_graphviz(tree_info, show_info, feature_names, precision=None, **kwargs):
    """Convert specified tree to graphviz instance.

    See:
      - https://graphviz.readthedocs.io/en/stable/api.html#digraph
    """
    if GRAPHVIZ_INSTALLED:
        from graphviz import Digraph
    else:
        raise ImportError('You must install graphviz to plot tree.')

    def add(root, parent=None, decision=None):
        """Recursively add node or edge."""
        if 'split_index' in root:  # non-leaf
            name = 'split{0}'.format(root['split_index'])
            if feature_names is not None:
                label = 'split_feature_name: {0}'.format(feature_names[root['split_feature']])
            else:
                label = 'split_feature_index: {0}'.format(root['split_feature'])
            label += r'\nthreshold: {0}'.format(_float2str(root['threshold'], precision))
            for info in show_info:
                if info in {'split_gain', 'internal_value','average_CTR','rmse',"info_gain"}:
                    label += r'\n{0}: {1}'.format(info, _float2str(root[info], precision))
                elif info in {'SD_CTR'}:
                    label += r'\n{0}: {1}'.format(info, _float2str(root[info], precision+3))
                elif info in {'abs_mean_bias','abs_mean_bias_root','CV_CTR','Positive_Rate','SD_Ratio','bias_ratio_parent','bias_ratio_root'}:
                    label += r'\n{0}: {1}'.format(info, _float2str(root[info], 2)+"%")
                elif info in {'internal_count','n_samples','node_id'}:
                    label += r'\n{0}: {1}'.format(info, root[info])
            graph.node(name, label=label)
            if root['decision_type'] == '<=':
                l_dec, r_dec = '<=', '>'
            elif root['decision_type'] == '==':
                l_dec, r_dec = 'is', "isn't"
            else:
                raise ValueError('Invalid decision type in tree model.')
            add(root['left_child'], name, l_dec)
            add(root['right_child'], name, r_dec)
        else:  # leaf
            name = 'leaf{0}'.format(root['leaf_index'])
            label = 'leaf_index: {0}'.format(root['leaf_index'])
            label += r'\nleaf_value: {0}'.format(_float2str(root['leaf_value'], precision))
            if 'leaf_count' in show_info:
                label += r'\nleaf_count: {0}'.format(root['leaf_count'])
            if 'average_CTR' in show_info:
                label += r'\naverage_CTR: {0}'.format(_float2str(root['average_CTR'], precision))
#            if 'info_gain' in show_info:
#                label += r'\ninfo_gain: {0}'.format(_float2str(root['info_gain'], precision))
            if 'SD_CTR' in show_info:
                label += r'\nSD_CTR: {0}'.format(_float2str(root['SD_CTR'], precision+3))
            if 'CV_CTR' in show_info:
                label += r'\nCV_CTR: {0}'.format(_float2str(root['CV_CTR'], 2) + "%")
            if 'SD_Ratio' in show_info:
                label += r'\nSD_Ratio: {0}'.format(_float2str(root['SD_Ratio'], 2) + "%")
            if 'bias_ratio_parent' in show_info:
                label += r'\nbias_ratio_parent: {0}'.format(_float2str(root['bias_ratio_parent'], 2) + "%")
            if 'bias_ratio_root' in show_info:
                label += r'\nbias_ratio_root: {0}'.format(_float2str(root['bias_ratio_root'], 2) + "%")
            if 'abs_mean_bias' in show_info:
                label += r'\nabs_mean_bias: {0}'.format(_float2str(root['abs_mean_bias'], 2) + "%")
            if 'abs_mean_bias_root' in show_info:
                label += r'\nabs_mean_bias_root: {0}'.format(_float2str(root['abs_mean_bias_root'], 2) + "%")
            if 'rmse' in show_info:
                label += r'\nrmse: {0}'.format(_float2str(root['rmse'], precision))
            if 'Positive_Rate' in show_info:
                label += r'\nPositive_Rate: {0}'.format(_float2str(root['Positive_Rate'], 2) + "%")    
            if 'n_samples' in show_info:
                label += r'\nn_samples: {0}'.format(root['n_samples'])
            if 'node_id' in show_info:
                label += r'\nnode_id: {0}'.format(root['node_id'])
            graph.node(name, label=label)
        if parent is not None:
            graph.edge(parent, name, decision)

    graph = Digraph(**kwargs)
    add(tree_info['tree_structure'])
    
    return graph


def summary_plot_to_graphviz(X,lgbm_classifier, index,ft_dict,translate=False,precision = 4, graph = True):
    '''
    Plot the selected tree (index) in the supplied lgbm_classifier, along with custom metrics. 
    '''
    feature_names = lgbm_classifier._Booster.dump_model()["feature_names"]
    dt = lgbm_classifier._Booster.dump_model()["tree_info"][index].copy()
    
    parent_node = parent_node = dt["tree_structure"]
        
    if isinstance(parent_node["threshold"],str):
        current_threshold = [int(item) for item in parent_node["threshold"].split("||")]
    else:
        current_threshold = parent_node["threshold"]
        
    current_feature = feature_names[parent_node["split_feature"]]
    current_data = X 
    
    dt.update({"tree_structure": update_metrics(parent_node,current_threshold,current_feature,current_data,feature_names,"root")})
    dt.update({"tree_structure": update_node_id(parent_node,current_threshold,current_feature,current_data,feature_names,"root")})
    if translate:
        dt.update({"tree_structure": translate_node(parent_node,ft_dict,feature_names)})
        
    if graph:
        return _to_graphviz(dt, show_info=["average_CTR","node_id","abs_mean_bias","abs_mean_bias_root","CV_CTR","SD_Ratio","n_samples","bias_ratio_root","bias_ratio_parent"],feature_names=feature_names,precision=precision) ## ["Positive_Rate","average_CTR","Abs_Mean_Bias","SD_CTR","CV_CTR","n_samples"]
    else:
        return dt

################################
####### OUTPUT FUNCTIONS #######
################################

def get_path(node,paths=None,current_path = None):
    '''
    takes a node, returns all the paths from the node to any leaves
    '''
    if paths is None:
        paths = []
    if current_path is None:
        current_path = []
        
    
    if is_leaf(node):
        paths.append(current_path)
    else:
        current_path.append(node["split_feature"])
        get_path(node["right_child"],paths,list(current_path))
        get_path(node["left_child"],paths,list(current_path))
    return [list(i) for i in set(tuple(i) for i in paths)]

def get_all_paths(lgbm_ensemble,verbose=True):
    '''
    takes a lgb model object, returns all the paths in the ensemble estimator
    '''
    num_trees = lgbm_ensemble.num_trees
    paths = {}
    for i in range(num_trees):
        if verbose:
            print("Now Collecting Tree #"+str(i))
        current_tree = lgbm_ensemble._Booster.dump_model()["tree_info"][i].copy()
        paths[i] = get_path(current_tree["tree_structure"])
        
    return paths

def get_node_id(input_data, ensemble_estimator,feature_names,index = 0):
    '''
    input_data: contains all the feature used in building the tree
    ensemble_estimator: lgb object with single decision tree
    
    returns a list of leaf_index if list length > 1; otherwise return the single leaf_index
    '''
    ret_list = np.ones(len(input_data),dtype = 'object')
    
    starting_node = deepcopy(ensemble_estimator._Booster.dump_model()["tree_info"][index])
    starting_node.update({"tree_structure": update_node_id(starting_node["tree_structure"],0,"root")})    
    
    for i in range(len(input_data)):
        current_data = input_data.iloc[i,:]
        current_node = starting_node['tree_structure']
        ret_list[i] = []
        while not is_leaf(current_node):
            ret_list[i].append(current_node['node_id'])
            #print(current_node["split_feature"])
            ## if current feature is continuous
            if current_node["decision_type"] == "<=":
                threshold = current_node["threshold"]
                if current_data[feature_names[current_node["split_feature"]]] <= threshold:
                    current_node = current_node['left_child']
                else:
                    current_node = current_node['right_child']
                    
            ## if current feature is discrete:
            elif current_node["decision_type"] == "==":
                threshold = [int(item) for item in current_node["threshold"].split("||")]
                if current_data[feature_names[current_node["split_feature"]]] in threshold:
                    current_node = current_node['left_child']
                else:
                    current_node = current_node['right_child']
                    
        ret_list[i].append(current_node['node_id'])
                
    ## return leaf index
    if len(ret_list) == 1:
        return ret_list[0]
    else:
        return ret_list


def get_high_CTR_path(node,feature_names,ctr_cutoff,bias_cutoff,current_ft=None,decision_type=None,current_threshold=None, paths=None,current_path = None,branch='root'):
    '''
    takes a node, returns all the paths from the node to any leaves, satisfying the two cutoffs:
        ctr_cutoff: only record the path if the leaf node has an average CTR above the cutoff
        bias_cutoff: only record the path is the leaf node has an average mean bias below the cutoff
    '''
    if paths is None:
        paths = []
    if current_path is None:
        current_path = []
        
    if branch =="R":
        if decision_type == "==":
            current_decision = "!="
        elif decision_type == "<=":
            current_decision = ">"
            
        current_path.append(tuple([current_ft,current_decision,current_threshold]))
            
    elif branch =="L":
        if decision_type == "==":
            current_decision = "=="
        elif decision_type == "<=":
            current_decision = "<="
        
        current_path.append(tuple([current_ft,current_decision,current_threshold]))
    
    
    if is_leaf(node):
        if node['average_CTR'] >= ctr_cutoff and node['Abs_Mean_Bias'] <= bias_cutoff:
            paths.append(current_path)
    else:
        current_ft = feature_names[node["split_feature"]]
        decision_type = node['decision_type']
        if isinstance(node['threshold'],str):
            current_threshold = "["+",".join(node['threshold'].split("||"))+"]"
        else:
            current_threshold = str(node['threshold'])
#        current_path.append(tuple([current_ft_name,current_decision,current_threshold]))
        get_high_CTR_path(node["right_child"],feature_names,ctr_cutoff,bias_cutoff,current_ft,decision_type,current_threshold,paths,list(current_path),"R")
        get_high_CTR_path(node["left_child"],feature_names,ctr_cutoff,bias_cutoff,current_ft,decision_type,current_threshold,paths,list(current_path),"L")
        
        
    return [list(i) for i in set(tuple(i) for i in paths)]

def get_all_high_CTR_path(X, lgbm_ensemble,feature_names,ctr_cutoff,bias_cutoff,verbose=True):
    '''
    Returns all the paths satisfying the two cutoffs in a tree ensemble estimator. 
    '''
    num_trees = lgbm_ensemble.num_trees
    paths = {}
    for i in range(num_trees):
        if verbose:
            print("Now Collecting Tree #"+str(i))
        current_tree = update_tree(X,lgbm_ensemble,index = i,info_gain=False)
        current_paths = get_high_CTR_path(current_tree["tree_structure"],feature_names,ctr_cutoff,bias_cutoff)
        
        paths[i] = [" & ".join([" ".join(node) for node in path]) for path in current_paths]
        
    return paths
        
def global_bias(node, current_bias = None):
    '''
    Return the sum of (bias * impression) of all the leaf nodes, using the user-supplied node as the root.  
    '''
    if current_bias is None:
        current_bias = 0
        
    if is_leaf(node):
        if node['total_imp'] != 0:
            current_bias += node["total_imp"] * (node["abs_mean_bias"]/100)
    else:
        current_bias += global_bias(node["right_child"], current_bias) + global_bias(node["left_child"], current_bias)
        
    return current_bias

def global_imp(node, current_count = None):
    '''
    Return the total impressions of samples in all of the leaf nodes in the given tree, using the user-supplied node as the root. 
    '''
    if current_count is None:
        current_count = 0
        
    if is_leaf(node):
        if node['total_imp'] != 0:
            current_count += node["total_imp"]
    else:
        current_count += global_imp(node["right_child"], current_count) + global_imp(node["left_child"], current_count)
        
    return current_count

def get_global_bias(X,lgbm_ensemble,head = np.inf):
    '''
    Return the global bias of an ensemble estimator; weighted average of bias by impression
    '''
    num_trees = lgbm_ensemble.num_trees
    
    current_bias = 0
    current_imp = 0
    
    for i in range(min(head,num_trees)):
        current_tree = update_tree(X,lgbm_ensemble,index = i)
        current_bias += global_bias(current_tree["tree_structure"])
        current_imp += global_imp(current_tree["tree_structure"])
        
    return current_bias/current_imp

################################################
####### HI FREQUENCY PATHS FUNCTIONALITY #######
################################################
    
def get_hi_freq_pattern(paths,length = 4,head = 50,translate=False,feature_names =None):
    '''
    THIS FUNCTION IS WRITTEN WITH PYSPARK
    
    Takes a dictionary of all paths in a tree estimator, returns the combination of features of the user-supplied length.
    
    length: Desired length of feature combinations
    head: only return the top-k combinations, in terms of frequency
    translate: if true, translate the number encoding of features into its names, according to the supplied feature_names
    feature_names: an array of feature_names, in the same order as it is used in training the tree ensemble
    '''
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    
    paths = sc.parallelize(list(paths.items()))

    L_spark = (paths.flatMap(lambda l:  ( (l[0], [(pat,1) for pat in generate_non_duplicate_paths(path,length) ]) for path in l[1] )  ) 
               .reduceByKey(lambda n1,n2:n1+n2)
               .mapValues(lambda l: list(set(l)))
               .flatMap(lambda l: (pairs for pairs in l[1]))
               .reduceByKey(lambda n1,n2:n1+n2)
               .sortBy(lambda l: l[1],ascending=False))
    
    if head is not None:
        ret = L_spark.take(head)
    else:
        ret = L_spark.collect()
    sc.stop()
    
    ret = [[ret[i][0],ret[i][1]] for i in range(len(ret))]
    
    if translate:
        if feature_names is None:
            raise ValueError("Need Feature Names to Translate Features!")
        else:
            ret = [[ [feature_names[ret[i][0][j]] for j in range(len(ret[i][0]))] ,ret[i][1]] for i in range(len(ret))] 
            return ret
    else:
        return ret

def remove_duplicates(L,filter_length = 2):
    '''
    Takes an array-like object; only returns an array without duplicate values if the resulting array is of user-specified length.
    '''
    ret = []
    for i in range(len(L)):
        if L[i] not in ret:
            ret.append(L[i])
    if len(ret) == filter_length:
        return tuple(ret)

def generate_non_duplicate_paths(L,filter_length = 2):
    '''
    Takes an array-like object; returns all the unique sequences of values in the array of user-specified length. 
    '''
    ret = set()
    for i in range(len(L)):
        for j in range(i,len(L)):
            cur_path = remove_duplicates(L[i:j],filter_length)
            if cur_path is not None:
                ret.add(remove_duplicates(L[i:j],filter_length))
            
    return list(ret)

def remove_duplicate_combinations(ft_list):
    '''
    Removing possible duplicate combinations, i.e. same combination but in a different order, etc.
    '''
    ret = [list(item) for item in list(set([tuple(sorted(list_)) for list_ in ft_list]))]
    return ret

def replace_varname_to_id(path_list):
    '''
    Customized helper function, takes labels for continuous features and changes them into corresponding ID labels. 
    '''
    ft_list = [item[0] for item in path_list]
    ret = deepcopy(ft_list)
    for i in range(len(ft_list)):
        current=ft_list[i]
        for j in range(len(current)):
            current_ft = current[j]
            if current_ft == 'bid_price':
                ret[i][j] = 'bid_id'
            elif current_ft == 'kadpage_image_profile_aesthetics':
                ret[i][j] = 'aesthetics_id'
            elif current_ft == 'kadpage_image_profile_quality':
                ret[i][j] = 'quality_id'
    return ret

############################
####### BIAS OUTPUTS #######
############################


def get_feature_bias(dat,target_ft,weight=None,multi_ft_cutoff=None):
    '''
    Given target feature, return the bias associated with using only this feature. 
    If target_ft is a list, create a new target feature 'TargetGroupId', based on the groupings created by the listed feature.
    
    weight:
        imp - weight by impression
        size - weight by sample size
        None - no weighting, use arithmetic mean
    
    multi_ft_cutoff:
        To prevent the resulting groups having too small a sample, filtering groups with less than this supplied number of samples. 
        
    If multi_ft_cutoff is not none, returns:
        (average mean bias using all groups, average mean bias using filtered groups, percentage of groups satisfying the cutoff)
    Otherwise, returns:
        (average mean bias using all groups)
        
    
    
    '''
    ## Grouping by Multiple Features, Create New Combined Feature
    dat = deepcopy(dat)
    if isinstance(target_ft,list):
        for target in target_ft:
            if target not in dat.columns:
                raise ValueError('Invalid Target.')
        dat["TargetGroupId"] = dat.groupby(target_ft).grouper.group_info[0]
        target_ft = "TargetGroupId"

    
    if target_ft not in dat.columns:
        raise ValueError('Invalid Target.')
        
    else:
        ret = []
        for i in range(len(np.unique(dat[target_ft]))):
            current = dat[dat[target_ft]==  np.unique(dat[target_ft])[i]]
            if len(current) == 0:
                continue
            
            ## Different Mean Calculations
            ## Weight by Impression
            if weight is not None:
                current_mean = total_ctr(current)
                
            ## Weight by None / Sample Size
            else:
                current_mean = np.mean(current['CTR'])
            
            current_bias = np.mean([abs((current["CTR"].iloc[j] - current_mean) / current["CTR"].iloc[j]) for j in range(len(current))])
            
            ## Different Returns
            ## Weight by Impression
            if weight == "weightcount":
                ret.append([current_bias,np.sum(current['weightcount']),len(current)])

            ## Weight by None
            else:
                ret.append([current_bias,len(current)])
            
        ret = pd.DataFrame(np.array(ret))
            
        if weight is not None:
            
            ## Regular Bias
            w_avg1 = 0
            count1 = sum(ret[1])
            if multi_ft_cutoff is not None:
                w_avg2 = 0
                if weight == "imp":
                    count2 = sum(ret[ret[2] >= multi_ft_cutoff][1])
                    ratio = len(ret[ret[2] >= multi_ft_cutoff])/len(ret)
                else:
                    count2 = sum(ret[ret[1] >= multi_ft_cutoff][1])      
                    ratio = len(ret[ret[1] >= multi_ft_cutoff])/len(ret)
                                  
            for i in range(len(ret)):
                current = ret.iloc[i,:]
                w_avg1 += (current[1]/count1) * current[0]
                if multi_ft_cutoff is not None:
                    if current[2] >= multi_ft_cutoff:
                        w_avg2 += (current[1]/count2) * current[0]
            
            if multi_ft_cutoff is not None:
                if w_avg2 is not None:
                    return (w_avg1,w_avg2,ratio)
                else:
                    return (w_avg1,"None",ratio)
            else:
                return w_avg1

        else:
            if multi_ft_cutoff is not None:
                return (np.mean(ret[0]),np.mean(ret[ret[1] >= multi_ft_cutoff][0]),len(ret[ret[1] >= multi_ft_cutoff])/len(ret))
            else:
                np.mean(ret[0])
        
def get_all_feature_bias(dat, features,weight = "weightcount",verbose=False,multi_ft_cutoff = None):
    '''
    Given a list of features, return a list of the bias associated with only using each of the feature.
    
    weight:
        imp - weight by impression
        size - weight by sample size
        None - no weighting, use arithmetic mean
    
    multi_ft_cutoff:
        To prevent the resulting groups having too small a sample, filtering groups with less than this supplied number of samples. 
        
    If multi_ft_cutoff is not none, returns a list of :
        (average mean bias using all groups, average mean bias using filtered groups, percentage of groups satisfying the cutoff)
    Otherwise, returns a list of:
        (average mean bias using all groups)
    '''
    ret = []
    for name in features:
        ft_bias = get_feature_bias(dat,name,weight,multi_ft_cutoff)
        if verbose:
            print("Feature: "+name+" || Feature Bias: " + str(ft_bias))
        if multi_ft_cutoff is not None:
            ret.append([name, ft_bias[0],ft_bias[1],ft_bias[2]])
        else:
            ret.append([name, ft_bias])
    
    if weight is not None:
        if multi_ft_cutoff is not None:
            ret.append(['global',np.mean(abs((dat['CTR']-total_ctr(dat))/dat['CTR'])),np.mean(abs((dat['CTR']-total_ctr(dat))/dat['CTR'])),1])
        else:
            ret.append(['global',np.mean(abs((dat['CTR']-total_ctr(dat))/dat['CTR']))])
    else:
        if multi_ft_cutoff is not None:
            ret.append(['global',np.mean(abs((dat['CTR']-np.mean(dat['CTR']))/dat['CTR'])),np.mean(abs((dat['CTR']-np.mean(dat['CTR']))/dat['CTR'])),1])
        else:
            ret.append(['global',np.mean(abs((dat['CTR']-np.mean(dat['CTR']))/dat['CTR']))])
    
    ret = pd.DataFrame(ret)
    
    if multi_ft_cutoff is not None:
        ret.rename(columns = {0:"Path",1:"Global",2:"Filtered",3:"Percentage of Selected Groups"},inplace=True)
    else:
        ret.rename(columns = {0:"Path",1:"Global"},inplace=True)

    return ret