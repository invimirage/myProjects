# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 20:04:23 2019

@author: haozhjiang
"""

import os
import datetime
import pandas as pd
import actoolkit.lgtplus as lgtplus
import actoolkit.routine as routine
import pickle
import shap

#### Getting File Name, Checking if the file has been loaded yet
dir_files = os.listdir()
today_date = datetime.date.today().strftime("%Y%m%d")

for file in dir_files:
    if today_date in file:
        file_name = file
    else:
        continue
        ### do something to delay the procedure

### Loading Data into Workspace
main_dat = pd.read_csv(file_name,sep=" ",header=None)

ft_name = [ "tid",
            #"xspace",
            "xdatetime",
            "CTR",
            "clk",
            "imp",
            "xad_crtsize",
            "gender",
            "age",
            "product_quality_id",
            "xad_cagegory1id",
            "xad_cagegory2id",
            "bid_price",
            "ds",
            "importds",
            "id",
            "adid",
            "updatedtime",
            "kadpage_image_url",
            "kadpage_image_ocr",
            "kadpage_image_profile_quality",
            "kadpage_image_profile_tone",
            "kadpage_image_profile_aesthetics",
            "kadpage_image_profile_md5",
            "kadpage_image_profile_similarclassid",
            "kadpage_image_profile_saturation",
            "kadpage_image_profile_light",
            "kadpage_image_profile_colorful",
            "kadpage_image_profile_colorlayout",
            "kadpage_image_profile_coloremotion",
            "kadpage_image_profile_scene_interest1",
            "kadpage_image_profile_scene_interest2",
            "kadpage_image_profile_scene_interest3",
            "kadpage_image_profile_scene_interest4",
            "kadpage_image_profile_scene_interest5",
            "kadpage_image_profile_scene_interest6",
            "kadpage_image_profile_style_interest1",
            "kadpage_image_profile_style_interest2",
            "kadpage_image_profile_style_interest3",
            "kadpage_image_profile_style_interest4",
            "kadpage_image_profile_style_interest5",
            "kadpage_image_profile_style_interest6",
            "kadpage_image_profile_color_interest1",
            "kadpage_image_profile_color_interest2",
            "kadpage_image_profile_color_interest3",
            "kadpage_image_profile_color_interest4",
            "kadpage_image_profile_color_interest5",
            "kadpage_image_profile_color_interest6",
            "kadpage_image_profile_object_interest1",
            "kadpage_image_profile_object_interest2",
            "kadpage_image_profile_object_interest3",
            "kadpage_image_profile_object_interest4",
            "kadpage_image_profile_object_interest5",
            "kadpage_image_profile_object_interest6",
            "kadpage_image_profile_logo_interest1",
            "kadpage_image_profile_logo_interest2",
            "kadpage_image_profile_logo_interest3",
            "kadpage_image_profile_logo_interest4",
            "kadpage_image_profile_logo_interest5",
            "kadpage_image_profile_logo_interest6",
            "kadpage_image_profile_texttags_interest1",
            "kadpage_image_profile_texttags_interest2",
            "kadpage_image_profile_texttags_interest3",
            "kadpage_image_profile_texttags_interest4",
            "kadpage_image_profile_texttags_interest5",
            "kadpage_image_profile_texttags_interest6",
            "kadpage_image_profile_starface_interest1",
            "kadpage_image_profile_starface_interest2",
            "kadpage_image_profile_starface_interest3",
            "kadpage_image_profile_starface_interest4",
            "kadpage_image_profile_starface_interest5",
            "kadpage_image_profile_starface_interest6",
            "kadpage_image_profile_clothestagging_interest1",
            "kadpage_image_profile_clothestagging_interest2",
            "kadpage_image_profile_clothestagging_interest3",
            "kadpage_image_profile_clothestagging_interest4",
            "kadpage_image_profile_clothestagging_interest5",
            "kadpage_image_profile_clothestagging_interest6",
            "kadpage_image_profile_bgcolor_interest1",
            "kadpage_image_profile_bgcolor_interest2",
            "kadpage_image_profile_bgcolor_interest3",
            "kadpage_image_profile_bgcolor_interest4",
            "kadpage_image_profile_bgcolor_interest5",
            "kadpage_image_profile_bgcolor_interest6",
            "kadpage_image_profile_fgcolor_interest1",
            "kadpage_image_profile_fgcolor_interest2",
            "kadpage_image_profile_fgcolor_interest3",
            "kadpage_image_profile_fgcolor_interest4",
            "kadpage_image_profile_fgcolor_interest5",
            "kadpage_image_profile_fgcolor_interest6"]

main_dat.columns=ft_name

## Load Model and Categorical Mapping
lgr_model,cat_idname_mapping,cat_new_id_mapping,bid_bins = pickle.load(open("lgt_model.p", "rb")) ## Load Model
feature_names = lgr_model._Booster.dump_model()['feature_names']

## Get New ID Mapping
for category in cat_idname_mapping.keys():
    main_dat[cat_idname_mapping[category]] = routine.get_new_ID(main_dat,cat_new_id_mapping[category],id_type = category)
    
main_dat['bid_id'] = routine.get_bid_id(main_dat, bid_bins)

## Get SHAP Value
lm_ex = shap.TreeExplainer(lgr_model)
lm_ex_shap = lm_ex.shap_values(main_dat[feature_names],main_dat['CTR'])
## Get SHAP Ranking
shap_ranking = routine.get_shap_ranking(lm_ex_shap, feature_names)

## Get Top-K SHAP Features
topKSHAP_Negative, topKSHAP_Positive = routine.get_top_k_shap(lm_ex_shap, feature_names, k = 5)

## Get Leaf_ID
leaf_id = lgtplus.get_node_id(main_dat,lgr_model,feature_names)
