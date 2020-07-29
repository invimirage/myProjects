import pandas as pd
a = {'a':[0,0,0], 'b':[0, 0, 0]}
b = {'b': [0,0,0]}
df = pd.DataFrame(a)
print(df)
feat_list = 'ds,\
       adid,\
       1_clk,\
       1_kcontextsidefeats_devicecontext_deviceos,\
       1_kcontextsidefeats_devicecontext_deviceconn,\
       1_kadsidefeats_adgroupid,\
       1_kusersidefeats_mpuserinfo_userage,\
       1_kusersidefeats_mpuserinfo_usergender,\
       1_kusersidefeats_mpuserinfo_citycode,\
       1_kadsidefeats_adgroupinfo_producttype,\
       1_kadsidefeats_adgroupinfo_mpadinfo_crtsize,\
       1_kadsidefeats_adgroupinfo_mpadinfo_advertisercat,\
       1_kadsidefeats_adgroupinfo_mpadinfo_advertiserid,\
       1_kadsidefeats_adgroupinfo_mpadinfo_iscanvas,\
       1_kadsidefeats_adgroupinfo_mpadinfo_primaryindustry,\
       1_kadsidefeats_adgroupinfo_mpadinfo_secondaryindustry,\
       1_kadsidefeats_expansionhittargetingmask,\
       1_kusersidefeats_sitefriendsclktwomonth,\
       1_kusersidefeats_sitefriendsclkurltwomonth,\
       1_kusersidefeats_sitefriendsclkheadtwomonth,\
       1_kusersidefeats_sitefriendsclkimgtwomonth,\
       1_kusersidefeats_sitefriendsclkvaluetwomonth,\
       1_kusersidefeats_sitefriendsclkthreemonth,\
       1_kusersidefeats_sitefriendsclkurlthreemonth,\
       1_kusersidefeats_sitefriendsclkheadthreemonth,\
       1_kusersidefeats_sitefriendsclkimgthreemonth,\
       1_kusersidefeats_sitefriendsclkvaluethreemonth,\
       1_kusersidefeats_sitefriendsclksixmonth,\
       1_kusersidefeats_sitefriendsclkurlsixmonth,\
       1_kusersidefeats_sitefriendsclkheadsixmonth,\
       1_kusersidefeats_sitefriendsclkimgsixmonth,\
       1_kusersidefeats_sitefriendsclkvaluesixmonth,\
       1_kadsidefeats_adgroupinfo_mpadinfo_checktype,\
       1_kusersidefeats_appinstallactivefeature_monthisinstall,\
       1_kusersidefeats_appinstallactivefeature_monthisactive,\
       1_kusersidefeats_appinstallactivefeature_monthmaxusedays,\
       1_kadsidefeats_adgroupinfo_mpadinfo_jdproductinfo_itemid,\
       1_kadsidefeats_adgroupinfo_mpadinfo_jdproductinfo_shopid,\
       1_kusersidefeats_activegameuser,\
       1_kadsidefeats_templateid_interest1,\
       1_kadsidefeats_templateid_interest2,\
       1_kadsidefeats_templateid_interest3,\
       1_kadsidefeats_templateid_interest4,\
       1_kadsidefeats_templateid_interest5,\
       1_kadsidefeats_templateid_interest6,\
       1_kadsidefeats_crtmd5,\
       1_kusersidefeats_clkonemonth,\
       1_kusersidefeats_clkheadonemonth,\
       1_kusersidefeats_clkvalueonemonth,\
       1_kusersidefeats_clkurlonemonth,\
       1_kusersidefeats_clkimgonemonth,\
       1_kusersidefeats_clkfollowonemonth,\
       1_kadsidefeats_productid,\
       1_kusersidefeats_wechatusertag_interest1,\
       1_kusersidefeats_wechatusertag_interest2,\
       1_kusersidefeats_wechatusertag_interest3,\
       1_kusersidefeats_wechatusertag_interest4,\
       1_kusersidefeats_wechatusertag_interest5,\
       1_kusersidefeats_wechatusertag_interest6,\
       1_kusersidefeats_appinterestwuid1,\
       1_kusersidefeats_appinterestwuid2,\
       1_kusersidefeats_appinterestwuid3,\
       1_kusersidefeats_appinterestwuid4,\
       1_kusersidefeats_appinterestwuid5,\
       1_kusersidefeats_appinterestwuid6,\
       1_kusersidefeats_commercialinterestnew1,\
       1_kusersidefeats_commercialinterestnew2,\
       1_kusersidefeats_commercialinterestnew3,\
       1_kusersidefeats_commercialinterestnew4,\
       1_kusersidefeats_commercialinterestnew5,\
       1_kusersidefeats_commercialinterestnew6,\
       1_kusersidefeats_wechatsubscriptioninterest1,\
       1_kusersidefeats_wechatsubscriptioninterest2,\
       1_kusersidefeats_wechatsubscriptioninterest3,\
       1_kusersidefeats_wechatsubscriptioninterest4,\
       1_kusersidefeats_wechatsubscriptioninterest5,\
       1_kusersidefeats_wechatsubscriptioninterest6,\
       1_kusersidefeats_userimgsimiinterest1,\
       1_kusersidefeats_userimgsimiinterest2,\
       1_kusersidefeats_userimgsimiinterest3,\
       1_kusersidefeats_userimgsimiinterest4,\
       1_kusersidefeats_userimgsimiinterest5,\
       1_kusersidefeats_userimgsimiinterest6,\
       1_kusersidefeats_wxfrienduseradtaginterest1,\
       1_kusersidefeats_wxfrienduseradtaginterest2,\
       1_kusersidefeats_wxfrienduseradtaginterest3,\
       1_kusersidefeats_wxfrienduseradtaginterest4,\
       1_kusersidefeats_wxfrienduseradtaginterest5,\
       1_kusersidefeats_wxfrienduseradtaginterest6,\
       1_kusersidefeats_usertidmd5interest1,\
       1_kusersidefeats_usertidmd5interest2,\
       1_kusersidefeats_usertidmd5interest3,\
       1_kusersidefeats_usertidmd5interest4,\
       1_kusersidefeats_usertidmd5interest5,\
       1_kusersidefeats_usertidmd5interest6,\
       1_kadsidefeats_jdproductinfo_itemcategory_interest1,\
       1_kadsidefeats_jdproductinfo_itemcategory_interest2,\
       1_kadsidefeats_jdproductinfo_itemcategory_interest3,\
       1_kadsidefeats_jdproductinfo_itemcategory_interest4,\
       1_kadsidefeats_jdproductinfo_itemcategory_interest5,\
       1_kadsidefeats_jdproductinfo_itemcategory_interest6,\
       1_kadsidefeats_materialid_interest1,\
       1_kadsidefeats_materialid_interest2,\
       1_kadsidefeats_materialid_interest3,\
       1_kadsidefeats_materialid_interest4,\
       1_kadsidefeats_materialid_interest5,\
       1_kadsidefeats_materialid_interest6,\
       1_kusersidefeats_grade,\
       1_kusersidefeats_profession,\
       1_kusersidefeats_bindcard,\
       1_kusersidefeats_paymentstatus,\
       1_kusersidefeats_consumeid,\
       1_kusersidefeats_professionlevel,\
       1_kusersidefeats_marriageall_interest1,\
       1_kusersidefeats_marriageall_interest2,\
       1_kusersidefeats_marriageall_interest3,\
       1_kusersidefeats_marriageall_interest4,\
       1_kusersidefeats_marriageall_interest5,\
       1_kusersidefeats_marriageall_interest6,\
       1_kusersidefeats_assets_interest1,\
       1_kusersidefeats_assets_interest2,\
       1_kusersidefeats_assets_interest3,\
       1_kusersidefeats_assets_interest4,\
       1_kusersidefeats_assets_interest5,\
       1_kusersidefeats_assets_interest6,\
       1_kadsidefeats_ec_price,\
       1_kusersidefeats_productid_interest1,\
       1_kusersidefeats_productid_interest2,\
       1_kusersidefeats_productid_interest3,\
       1_kusersidefeats_productid_interest4,\
       1_kusersidefeats_productid_interest5,\
       1_kusersidefeats_productid_interest6,\
       1_kusersidefeats_locardkeyword_interest1,\
       1_kusersidefeats_locardkeyword_interest2,\
       1_kusersidefeats_locardkeyword_interest3,\
       1_kusersidefeats_locardkeyword_interest4,\
       1_kusersidefeats_locardkeyword_interest5,\
       1_kusersidefeats_locardkeyword_interest6,\
       1_kusersidefeats_locardcategory_interest1,\
       1_kusersidefeats_locardcategory_interest2,\
       1_kusersidefeats_locardcategory_interest3,\
       1_kusersidefeats_locardcategory_interest4,\
       1_kusersidefeats_locardcategory_interest5,\
       1_kusersidefeats_locardcategory_interest6,\
       1_kusersidefeats_fituserpaytag_interest1,\
       1_kusersidefeats_fituserpaytag_interest2,\
       1_kusersidefeats_fituserpaytag_interest3,\
       1_kusersidefeats_fituserpaytag_interest4,\
       1_kusersidefeats_fituserpaytag_interest5,\
       1_kusersidefeats_fituserpaytag_interest6,\
       1_kadsidefeats_corporationhash,\
       1_kadsidefeats_corporationgrouplabelid,\
       1_kadsidefeats_origindesturlhash,\
       1_kusersidefeats_wxuserptboconvinterest1,\
       1_kusersidefeats_wxuserptboconvinterest2,\
       1_kusersidefeats_wxuserptboconvinterest3,\
       1_kusersidefeats_wxuserptboconvinterest4,\
       1_kusersidefeats_wxuserptboconvinterest5,\
       1_kusersidefeats_wxuserptboconvinterest6,\
       1_kadsidefeats_age_secondaryindustry_cross,\
       1_kadsidefeats_gender_secondaryindustry_cross,\
       1_kadsidefeats_age_gender_secondaryindustry_cross,\
       1_kadsidefeats_age_primaryindustry_cross,\
       1_kadsidefeats_gender_primaryindustry_cross,\
       1_kadsidefeats_age_gender_primaryindustry_cross,\
       1_kadsidefeats_age_crtsize_cross,\
       1_kadsidefeats_gender_crtsize_cross,\
       1_kadsidefeats_age_gender_crtsize_cross,\
       1_kusersidefeats_eccategorythirdinterest1,\
       1_kusersidefeats_eccategorythirdinterest2,\
       1_kusersidefeats_eccategorythirdinterest3,\
       1_kusersidefeats_eccategorythirdinterest4,\
       1_kusersidefeats_eccategorythirdinterest5,\
       1_kusersidefeats_eccategorythirdinterest6,\
       1_kusersidefeats_eccategoryfourthinterest1,\
       1_kusersidefeats_eccategoryfourthinterest2,\
       1_kusersidefeats_eccategoryfourthinterest3,\
       1_kusersidefeats_eccategoryfourthinterest4,\
       1_kusersidefeats_eccategoryfourthinterest5,\
       1_kusersidefeats_eccategoryfourthinterest6,\
       1_kusersidefeats_wxuserapplistinterest1,\
       1_kusersidefeats_wxuserapplistinterest2,\
       1_kusersidefeats_wxuserapplistinterest3,\
       1_kusersidefeats_wxuserapplistinterest4,\
       1_kusersidefeats_wxuserapplistinterest5,\
       1_kusersidefeats_wxuserapplistinterest6,\
       1_kusersidefeats_originaldevicemodel,\
       1_kusersidefeats_devicealias,\
       1_kusersidefeats_modelsize,\
       1_kusersidefeats_expmemory,\
       1_kusersidefeats_camtype,\
       1_kusersidefeats_corecnt,\
       1_kusersidefeats_screenpixel,\
       1_kusersidefeats_originalbrand,\
       1_kusersidefeats_ram,\
       1_kusersidefeats_rom,\
       1_kusersidefeats_gamepidtag_interest1,\
       1_kusersidefeats_gamepidtag_interest2,\
       1_kusersidefeats_gamepidtag_interest3,\
       1_kusersidefeats_gamepidtag_interest4,\
       1_kusersidefeats_gamepidtag_interest5,\
       1_kusersidefeats_gamepidtag_interest6,\
       1_kusersidefeats_locardintention_interest1,\
       1_kusersidefeats_locardintention_interest2,\
       1_kusersidefeats_locardintention_interest3,\
       1_kusersidefeats_locardintention_interest4,\
       1_kusersidefeats_locardintention_interest5,\
       1_kusersidefeats_locardintention_interest6,\
       1_kusersidefeats_wxuserfinanciallabel_interest1,\
       1_kusersidefeats_wxuserfinanciallabel_interest2,\
       1_kusersidefeats_wxuserfinanciallabel_interest3,\
       1_kusersidefeats_wxuserfinanciallabel_interest4,\
       1_kusersidefeats_wxuserfinanciallabel_interest5,\
       1_kusersidefeats_wxuserfinanciallabel_interest6,\
       1_pcvr,\
       2_producttype_corporationgrouplabelid'
feat_list = feat_list.split(',')
res = []
for feat in feat_list:
    feat = feat.strip(' ')
    org = feat
    if feat.split('_')[0] == '1' or feat.split('_')[0] == '2':
        feat = feat.split('_')[1:]
        feat = '_'.join(feat)
    newfeat = '{} as {}'.format(feat, org)
    res.append(newfeat)
sql = 'select \n {}'.format(',\n'.join(res))
print(sql)
# featlen = len(feat_list)
# for i in range(featlen):
#     for j in range(i+1, featlen):
#         feat_names = feat_list[i], feat_list[j]
#         conbine = '_'.join(map(lambda x: x.split('_')[-1], feat_names))
#         sql = "concat_ws(',', {}, {}) as {}".format(*feat_names, conbine)
#         res.append(sql)
# print(',\n'.join(res))

