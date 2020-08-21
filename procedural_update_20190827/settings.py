# 应用组
group_name = 'g_teg_weixin_profile_teg_datamining_wx_dm'
# 集群
cluster_name = 'ss-shuiping'
# 申请资源
properties = {
    'spark.executor.cores': 4,
    'spark.executor.memory': '4g',
    'spark.executor.instances': 15
}

# 获取数据的来源数据库，表，分区
db, tb = 'wx_ad_fs', 'feature_selection_wx_friends_cvr_total_6110000001'
# 分区有三种格式，或是一个整数表示日期，或是tuple表示开始日期、结束日期（均包含）、日期选取间隔，或是一个数组表示选取多个日期
# 日期的格式必须如下YYYYMMDD
# parts = 20200729
parts = (20200811, 20200818, 1)
# parts = [20200729, 20200730]

# Join为从广告画像表直接获取广告侧特征，并与离线日志表join得到cvr等数据；填入任何其他值使用离线日志表直接导出数据
mode = 'join'

# 如果为join模式。设置广告画像表名
adfeat_table = 'adprofile_feature_wx'

# 希望研究的所有特征
# ad_feats = [
#             'producttype',
#             'kadsidefeats_adgroupinfo_mpadinfo_iscanvas',
#             'ams_second_industry',
#             'kadsidefeats_adgroupinfo_mpadinfo_crtsize',
#             'kadsidefeats_origindesturlhash',
#             'kadsidefeats_corporationgrouplabelid',
#             'kadsidefeats_adgroupinfo_mpadinfo_checktype',
#            ]
ad_feats = """
       kcommon_crtsize,
       kcommon_producttype,
       kcommon_bidobjective,
       kcommon_secondbidobjective,
       kcommon_bidprice,
       kcommon_secondbidprice,
       kcommon_adstatus,
       kcommon_campaignid,
       kcommon_platform,
       kcommon_cates,
       kcommon_words,
       kcommon_topic,
       kcommon_adposid,
       kadvertiser_adverid,
       kadvertiser_amsfirstindustryid,
       kadvertiser_amssecondindustryid,
       kadvertiser_corporation,
       kadvertiser_corporationgroup,
       kadvertiser_spidv2,
       kadpage_text_profile_brand,
       kadpage_text_profile_product,
       kadpage_text_profile_prop,
       kadpage_text_profile_locard_cates,
       kadpage_text_profile_locard_emb,
       kadpage_text_profile_locard_keywords,
       kadpage_text_profile_selltag_rebate,
       kadpage_text_profile_selltag_price,
       kadpage_text_profile_selltag_discount,
       kadpage_text_profile_textsellpointsv2,
       kadpage_text_profile_jdcates,
       kadpage_multimodal_keyword,
       kadpage_image_profile_quality,
       kadpage_image_profile_tone,
       kadpage_image_profile_aesthetics,
       kadpage_image_profile_md5,
       kadpage_image_profile_similarclassid,
       kadpage_image_profile_saturation,
       kadpage_image_profile_light,
       kadpage_image_profile_colorful,
       kadpage_image_profile_colorlayout,
       kadpage_image_profile_coloremotion,
       kadpage_image_profile_ocrlen,
       kadpage_image_profile_ocrparagraphnum,
       kadpage_image_profile_ocrarearate,
       kadpage_image_profile_dismodelarearate,
       kadpage_image_profile_disitemarearate,
       kadpage_image_profile_distitlearearate,
       kadpage_image_profile_scene,
       kadpage_image_profile_style,
       kadpage_image_profile_color,
       kadpage_image_profile_object,
       kadpage_image_profile_logo,
       kadpage_image_profile_texttags,
       kadpage_image_profile_starface,
       kadpage_image_profile_clothestagging,
       kadpage_image_profile_bgcolor,
       kadpage_image_profile_fgcolor,
       kadpage_image_profile_multitags,
       klpage_page_type,
       klpage_page_landingpageid,
       klpage_page_common_cates,
       klpage_page_common_words,
       klpage_page_common_topic,
       klpage_page_fyprofile_headpicscnt,
       klpage_page_fyprofile_detailpicscnt,
       klpage_page_fyprofile_headpicsautochange,
       klpage_page_fyprofile_isdetailvideo,
       klpage_page_fyprofile_iscomments,
       klpage_page_fyprofile_pkgcnt,
       klpage_page_fyprofile_hasphoneconsult,
       klpage_page_fyprofile_hastextbooking,
       klpage_page_fyprofile_hasorderdir,
       klpage_page_fyprofile_hasbuyindex,
       klpage_page_fyprofile_minprice,
       klpage_page_fyprofile_maxprice,
       kitempage_productid,
       kitempage_item_ecommerceinfo_textclass,
       kitempage_item_ecommerceinfo_dpcategory,
       kitempage_item_ecommerceinfo_dpbrand,
       kitempage_item_ecommerceinfo_dpproduct,
       kitempage_item_ecommerceinfo_dpcluster,
       kitempage_item_ecommerceinfo_dppropvalue,
       kitempage_item_appinfo_score,
       kitempage_item_appinfo_pkgsize,
       kitempage_item_appinfo_category,
       kitempage_item_appinfo_appname,
       kitempage_item_appinfo_pkgmd5,
       kitempage_item_appinfo_pkgname,
       kitempage_item_appinfo_downloadcount,
       kitempage_item_educationinfo_educationskuinfo_skustage,
       kitempage_item_educationinfo_educationskuinfo_skucourse,
       kitempage_item_educationinfo_educationskuinfo_skuprice,
       kitempage_item_gameinfo_vestid,
       kitempage_item_gameinfo_gametype,
       kitempage_item_gameinfo_gamecore,
       kitempage_item_gameinfo_gamebattletype,
       kitempage_item_gameinfo_gametheme,
       kitempage_item_gameinfo_iptype,
       kitempage_item_gameinfo_markettype,
       kitempage_item_gameinfo_wxgamevideoendpagetype,
       kitempage_item_gameinfo_cvtagging,
       kitempage_item_baseinfo_industry,
       kitempage_item_baseinfo_amsfirstcategory,
       kitempage_item_baseinfo_amssecondcategory,
       kitempage_item_baseinfo_amsthirdcategory,
       kitempage_item_baseinfo_amsfourthcategory,
       kitempage_item_baseinfo_type,
       kitempage_item_baseinfo_productid,
       kitempage_item_baseinfo_libraryid,
       kitempage_item_baseinfo_sourceid,
       kitempage_item_baseinfo_price,
       kitempage_item_baseinfo_brandname,
       kitempage_item_baseinfo_saleprice
       """
context_feats = [
                 'kcontextsidefeats_devicecontext_deviceos',
                 'kcontextsidefeats_devicecontext_deviceconn',
                 'kadsidefeats_expansionhittargetingmask'
                ]
user_feats = [
              'kusersidefeats_mpuserinfo_userage',
              'kusersidefeats_mpuserinfo_usergender',
              'kusersidefeats_grade',
              'kusersidefeats_profession',
              'kusersidefeats_bindcard',
              'kusersidefeats_paymentstatus',
              'kusersidefeats_consumeid' 
             ]

# 希望被交叉的特征
cross_feats = ['kadvertiser_corporation', 'kadvertiser_amssecondindustryid', 'kadpage_image_profile_similarclassid', 'kadvertiser_adverid', 'kcommon_producttype',
           'kcommon_bidobjective', 'kcommon_secondbidobjective', 'kadpage_image_profile_md5', 'kadpage_text_profile_prop', 'kadvertiser_corporationgroup',
           'kadpage_image_profile_scene']

# 连续型变量
continuous_feats = [
        'kadpage_image_profile_quality',
        'kadpage_image_profile_aesthetics',
        'kitempage_item_baseinfo_price',
        'kcommon_bidprice'
    ]

bins = 20

cols = ['clk', 'pcvr', 'cvr', 'pcvrbias', 'weightcount']

# 多少阶交叉（目前可视化和数据分析只支持二阶交叉）
rank = 2

# 设置用于LightGBM模型测试的数据表名称，None为不需要进行模型测试
LGBM_train = 'LGBM_test'

# 存放交叉结果（csv文件）的目录，可以设置多级
datadir = 'Data'

# 只有转化（研究cvr）或点击（研究ctr）超过阈值的特征值，才会被记录
# Wilson区间给出，若bias比较有意义，转化数量基本上要超过一个定值，约在70到80之间
threshold = 10

# 以下用于可视化部分
# 选择转化量的分组情况，如[0, 10, 100]代表分为3组：[0, 10), [10, 100), [100, ∞)
# 将特征的每个维度的转化量按照该分组，计算每一组的加权平均bias绝对值
trans_group = [0, 3, 10, 20, 40, 65, 100, 1000]

# 选择需要详细分析的特征数量
topn = 20

# 保存可视化图片的路径
figuredir = 'figures'

# 全部特征的可视化结果
allname = 'features.png'

# 详细分析特征的可视化结果
detailname = 'features_detailed_bias.png'

import datetime
def get_parts()->list:
    if type(parts) == int:
        res = ['p_' + str(parts)]
    elif type(parts) == tuple:
        time = parts
        res = []
        cur = datetime.datetime.strptime(str(time[0]), '%Y%m%d')
        while True:
            temp = int(cur.strftime('%Y%m%d'))
            if temp <= time[1]:
                res.append(temp)
            else:
                break
            cur = cur + datetime.timedelta(time[2])
        res = list(map(lambda x: 'p_' + str(x), res))
    elif type(parts) == list:
        res = list(map(lambda x: 'p_' + str(x), parts))
    else:
        print('parts error!')
        res = ['p_20200730']
    return res

def parse_key(key, joiner=','):
    """
    将特征的名称处理的更加简洁
    输入: 特征名称string，如果是交叉特征选择特征连接方式
    输出: 处理之后的特征名称
    """
    if (key.split(','))[0] == key:
        return key.split('_')[-1]
    return joiner.join(map(lambda x: x.split('_')[-1], key.split(',')))


def create_test(key, fromtb='wx_ad_fs::feature_selection_wx_friends_cvr_total_6110000001', totb='t_rayerfzhang_cross_features_adid'):
    ds = get_parts()
    ds = ' or '.join(list(map(lambda x: 'ds = ' + x.split('_')[-1], ds)))
    all_feature = ad_feats + context_feats + user_feats
    sqls = []
    key1, key2 = key.split(',')
    key1 = key1.strip(' ')
    key2 = key2.strip(' ')
    if key1 not in all_feature or key2 not in all_feature:
        return 'Feature name is missing'
    if key1 in ad_feats and key2 in ad_feats:
        key = 'ad'
    elif key1 in user_feats and key2 in user_feats:
        key = 'user'
    if key in ['ad', 'user']:
        cross_feats = []
        l = len(all_feature)
        for i in range(l):
            for j in range(i + 1, l):
                cross_feats.append(','.join([all_feature[i], all_feature[j]]))
        for feat in cross_feats:
            if key == 'ad' and (feat.split(',')[0] not in ad_feats or feat.split(',')[1] not in ad_feats):
                continue
            if key == 'user' and (feat.split(',')[0] not in user_feats or feat.split(',')[1] not in user_feats):
                continue
            sqls.append("   concat_ws(',', {}, {}) as {}".format(*(feat.split(',')), parse_key(feat, '_')))
        res = ',\n'.join(sqls)
        if key == 'ad':
            key_val = 'adid'
        else:
            key_val = 'uin'
        res = 'select distinct\n   {},\n'.format(key_val) + res + '\nfrom {}\nwhere {};'.format(fromtb, ds)
    else:
        temp = '   select distinct uin, {} as cleft, {} as cright\n'.format(key1, key2) + \
               '   from %s\n'%fromtb + \
               '   where %s\n'%ds
        res = "select uniqueuin.uin, concat_ws(',', cleft, cright) as cross_feature\n" + \
        "from\n(\n" + temp + ") temptable\njoin\n" + \
        '\n   '.join(("(\nselect uin\nfrom\n(\n" + temp + ")\ngroup by uin\nhaving count(uin) = 1").split('\n')) + \
        "\n)\nuniqueuin\non temptable.uin = uniqueuin.uin;"
    res = 'use wx_dm;\n' + \
          'drop table {0};\n'.format(totb) + \
          'create table {0} as\n'.format(totb) + res
    return res

from xml.dom.minidom import parse
import xml.dom
rootdir="hdfs://ss-teg-3-v2/stage/outface/TEG/g_teg_weixin_profile_teg_datamining_wx_dm/cross_feature_explore/"
def build_xml(cross_name, filename='fs_template.xml', basetb='wx_ad_fs::feature_selection_wx_friends_cvr_total_6110000001', newtb='t_rayerfzhang_cross_features_adid', targettb='t_rayerfzhang_cross'):
    key1, key2 = cross_name.split(',')
    key1 = key1.strip(' ')
    key2 = key2.strip(' ')
    side = ''
    if key1 in ad_feats and key2 in ad_feats:
        side = 'ad'
    cross_name = parse_key(cross_name, '_')
    if side == 'ad':
        keyname = 'adid'
    else:
        keyname = 'uin'
    parts = ','.join(get_parts())
    # 读取文件
    dom = parse(filename)
    # 获取文档元素对象
    data = dom.documentElement
    # 获取 student
    tables = data.getElementsByTagName('table')
    basetable = tables[0]
    if len(basetb.split('::')) == 2:
        basetb = basetb.split('::')[1]
    basetable.setAttribute('name', basetb)
    basetable.setAttribute('partitions', parts)
    newtable = tables[1]
    newtable.setAttribute('name', newtb)
    targettable = tables[2]
    targettable.setAttribute('name', targettb + '_' + cross_name)
    targettable.setAttribute('partitions', parts)
    transformtable = tables[3]
    transformtable.setAttribute('partitions', parts)
    criteriatable = tables[4]
    criteriatable.setAttribute('partitions', parts)
    binning = data.getElementsByTagName('binning')[0]
    binning.setAttribute('name', '^2_.*')
    binning.setAttribute('type', 'S')
    values = data.getElementsByTagName('value')
    newval = values[1]
    newval.setAttribute('name', cross_name)
    newval.setAttribute('exclude', 'uin, adid')
    keys = data.getElementsByTagName('key')
    basekey = keys[0]
    newkey = keys[1]
    basekey.setAttribute('name', keyname)
    newkey.setAttribute('name', keyname)
    root = data
    root.setAttribute('dir', rootdir + cross_name)
    f = open('fs_{}.xml'.format(cross_name), 'w+')
    data.writexml(f)
    f.close()