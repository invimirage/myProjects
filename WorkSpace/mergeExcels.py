import os
from math import *

import pandas as pd
from scipy.stats import chi2_contingency

# pip install black -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
dir = 'D:\Data'
filelist = os.listdir(dir)
print(filelist)
avgBiasDict = {}
weightedBiasDict = {}
stdErrorDict = {}
chiSquareDict = {}
changesDict = {}
sigleDoc = {}
crossFeat = []
crossFeatNames = {'product_type': 'pro', 'primary_industry_id': 'pri', 'secondary_industry_id': 'sec', 'advertiser_category': 'cat',
                   'destiny_url': 'url', 'group_id': 'group', 'is_canvas': 'isc', 'expansion_hit_targeting_mask': 'exp', 'vest_id': 'vest', 'creative_size': 'crt', 'check_type': 'check'}

def parse_sigle(mydata, key):
    # if key.find('pro') == -1 and key.find('pri') == -1:
    #     return
    val = 0
    valsquare = 0
    valsum = 0
    datalen = len(mydata['pcvrbias'])
    # for chi-square test
    table = [[], []]
    if len(mydata['weight_count']) < datalen:
        datalen = len(mydata['weight_count'])
    tagnum = -1
    for col in mydata.columns:
        if col == 'pcvr':
            break
        tagnum += 1

    tagname = mydata.columns[tagnum]
    sigleDoc[key] = {}
    tagnames = {}
    for i in range(datalen):
        try:
            weightCount = float(mydata['weight_count'][i])
        except ValueError:
            continue
        cvr = float(mydata['cvr'][i])
        if cvr is None or cvr != cvr:
            continue
        if weightCount < 10000 or weightCount != weightCount:
            continue
        val += cvr
        valsquare += pow(cvr, 2)
        valsum += 1
        trans = weightCount * cvr / (1 + cvr)
        if trans <= 50:
            continue
        tag = mydata[tagname][i]
        try:
            tag = int(tag)
        except:
            pass
        tagnames[tag] = len(table[0])
        click = weightCount / (1 + cvr)
        table[0].append(click)
        table[1].append(trans)
    chi, _, dof, _ = chi2_contingency(table)
    if dof > 0:
        chiSquareDict[key] = chi / sqrt(dof)
    sigleDoc[key]['tags'] = tagnames
    sigleDoc[key]['table'] = table
    sigleDoc[key]['sum'] = sum([sum(table[i]) for i in range(len(table))])
    stdErrorDict[key] = sqrt((valsquare / valsum) - pow((val / valsum), 2))

    val = 0
    valsum = 0
    datalen = len(mydata['pcvrbias'])
    if len(mydata['weight_count']) < datalen:
        datalen = len(mydata['weight_count'])
    for i in range(datalen):
        try:
            weight = float(mydata['weight_count'][i])
        except ValueError:
            continue
        bias = float(mydata['pcvrbias'][i])
        if bias is None or bias == -1.0 or bias != bias:
            continue
        if weight < 10000:
            continue
        val += abs(bias) * weight
        valsum += weight
    weightedBiasDict[key] = val / valsum

def parse_cross(mydata, key):
    val = 0
    valsquare = 0
    valsum = 0
    datalen = len(mydata['pcvrbias'])
    # for chi-square test
    table = [[], []]
    if len(mydata['weight_count']) < datalen:
        datalen = len(mydata['weight_count'])
    # 记录cross feature的两个父关键词
    f_keys = ['', '']

    #
    # if key != 'cat_exp':
    #     return

    parts = key.split('_')
    key_count = 2
    for fkey, skey in crossFeatNames.items():
        if parts[0].find(skey) > -1:
            f_keys[0] = fkey
            key_count -= 1
        elif parts[1].find(skey) > -1:
            f_keys[1] = fkey
            key_count -= 1
        if key_count == 0:
            break
    # if len(f_keys) < 2:
    #     print(parts)
    tagnum = -1
    for col in mydata.columns:
        if col == 'pcvr':
            break
        tagnum += 1
    t2, t1 = mydata.columns[tagnum], mydata.columns[tagnum-1]
    T1 = mydata[t1][0]
    change = 0
    total_ob = 0
    for i in range(datalen):
        if mydata[t1][i] == mydata[t1][i] and mydata[t1][i] != T1:
            T1 = mydata[t1][i]
        T2 = mydata[t2][i]
        try:
            weightCount = float(mydata['weight_count'][i])
        except ValueError:
            continue
        cvr = float(mydata['cvr'][i])
        if cvr is None or cvr != cvr:
            continue
        if weightCount < 10000 or weightCount != weightCount:
            continue
        val += cvr
        valsquare += pow(cvr, 2)
        valsum += 1
        trans = weightCount * cvr / (1 + cvr)
        if trans <= 50:
            continue
        click = weightCount / (1 + cvr)
        table[0].append(click)
        table[1].append(trans)
        try:
            T1 = int(T1)
        except:
            pass
        try:
            T2 = int(T2)
        except:
            pass
        try:
            D = sigleDoc[f_keys[0]]
            Pt1j1 = D['table'][0][D['tags'][T1]] / D['sum']
            Pt1j2 = D['table'][1][D['tags'][T1]] / D['sum']
            D = sigleDoc[f_keys[1]]
            Pt2j1 = D['table'][0][D['tags'][T2]] / D['sum']
            Pt2j2 = D['table'][1][D['tags'][T2]] / D['sum']
            # E1 = Pt1j1 * Pt2j1 * weightCount / (Pt1j1 * Pt2j1 + Pt1j2 * Pt2j2)
            # E2 = Pt1j2 * Pt2j2 * weightCount / (Pt1j1 * Pt2j1 + Pt1j2 * Pt2j2)
            O1 = trans
            O2 = click
            change += (O1 * (log(O1) - log(Pt1j1 * Pt2j1)) + O2 * (log(O2) - log(Pt1j2 * Pt2j2)) + \
                      weightCount * (log(Pt1j1 * Pt2j1 + Pt1j2 * Pt2j2) - log(weightCount)))
            total_ob += (O1 + O2)
        except KeyError:
            try:
                D = sigleDoc[f_keys[1]]
                Pt1j1 = D['table'][0][D['tags'][T1]] / D['sum']
                Pt1j2 = D['table'][1][D['tags'][T1]] / D['sum']
                D = sigleDoc[f_keys[0]]
                Pt2j1 = D['table'][0][D['tags'][T2]] / D['sum']
                Pt2j2 = D['table'][1][D['tags'][T2]] / D['sum']
                # E1 = Pt1j1 * Pt2j1 * weightCount / (Pt1j1 * Pt2j1 + Pt1j2 * Pt2j2)
                # E2 = Pt1j2 * Pt2j2 * weightCount / (Pt1j1 * Pt2j1 + Pt1j2 * Pt2j2)
                O1 = trans
                O2 = click
                change += (O1 * (log(O1) - log(Pt1j1 * Pt2j1)) + O2 * (log(O2) - log(Pt1j2 * Pt2j2)) + \
                           weightCount * (log(Pt1j1 * Pt2j1 + Pt1j2 * Pt2j2) - log(weightCount)))
                total_ob += (O1 + O2)
            except KeyError:
                pass



    chi, _, dof, _ = chi2_contingency(table)
    if dof > 0:
        chiSquareDict[key] = chi / sqrt(dof)

    stdErrorDict[key] = sqrt((valsquare / valsum) - pow((val / valsum), 2))
    changesDict[key] = change / total_ob
    val = 0
    valsum = 0
    datalen = len(mydata['pcvrbias'])
    if len(mydata['weight_count']) < datalen:
        datalen = len(mydata['weight_count'])
    for i in range(datalen):
        try:
            weight = float(mydata['weight_count'][i])
        except ValueError:
            continue
        bias = float(mydata['pcvrbias'][i])
        if bias is None or bias == -1.0 or bias != bias:
            continue
        if weight < 10000:
            continue
        val += abs(bias) * weight
        valsum += weight
    weightedBiasDict[key] = val / valsum

for fname in filelist:
    if fname[-3:] == 'csv':
        mydata = pd.read_csv(dir + '\\' + fname)
        key = fname[14:-4]

        if key[0] == '!':
            continue
        parse_sigle(mydata, key)
print(sigleDoc)
for fname in filelist:
    if fname[-3:] == 'csv':
        mydata = pd.read_csv(dir + '\\' + fname)
        key = fname[14:-4]
        if key[0] != '!':
            continue
        key = key[1:]
        crossFeat.append(key)
        parse_cross(mydata, key)
        # val = 0
        # valsquare = 0
        # valsum = 0
        # datalen = len(mydata['pcvrbias'])
        # # for chi-square test
        # table = [[], []]
        # if len(mydata['weight_count']) < datalen:
        #     datalen = len(mydata['weight_count'])
        # # 记录cross feature的两个父关键词
        # f_keys = []
        # if cross_mark == 1:
        #     parts = key.split('_')
        #     for fkey, skey in crossFeatNames.items():
        #         if skey.find(parts[0]) > -1 or skey.find(parts[1]) > -1:
        #             f_keys.append(fkey)
        #         if len(f_keys) == 2:
        #             break
        # for i in range(datalen):
        #     try:
        #         weightCount = float(mydata['weight_count'][i])
        #     except ValueError:
        #         continue
        #     cvr = float(mydata['cvr'][i])
        #     if cvr is None or cvr != cvr:
        #         continue
        #     if weightCount < 10000 or weightCount != weightCount:
        #         continue
        #     val += cvr
        #     valsquare += pow(cvr, 2)
        #     valsum += 1
        #     trans = weightCount * cvr / (1 + cvr)
        #     if trans <= 50:
        #         continue
        #     click = weightCount / (1 + cvr)
        #     table[0].append(click)
        #     table[1].append(trans)
        # chi, _, dof, _ = chi2_contingency(table)
        # if dof > 0:
        #     chiSquareDict[key] = chi / sqrt(dof)
        #
        # stdErrorDict[key] = sqrt((valsquare / valsum) - pow((val / valsum), 2))
        #
        # val = 0
        # valsum = 0
        # datalen = len(mydata['pcvrbias'])
        # if len(mydata['weight_count']) < datalen:
        #     datalen = len(mydata['weight_count'])
        # for i in range(datalen):
        #     try:
        #         weight = float(mydata['weight_count'][i])
        #     except ValueError:
        #         continue
        #     bias = float(mydata['pcvrbias'][i])
        #     if bias is None or bias == -1.0 or bias != bias:
        #         continue
        #     if weight < 10000:
        #         continue
        #     val += abs(bias) * weight
        #     valsum += weight
        # weightedBiasDict[key] = val / valsum
        #
        # # val = 0
        # # valsum = 0
        # # for i in mydata['Unnamed: 4'][1:]:
        # #     i = float(i)
        # #     if i == -1.0 or i != i:
        # #         continue
        # #     val += abs(i)
        # #     valsum += 1
        # # avgBiasDict[key] = val / valsum

avgBiasDict = list(sorted(avgBiasDict.items(), key=lambda x: x[1], reverse=True))
weightedBiasDict = list(sorted(weightedBiasDict.items(), key=lambda x: x[1], reverse=True))
# for fname, bias in avgBiasDict:
#     print('Feature:{:<35}PcvrBias Average:{:%}'.format(fname, bias))
# for fname, bias in weightedBiasDict:
#     print('Feature:{:<35}PcvrBias Weighted Average:{:%}'.format(fname, bias))
# stdErrorDict = list(sorted(stdErrorDict.items(), key=lambda x: x[1], reverse=True))
# for fname, stderr in stdErrorDict:
#     print('Feature:{:<35}Square Error:{}'.format(fname, stderr))
# chiSquareDict = list(sorted(chiSquareDict.items(), key=lambda x: x[1], reverse=True))
# for fname, pval in chiSquareDict:
#     print('Feature:{:<35}ChiSquare value:{}'.format(fname, pval))
print(crossFeat)
changesDict = dict(sorted(changesDict.items(), key=lambda  x: x[1], reverse=True))
for fname, change in changesDict.items():
    print('Feature:{:<35}Changed mutual-information value:{}'.format(fname, change))
# outData = {}
# for i in ['weightedAvgPcvrBias', 'cvrStandardError']:
#     outData[i] = {}
# for fname, stderr in stdErrorDict:
#     outData['cvrStandardError'][fname] = stderr
# for fname, bias in weightedBiasDict:
#     outData['weightedAvgPcvrBias'][fname] = bias
# outData = pd.DataFrame(outData)
# outData.to_csv('pcvr_bias_single_feature.csv')

weightedBiasDict = dict(weightedBiasDict)
chiSquareDict = dict(chiSquareDict)
stdErrorDict = dict(stdErrorDict)
maxBias = max(weightedBiasDict.values())
maxChi2 = max(chiSquareDict.values())
maxStderr = max(stdErrorDict.values())
print(maxBias, maxChi2, maxStderr)

stdErr = []
chi2 = []
weightedBias = []
keys = []
colors = []
for key in chiSquareDict.keys():
    keys.append(key)
    stdErr.append(stdErrorDict[key] / maxStderr)
    chi2.append(chiSquareDict[key] / maxChi2)
    weightedBias.append(weightedBiasDict[key] / maxBias)
    if key in crossFeat:
        colors.append('red')
    else:
        colors.append('black')
print(chi2, stdErr, weightedBias)

from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure(figsize=(50, 20), dpi=80)
x = np.arange(len(keys))
print(x)
width = 0.3
b1 = plt.bar(x - width, chi2, width, color='salmon', label='chi2')
b2 = plt.bar(x, weightedBias, width, color='orchid', label='bias')
b3 = plt.bar(x + width, stdErr, width, color='gray', label='stderr')
for b in b1 + b2 + b3:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width() / 2, h, '%.3lf' % h, ha='center', va='bottom', fontsize=15, alpha=0.7)
plt.title('Features', fontsize=40)
plt.tick_params(labelsize=20)
xlabels = plt.xticks(x, keys, fontsize=25, rotation=45, ha="right")
for i in range(len(keys)):
    xlabels[1][i]._color = colors[i]
plt.legend(fontsize=30, loc='best')
plt.show()
plt.tight_layout()
plt.savefig('fetures.jpg')