import os
from math import *

import numpy as np
import pandas as pd
# pip install keras -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
from lxml import etree
from scipy.stats import chi2_contingency


def parse_html(path):
    html = etree.parse(path, etree.HTMLParser())
    table = html.xpath('//*[@id="table-to-xls"]')
    table = etree.tostring(table[0], encoding='utf-8').decode()
    df = pd.read_html(table, encoding='utf-8', header=1)[0]
    return df


dir = 'Data'
# 点击数量超过阈值，才会被接受
threshold = 100
filelist = os.listdir(dir)
weightedBiasDict = {}
mutualInfoDict = {}
chiSquareDict = {}
crossFeat = []
changesDict = {}
sigleDoc = {}
sumDict = {}
dimDict = {}
g_cvr = 0
col_weightcount = 'weightcount'
col_pcvrbias = 'pcvrbias'
col_cvr = 'cvr'

# 过滤值为空的属性
def namefilter(name):
    if name != name or name is None:
        return True
    if type(name) in [int, float]:
        return False
    try:
        name = name[name.find('(') + 1:-1]
        if len(name) == 0:
            return True
        name = name.split(',')
        for i in name:
            if int(i) != 0:
                return False
        return True
    except:
        pass
    return False


def parse_sigle(mydata, key):
    val = 0
    valsquare = 0
    valsum = 0
    biasval = 0
    biassum = 0
    datalen = len(mydata[col_pcvrbias])
    # for chi-square test
    table = [[], []]
    if len(mydata[col_weightcount]) < datalen:
        datalen = len(mydata[col_weightcount])

    tagname = mydata[key]
    sigleDoc[key] = {}
    tagnames = {}
    for i in range(datalen):
        if namefilter(tagname[i]):
            continue
        try:
            weightCount = float(mydata[col_weightcount][i])
        except ValueError:
            continue
        if weightCount < threshold or weightCount != weightCount:
            continue
        cvr = float(mydata[col_cvr][i])
        if cvr is None or cvr != cvr or cvr < 0:
            continue
        bias = float(mydata[col_pcvrbias][i])
        if bias is None or bias == -1.0 or bias != bias:
            continue
        val += cvr
        valsquare += pow(cvr, 2)
        valsum += 1
        trans = weightCount * cvr / (1 + cvr)
        tag = tagname[i]
        tagnames[tag] = len(table[0])
        click = weightCount / (1 + cvr)
        biasval += abs(bias) * weightCount
        biassum += weightCount
        table[0].append(click)
        table[1].append(trans)
    chi, _, dof, _ = chi2_contingency(table)
    if dof > 0:
        chiSquareDict[key] = chi / sqrt(dof)
    sigleDoc[key]['tags'] = tagnames
    sigleDoc[key]['table'] = table
    sigleDoc[key]['sum'] = sum([sum(table[i]) for i in range(len(table))])
    G, _, dof, _ = chi2_contingency(table, lambda_='log-likelihood')
    mutualInfoDict[key] = G
    chi, _, dof, _ = chi2_contingency(table)
    chiSquareDict[key] = chi
    weightedBiasDict[key] = biasval / biassum
    dimDict[key] = valsum
    sumDict[key] = biassum


def parse_cross(mydata, key):
    val = 0
    valsquare = 0
    valsum = 0
    biasval = 0
    biassum = 0
    datalen = len(mydata[col_pcvrbias])
    # for chi-square test
    table = [[], []]
    if len(mydata[col_weightcount]) < datalen:
        datalen = len(mydata[col_weightcount])
    # 记录cross feature的两个父关键词
    f_keys = key.split('_')
    beta = {}
    weights = {}
    tag1, tag2 = mydata[f_keys[0]], mydata[f_keys[1]]
    change = 0
    total_ob = 0
    for i in range(datalen):
        T1, T2 = tag1[i], tag2[i]
        if namefilter(T1) or namefilter(T2):
            continue
        try:
            weightCount = float(mydata[col_weightcount][i])
        except ValueError:
            continue
        if weightCount < threshold or weightCount != weightCount:
            continue
        cvr = float(mydata[col_cvr][i])
        if cvr is None or cvr != cvr or cvr <= 0:
            continue
        bias = float(mydata[col_pcvrbias][i])
        if bias is None or bias == -1.0 or bias != bias:
            continue
        biasval += bias
        biassum += weightCount
        val += cvr
        valsquare += pow(cvr, 2)
        valsum += 1
        trans = weightCount * cvr / (1 + cvr)
        click = weightCount / (1 + cvr)
        table[0].append(click)
        table[1].append(trans)
        try:
            D = sigleDoc[f_keys[0]]
            cvr1 = D['table'][1][D['tags'][T1]] / D['table'][0][D['tags'][T1]]
            D = sigleDoc[f_keys[1]]
            cvr2 = D['table'][1][D['tags'][T2]] / D['table'][0][D['tags'][T2]]
            betaki = 100 * cvr * g_cvr / cvr1 / cvr2
            try:
                beta[T1].append(betaki)
                weights[T1] += weightCount
            except KeyError:
                beta[T1] = [betaki]
                weights[T1] = weightCount
        except KeyError:
            pass

    def parser(beta):
        stderr = np.std(beta)
        return 2 / (1 + exp(-0.05 * stderr)) - 1

    for key1, val in beta.items():
        beta[key1] = parser(val)
    cross_importance = 0
    for key1, val in beta.items():
        cross_importance += val * weights[key1]

    chi, _, dof, _ = chi2_contingency(table)
    if dof > 0:
        chiSquareDict[key] = chi / sqrt(dof)

    try:
        changesDict[key] = cross_importance / sum(weights.values())
    except:
        changesDict[key] = 0

    weightedBiasDict[key] = val / valsum

    G, _, dof, _ = chi2_contingency(table, lambda_='log-likelihood')
    mutualInfoDict[key] = G
    chi, _, dof, _ = chi2_contingency(table)
    chiSquareDict[key] = chi
    chi, _, dof, _ = chi2_contingency(table)
    weightedBiasDict[key] = biasval / biassum
    try:
        changesDict[key] = change / total_ob
    except:
        changesDict[key] = 0
    dimDict[key] = valsum
    sumDict[key] = biassum


for fname in filelist:
    if fname[-4:] != '.csv':
        continue
    if len(fname.split(',')) == 2:
        continue
    if g_cvr == 0:
        weightsum = 0
        for cvr, weight in zip(mydata['cvr'], mydata['weight_count']):
            if cvr != cvr or cvr < 0:
                continue
            g_cvr += cvr * weight
            weightsum += weight
        g_cvr /= weightsum
    mydata = pd.read_csv(dir + '/' + fname, error_bad_lines=False)
    key = fname[:-4]
    parse_sigle(mydata, key)

for fname in filelist:
    if fname[-4:] != '.csv':
        continue
    if len(fname.split(',')) == 1:
        continue
    mydata = pd.read_csv(dir + '/' + fname, error_bad_lines=False)
    key = fname[:-4]
    crossFeat.append(key)
    parse_cross(mydata, key)


def parse_key(key):
    if (key.split(','))[0] == key:
        return key.split('_')[-1]
    return ','.join(map(lambda x: x.split('_')[-1], key.split(',')))


weightedBiasDict = list(sorted(weightedBiasDict.items(), key=lambda x: x[1], reverse=True))
for fname, bias in weightedBiasDict:
    print('Feature:{:<35}PcvrBias Weighted Average:{:%}'.format(parse_key(fname), bias))
mutualInfoDict = list(sorted(mutualInfoDict.items(), key=lambda x: x[1], reverse=True))
for fname, ll in mutualInfoDict:
    print('Feature:{:<35}log-likelihood:{}'.format(parse_key(fname), ll))
chiSquareDict = list(sorted(chiSquareDict.items(), key=lambda x: x[1], reverse=True))
for fname, pval in chiSquareDict:
    print('Feature:{:<35}ChiSquare value:{}'.format(parse_key(fname), pval))
changesDict = dict(sorted(changesDict.items(), key=lambda x: x[1], reverse=True))
for fname, change in changesDict.items():
    print('Feature:{:<35}Cross importance value:{}'.format(parse_key(fname), change))