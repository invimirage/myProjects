import numpy as np
import pandas as pd
from sympy import *


def wilson_score(pos_rat, total, p_z=2.0):
    """
    威尔逊得分计算函数
    参考：https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    :param pos: 正例数
    :param total: 总数
    :param p_z: 正太分布的分位数
    :return: 威尔逊得分
    """
    score = (
        pos_rat
        + (np.square(p_z) / (2.0 * total))
        - (
            (p_z / (2.0 * total))
            * np.sqrt(4.0 * total * (1.0 - pos_rat) * pos_rat + np.square(p_z))
        )
    ) / (1.0 + np.square(p_z) / total)
    return score


# print(wilson_score(0.01, 7622, 1.96))


def cal_n_need(lam, rate, cvr):
    n = Symbol("n")
    r = rate
    c = cvr
    l = lam
    solved_value = solve(
        [n * ((r * (n * c + l ** 2 / 2)) ** 2) - n * l ** 4 / 4 - c * (1 - c) * l ** 2],
        [n],
    )
    try:
        return complex(solved_value[-1][0]).real
    except BaseException:
        return 0


df = pd.read_csv("nNeed.csv")
dfn = pd.DataFrame()
dfn["cvr"] = df["cvr"]
dfn["nneed"] = df["nneed"]
dfn.to_csv("nNeed.csv", index=False)
# cvr = []
# for i in df['cvr']:
#     cvr.append('%.4lf' % (float(i) / 10000) )
# df['cvr'] = cvr
# df.to_csv('nNeed.csv')
# cal_n_need(1.96, 0.2, 0.02)
# n = []
# c = []
# x = []
# for i in range(1, 10001):
#     if i % 1000 == 0:
#         print(i)
#     n.append(cal_n_need(1.96, 0.2, i/10000))
#     c.append('%.4lf' % i)
#     x.append([i / 10000])
#
# cvr = {'cvr': c, 'nneed': n}
#
# df = pd.DataFrame(cvr)
# df.to_csv('nNeed.csv')
# plt.plot(x, n)
