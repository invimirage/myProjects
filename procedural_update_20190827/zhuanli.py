from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
bias = pd.read_csv('frequent_crosses_.csv')
cvr = pd.read_csv('frequent_crosses.csv')
feat_dict = {}
for feat, val in zip(bias['cross_name'], bias['times']):
    # if val <= 1:
    #     continue
    feat_dict[feat] = [val]
for feat, val in zip(cvr['cross_name'], cvr['times']):
    # if val <= 1:
    #     continue
    try:
        feat_dict[feat].append(val)
    except:
        feat_dict[feat] = [0, val]
b_ = []
c_ = []
for i in feat_dict.values():
    b_.append(i[0])
    if len(i) < 2:
        c_.append(0)
        continue
    c_.append(i[1])
y = np.arange(len(b_))
fig = plt.figure(figsize=[len(c_) *2, 40], dpi=80)
plt.bar(y - 0.3, b_, width=0.3, label='bias')
plt.bar(y, c_, width=0.3, color='crimson', label='cvr')
plt.xticks(y, list(feat_dict.keys()), rotation=45, ha='left', va='top', fontsize=25)
plt.legend(fontsize=30)
plt.tight_layout()
plt.savefig('res.png')
plt.show()
