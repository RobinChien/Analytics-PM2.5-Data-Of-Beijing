from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import Preprocessing

# Import data
data = Preprocessing.Preprocessing().importLevelData()
scaled_data = Preprocessing.Preprocessing().normalizeData(data)
df = pd.concat([data[['season','level']], scaled_data[['DEWP', 'HUMI', 'PRES', 'TEMP']]], axis=1)

# 熱力圖
corr_matrix = df[['season','DEWP', 'HUMI', 'PRES', 'TEMP', 'level']].corr('spearman')
heatmap = sns.heatmap(
    corr_matrix,
    cbar=True,
    annot=True,
    square=True,
    fmt='.2f',
    annot_kws={'size': 15},
    yticklabels=['season','DEWP', 'HUMI', 'PRES', 'TEMP', 'level'],
    xticklabels=['season','DEWP', 'HUMI', 'PRES', 'TEMP', 'level'],
    cmap='Dark2'
    )
plt.show()
