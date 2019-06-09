from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import Preprocessing

# Import data
data = Preprocessing.Preprocessing().importData()

# 熱力圖
corr_matrix = data[['DEWP', 'HUMI', 'PRES', 'TEMP', 'PM_US Post']].corr()
heatmap = sns.heatmap(
    corr_matrix,
    cbar=True,
    annot=True,
    square=True,
    fmt='.2f',
    annot_kws={'size': 15},
    yticklabels=['DEWP', 'HUMI', 'PRES', 'TEMP', 'PM_US Post'],
    xticklabels=['DEWP', 'HUMI', 'PRES', 'TEMP', 'PM_US Post'],
    cmap='Dark2'
    )
plt.show()
