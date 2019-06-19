import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import Preprocessing

split_line = "-"*25

# Import data
data = Preprocessing.Preprocessing().importData()

# Show data columns
print("Data columns:\n", data.columns)
print(split_line)

# Show data info
print("Data info:\n", data.info())
print(split_line)

# Show data describtion
print("Data describtion:\n", data.describe())
print(split_line)

# PM 2.5 常態分配
plt.title("PM2.5 Normal Distribution")
sns.distplot(data[['PM_US Post']])
plt.show()

# PM 2.5 曲線（依日期）
plt.figure(figsize=(125,10))
plt.title("PM2.5 line plot of date")
datetime_data = Preprocessing.Preprocessing().importData()
datetime_data['datetime'] = pd.to_datetime(datetime_data[['year', 'month', 'day']])
datetime_data.set_index(datetime_data['datetime'] , inplace=True)
datetime_data = datetime_data.drop(['year', 'month', 'day', 'hour', 'season', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'datetime'] , axis=1)
sns.lineplot(data=datetime_data)
plt.show()

# PM 2.5 圓餅圖（依年份）
plt.figure(figsize=(125,10))
plt.title("PM2.5 box plot of year")
sns.boxplot(x="year", y="PM_US Post", data=data)
plt.show()

# PM 2.5 圓餅圖（依月份）
plt.figure(figsize=(125,10))
plt.title("PM2.5 box plot of month")
sns.boxplot(x="month", y="PM_US Post", data=data)
plt.show()

# PM 2.5 圓餅圖（依季節）
plt.figure(figsize=(125,10))
plt.title("PM2.5 box plot of season")
sns.boxplot(x="season", y="PM_US Post", data=data)
plt.show()

# PM 2.5 等級長條圖（依季節）
plt.figure(figsize=(125,10))
season_data = Preprocessing.Preprocessing().importData()
conditions = [
    (season_data['PM_US Post'] <= 50),
    (season_data['PM_US Post'] > 50) & (season_data['PM_US Post']<=100),
    (season_data['PM_US Post'] > 100) & (season_data['PM_US Post']<=150),
    (season_data['PM_US Post'] > 151) & (season_data['PM_US Post']<=200),
    (season_data['PM_US Post'] > 201) & (season_data['PM_US Post']<= 300),
    (season_data['PM_US Post'] > 300)
    ]
choices = ["<50", "50~100", "101~150", "151~200", "201~300", ">300"]
season_data['level'] = np.select(conditions, choices, default="<50")
p = sns.catplot(x="season",y="PM_US Post",hue="level", data=season_data, kind="bar")
p.set_ylabels("count")
plt.show()
