#Perform EDA on your dataset
#Exploratory data analysis is a crucial step in any data workflow. 
#Create a Jupyter Notebook that explores your data mathematically and visually. Explore features, apply descriptive statistics, look at distributions, and determine how to handle sampling or any missing values.

#Requirements
#Create an exploratory data analysis notebook.
#Perform statistical analysis, along with any visualizations.
#Determine how to handle sampling or missing values.
#Clearly identify shortcomings, assumptions, and next steps.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')

data = pd.read_csv(r'C:\Users\linds\.spyder-py3\50_Startups.csv')
data.head(10)
data.shape
data.describe()
data.dtypes
data.isnull().sum(axis=0)
#no missing values
data.info
data.columns
#total three unique values under state
data.State.unique()

#how many startups per state
data.groupby('State')['State'].count()

#correlation
data.corr()
sns.heatmap(data.corr())
#strong correlation
#1. R&D spend and Profit 0.972900
#2. Marketing Spend and Profit 0.747766
#3. R&D spend and Makreing spend 0.724248

#market position
X_RND = data['R&D Spend']
X_admin = data.Administration
X_MRK_spend = data['Marketing Spend']
y=data['Profit']
color = data.State

#scatter chart
sns.scatterplot(x=X_RND, y=y,hue = color, data = data )
#clear & strong positive linear regression
sns.scatterplot(x=X_admin, y=y,hue = color, data = data)
#no linear
sns.scatterplot(x=X_MRK_spend, y=y,hue = color, data = data )
#positive linear regression
sns.scatterplot(x=X_RND, y=X_MRK_spend,hue = color, data = data )
#positive linear regression

#summary
df_state_sum = data.groupby('State').sum()
df_state_sum

df_state_avg = data.groupby('State').mean()
df_state_avg

#plot
import matplotlib.pyplot as plt
%matplotlib inline
#sum
fig, ax = plt.subplots()
df_state_sum.plot( legend = True);
ax.legend(['State']);
#average
fig, ax = plt.subplots()
df_state_avg.plot( legend = True);
ax.legend(['State']);

#top 10 profits
data.sort_values(['Profit'],ascending=[False]).head(10)

#ROI
data['R&D_Marketing_Spend'] = data['R&D Spend']+data['Marketing Spend']
data.head()
data['Total_spend'] = data['R&D Spend']+data['Marketing Spend']+data.Administration
data.head()
#marketing and R&D only
data['ROI_R&D_MRKG'] = \
(data['Profit'] -data['R&D_Marketing_Spend'])/data['R&D_Marketing_Spend']
#total expenses
data['ROI_TTL_SPND'] = \
(data['Profit'] -data['Total_spend'])/data['Total_spend']
data.head()
#net profit
data['net_profit'] = data['Profit'] - data['ROI_TTL_SPND']
data.head()
#top 10 profits
data.sort_values(['Profit'],ascending=[False]).head(10)
#top ROI
data.sort_values(['ROI_TTL_SPND'],ascending=[False]).head(10)

#top net profit
data.sort_values(['net_profit'],ascending=[False]).head(10)

#bottom net profit
data.sort_values(['net_profit'],ascending=[True]).head(10)

data.columns
#plot
plt.rcParams["figure.figsize"] = [16,9]
sns.scatterplot(x=data.net_profit, y=data.ROI_TTL_SPND,hue = color,data = data )
#plot
#plt.rcParams["figure.figsize"] = [16,9]
#sns.boxplot(x=data.net_profit, y=data['ROI_R&D_MRKG'], hue = color,data = data )

#modeling
#linear regression
#Lasso regression
#random forest











