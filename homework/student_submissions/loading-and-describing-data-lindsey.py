#---------------------------------------------------Practice Loading and Describing Data
"""1. Load the boston housing data (provided)"""
import urllib
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
# this saves a file called 'housing.data' locally'
urllib.urlretrieve(data_url, r'C:\Users\linds\.spyder-py3\housing.data')

"""2. Load the housing.data file with python"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import csv
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

house_df = pd.read_csv(r'C:\Users\linds\.spyder-py3\housing.csv',sep='\s+',header=None)

names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
         "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
house_df.columns = names
columns = house_df.columns.values.tolist()
print(columns)
"""3. Conduct a brief integrity check of your data."""
house_df.dtypes
house_df.info()
house_df.describe()
house_df.head()
house_df.isnull().sum(axis=0) #no null values

"""4.For what two attributes does it make the least sense to calculate 
mean and median? Why?"""
house_df.describe()
mean = np.mean(house_df)
median_CRIM = np.median(house_df.CRIM)
median_ZN = np.median(house_df.ZN)
#answer: CRIM and ZN 50% (median) have a big gap

"""5. Which two variables have the strongest linear association?"""
sns.pairplot(house_df);
# Make a heatmap on the correlations between variables in the housing data:
#sns.heatmap(housing.corr());
housing_correlations = house_df.corr();
sns.heatmap(housing_correlations);
#Answer: TAX and RAD, NOX and INDUS

"""6. Look at distributional qualities of variables."""
house_df.hist()
#Which variable has the most symmetric distribution?
#Answer: RM
#Which variable has the most left-skewed (negatively skewed) distribution?
#Answer: AGE 
#Which variable has the most right-skewed (positively skewed) distribution?
#Answer: DIS

"""8. Repeat question 6 but scale the variables by their range first."""
house_df.boxplot();
from matplotlib.pyplot import figure

house_df[['ZN']].sort_values(by='ZN',
                          ascending=False).plot(kind='hist',
                                                       stacked=True)
house_df[['CRIM']].sort_values(by='CRIM',
                          ascending=False).plot(kind='hist',
                                                       stacked=True)
                                               
                                                
"""9. Univariate analysis of your choice"""
"""Conduct a full univariate analysis on MEDV, CHAS, TAX, and RAD."""
sns.distplot(house_df.MEDV, kde=False)
plt.title('Histogram of Superhero Weight')
plt.show();
sns.distplot(house_df.CHAS, kde=False)
plt.title('Histogram of Superhero Weight')
plt.show();
sns.distplot(house_df.TAX, kde=False)
plt.title('Histogram of Superhero Weight')
plt.show();
sns.distplot(house_df.RAD, kde=False)
plt.title('Histogram of Superhero Weight')
plt.show();
#1. a measure of centoral tendency on MEDV,CHAS, TAX, and RAD
import statistics as st

#MEDV
st.mean(house_df["MEDV"])
st.mode(house_df["MEDV"])
st.median(house_df["MEDV"])
st.harmonic_mean(house_df["MEDV"])
st.median_low(house_df["MEDV"])
st.median_high(house_df["MEDV"])
st.median_grouped(house_df["MEDV"])
#CHAS
st.mean(house_df["CHAS"])
st.mode(house_df["CHAS"])
st.median(house_df["CHAS"])
st.harmonic_mean(house_df["CHAS"])
st.median_low(house_df["CHAS"])
st.median_high(house_df["CHAS"])
st.median_grouped(house_df["CHAS"])
#TAX
st.mean(house_df["TAX"])
st.mode(house_df["TAX"])
st.median(house_df["TAX"])
st.harmonic_mean(house_df["TAX"])
st.median_low(house_df["TAX"])
st.median_high(house_df["TAX"])
st.median_grouped(house_df["TAX"])
#RAD
st.mean(house_df["RAD"])
st.mode(house_df["RAD"])
st.median(house_df["RAD"])
st.harmonic_mean(house_df["RAD"])
st.median_low(house_df["RAD"])
st.median_high(house_df["RAD"])
st.median_grouped(house_df["RAD"])
house_df.columns.values.tolist()
new_df = house_df[["MEDV","CHAS", "TAX","RAD"]].copy()
new_df.describe()
new_df.corr()
#2. a measure of spread on MEDV,CHAS, TAX, and RAD
st.variance(new_df["MEDV"])
st.stdev(new_df["MEDV"])
st.variance(new_df["CHAS"])
st.stdev(new_df["CHAS"])
st.variance(new_df["TAX"])
st.stdev(new_df["TAX"])
st.variance(new_df["RAD"])
st.stdev(new_df["RAD"])
#allvariables at once
new_df.var()
new_df.std()
new_df.skew()
#.skew()
#A value less than -1 is skewed to the left; 
#that greater than 1 is skewed to the right. 
#A value between -1 and 1 is symmetric.
#3. a description of the shape of the distribution (plot or metric based)
pd.plotting.scatter_matrix(new_df, figsize=(10, 8));
new_df.hist()
sns.pairplot(new_df);
new_housing_correlations = new_df.corr();
sns.heatmap(new_housing_correlations);

new_df.plot(x='RAD', y='TAX', kind='scatter', color='dodgerblue', figsize=(7,7), s=20);

fig, axes = plt.subplots(2,2, figsize=(16,8))
new_df['MEDV'].plot(ax=axes[0][0]);
new_df['CHAS'].plot(ax=axes[0][1]);
new_df['TAX'].plot(ax=axes[1][0]);
new_df['RAD'].plot(ax=axes[1][1]);


"""11. Reducing the number of observations"""



#inferential-statistics-in-python
from scipy import stats

new_df['Score_ZScore_MEDV'] = (new_df['MEDV'] - new_df['MEDV'].mean())/new_df['MEDV'].std(ddof=0)
new_df['Score_ZScore_CHAS'] = (new_df['CHAS'] - new_df['CHAS'].mean())/new_df['CHAS'].std(ddof=0)
new_df['Score_ZScore_TAX'] = (new_df['TAX'] - new_df['TAX'].mean())/new_df['TAX'].std(ddof=0)
new_df['Score_ZScore_RAD'] = (new_df['RAD'] - new_df['RAD'].mean())/new_df['RAD'].std(ddof=0)
new_df

from sklearn.feature_selection import SelectKBest, \
                                    SelectPercentile, \
                                    GenericUnivariateSelect, \
                                    f_regression
                                    
                                    
""""Reducing the number of observations"""
#random.sample() function to select 50 observations from 'AGE'
np.random.seed(50)
age_df=house_df.AGE.copy()
age_df_sample = house_df.AGE.sample(n=50)
age_df_sample.dtypes
import random
random.sample(age_df, 50)


import numpy as np
new_df_sample = new_df.sample(50, replace=True)
new_df_sample_age = house_df.AGE.sample(50, replace=True)

new_df_sample.dtypes
new_df_sample_age.dtypes

"""
#below is for categorical data
sns.countplot(x='MEDV', data='house_df')
plt.show();
"""







