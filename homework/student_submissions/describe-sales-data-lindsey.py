#---------------------------------------------Describing Sales Data with numpy
import numpy as np
import scipy.stats as stats
import csv
import seaborn as sns
import pandas as pd
from scipy import stats
from ipywidgets import interact

%matplotlib inline

"""1. Loading the data"""

csv_rows = []
with open(r'C:\Users\linds\.spyder-py3\sales_info.csv', 'rU') as f:
    reader = csv.reader(f)
    for row in reader:
        csv_rows.append(row)
f.close()

"""2. Separate header and data"""
#conver to data frame
df = pd.DataFrame(csv_rows)
new_header = df.iloc[0] #grab the first row for the header
df = df[1:] #take the data less the header row
df.columns = new_header #set the header row as the df header

"""3. Create a dictionary with the data"""
sales_info = pd.DataFrame(df).to_dict('list')
sales_info
sales_info.keys()

"""3.A Print out the first 10 items of the 'volume_sold' column."""
#valoum_sold key
#A Print out the first 10 items of the 'volume_sold' column.
sales_info.get("volume_sold")[0:9]
sales_info["volume_sold"][0:9]

"""4. Convert data from string to float"""
df.dtypes
df = df.astype(float)
df.dtypes

"""5. Write function to print summary statistics"""
#1. Print out the column name
columns = df.columns.values.tolist()
print(columns)
#2. Print the mean of the data using np.mean()
volume_sold_mean = np.mean(df.volume_sold)
margin_mean = np.mean(df['2015_margin'])
five_q1_sales_mean = np.mean(df['2015_q1_sales'])
six_q1_sales_mean = np.mean(df['2016_q1_sales'])
mean = np.mean(df)
mean
#3. Print out the median of the data using np.median()
volume_sold_median = np.median(df.volume_sold)
margin_median = np.median(df['2015_margin'])
five_q1_sales_median = np.median(df['2015_q1_sales'])
six_q1_sales_median = np.median(df['2016_q1_sales'])
#or
median = df.median()
median
#4. Print out the mode of the rounded data using stats.mode()
volume_sold_mode = stats.mode(df.volume_sold)
margin_median = stats.mode(df['2015_margin'])
five_q1_sales_median = stats.mode(df['2015_q1_sales'])
six_q1_sales_median = stats.mode(df['2016_q1_sales'])
#or
mode = stats.mode(df)

#5.Print out the variance of the data using np.var()
print(df.var())
print(np.var(df))
#6. Print out the standard deviation of the data using np.std()
print(df.std())
print(np.std(df))

"""5.A Using your function, print the summary statistics for volume_sold."""
df.volume_sold.describe()
df['2015_margin'].describe()
df['2015_q1_sales'].describe()
df['2016_q1_sales'].describe()
#number format
pd.set_option('display.float_format', lambda x: '%.5f' % x)


"""6. [Bonus] Plot the distributions"""
def distribution_plotter(column, data_set):
    data = data_set[column]
    sns.set(rc={"figure.figsize": (10, 7)})
    sns.set(color_codes=True)
    sns.set(style="white", palette="muted")
    dist = sns.distplot(data, hist_kws={'alpha':0.2}, kde_kws={'linewidth':5})
    dist.set_title('Distribution of ' + column + '\n', fontsize=16)
      


