#--------------------Simple Linear Regression with Sacramento Real Estate Data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

%matplotlib inline

sac = pd.read_csv(r'C:\Users\linds\.spyder-py3\sacramento_real_estate_transactions.csv')
sac.head()
corr = sac.corr()
sac.dtypes
sac['zip'] = sac['zip'].astype(str)
sac.describe()
sac.city.value_counts()
sac[sac['beds'] == 0]
print(sac[sac['beds'] == 0].shape)
print(sac[sac['price'] < 1])
print(sac[sac['sq__ft'] < 0])
print(sac[sac['state'] != 'CA'])

sac.drop(703, inplace = True)


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

sns.lmplot(x='sq__ft', y='price', data=sac)
plt.show()
sns.lmplot(x='beds', y='price', data=sac)
plt.show()
sns.lmplot(x='baths', y='price', data=sac)
plt.show()

sac.to_csv(r'C:\Users\linds\.spyder-py3\sacramento_real_estate_transactions_Clean.csv')

import numpy as np
import scipy.stats

def lin_reg(x,y):
    # Using other libraries for standard Deviation and Pearson Correlation Coef.
    # Note that in SLR, the correlation coefficient multiplied by the standard
    # deviation of y divided by standard deviation of x is the optimal slope.
    beta_1 = (scipy.stats.pearsonr(x,y)[0])*(np.std(y)/np.std(x))
    
    # Pearson Co. Coef returns a tuple so it needs to be sliced/indexed
    # the optimal beta is found by: mean(y) - b1 * mean(x)
    beta_0 = np.mean(y)-(beta_1*np.mean(x)) 
    
    #Print the Optimal Values
    print('The Optimal Y Intercept is ', beta_0)
    print('The Optimal slope is ', beta_1)





from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
x = sac['sq__ft'].values
y = sac['price'].values
lin_reg(x,y)

"""Adding More Features to the Model¶"""
feature_cols = ['sq__ft']
# Create X and y.
x = sac[feature_cols].values
y = sac['price'].values

# Instantiate and fit.
linreg = LinearRegression()
linreg.fit(x, y)

# Print the coefficients.
print(linreg.intercept_)
print(linreg.coef_)

# Pair the feature names with the coefficients.
#Zip taking first interation and group them together
coeflist_zip = list(zip(feature_cols, linreg.coef_))

y_pred = []

for x in sac['sq__ft']:
    y = 162938.74 + (54.16*x)
    y_pred.append(y)
    
# Appending the predicted values to the Sacramento housing dataframe to do DF calcs
sac['Pred'] = y_pred
# Residuals equals the difference between Y-True and Y-Pred
sac['Residuals'] = abs(sac['price']-sac['Pred'])
sac['Residuals'].mean()


# Plot showing out linear forcast
fig = plt.figure(figsize=(20,20))

# change the fontsize of minor ticks label
plot = fig.add_subplot(111)
plot.tick_params(axis='both', which='major', labelsize=20)

# get the axis of that figure
ax = plt.gca()

# plot a scatter plot on it with our data
ax.scatter(x= sac['sq__ft'], y=sac['price'], c='k')
ax.plot(sac['sq__ft'], sac['Pred'], color='r');



fig = plt.figure(figsize=(20,20))
# change the fontsize of minor ticks label
plot = fig.add_subplot(111)
plot.tick_params(axis='both', which='major', labelsize=20)
# get the axis of that figure
ax = plt.gca()
# plot a scatter plot on it with our data
ax.scatter(x= sac['sq__ft'], y=sac['price'], c='k')
ax.plot(sac['sq__ft'], sac['Pred'], color='r');
# iterate over predictions
for _, row in shd.iterrows():
    plt.plot((row['sq__ft'], row['sq__ft']), (row['price'], row['Pred']), 'b-')
