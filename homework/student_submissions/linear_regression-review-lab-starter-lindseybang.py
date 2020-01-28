#linear_regression-review-lab-starter


import numpy as np
import pandas as pd
import random

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

%matplotlib inline

key = ['X', 'Y']
value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[ .5, .7, .8, .99, 1, 1.4, 1.8, 2.1, 2.4, 2.9]]
d = dict(zip(key, value))
pre_df = pd.DataFrame([d], columns=d.keys())

"""Create a Python dictionary"""
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y=[ .5, .7, .8, .99, 1, 1.4, 1.8, 2.1, 2.4, 2.9]
d = dict(zip(X, Y))
d
"""Using that dictionary, create a pandas DataFrame and call it pre_df"""
pre_df = pd.DataFrame(list(d.items()), columns=['X', 'Y'])
#first column as index column
#pre_df=pd.DataFrame.from_dict(d, orient = "index")

"""Using the Series from the DataFrame, create two new series"""

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in range(0,len(a)):
    i = i+10
    a.append(i)
a

b=[ .5, .7, .8, .99, 1.0, 1.4, 1.8, 2.1, 2.4, 2.9]

for i in b:
    while i<3:
        i = i+3
        b.append(i)
b
#Add those new lists to a new DataFrame and save it as new_data (hint: zip())
lists = list(zip(a, b)) 
new_data = pd.DataFrame(lists, columns = ['X', 'Y']) 
#different way to create a dataframe
new_data = pd.DataFrame(
    {'X': a,
     'Y': b,
    })
    
"""Using pd.concat, vertically concat the new DataFrame, new_data,""" 
"""to the original pre_df DataFrame. Save it as df."""    

df =  pre_df.append(new_data, ignore_index=True)
    
"""Plot the df DataFrame using pandas + matplotlib"""
plt.title("X Vs Y")
plt.xlabel("X")
plt.ylabel("Y")
df.plot(kind='scatter', figsize=(15,3),s=50, color='black',x='X', y='Y', alpha=1);

import seaborn as sns
sns.lmplot(x='X', y='Y', data=df, aspect=1.5,  scatter_kws={'alpha':0.2});

"""Using statsmodels, fit an OLS regression to your data and print our the summary"""
import statsmodels.api as sm
from numpy.random import randn
from sklearn.model_selection import train_test_split

df.corr()

X=df.iloc[:,-1]
y=df.iloc[:,1]
X=X.values.reshape(-1,1)
y=y.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
# Instantiate and fit.
linreg = LinearRegression()
linreg.fit(X, y)
# Print the coefficients.
print(linreg.intercept_)
print(linreg.coef_)

#OSL
X=df.iloc[:,-1]
y=df.iloc[:,1]
X=X.values.reshape(-1,1)
y=y.values.reshape(-1,1)
y=sm.add_constant(y)

model = sm.OLS(X,y)
results = model.fit()
print(results.summary())

"""Using the model you fitted, answer the folowing questions:"""
#R squared
#A: 0.98
#p-value
#A:0.009
#intercept
#A: -8
#the equation for our model
#y= 1*x -8


from sklearn.model_selection import train_test_split

# Define a function that accepts a list of features and returns testing RMSE.
def train_test_rmse(df, feature_cols):
    X = df.X
    y = df.Y
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)#random_state - set.seed
    
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print(train_test_rmse(df, ['X']))

# Calculate these metrics by hand!
from sklearn import metrics
import numpy as np

X = df.X
y = df.Y
    
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)#random_state - set.seed
    
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#error
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


## Without a constant

import statsmodels.api as sm

X = df["X"]
y = df["Y"]
# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model
# Print out the statistics
model.summary()

## With a constant
import statsmodels.api as sm # import statsmodels 

X = df["X"]
y = df["Y"]
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
# Print out the statistics
model.summary()

import statsmodels.formula.api as smf
result = smf.ols('Y ~ X',df).fit()
df['y_pred'] = result.predict()
from sklearn import metrics
print('MSE:', metrics.mean_squared_error(df['Y'], df['y_pred']))
print(metrics.mean_squared_error(df['Y'], df['y_pred'])*20 )
result.predict(pd.DataFrame({'X':[20,21]}))

sns.set(rc={"lines.linewidth": 0.7})
sns.lmplot(x='X', y='Y', data=df, aspect=1.5, scatter_kws={'alpha':0.6})

import sklearn
polyno = sklearn.preprocessing.PolynomialFeatures(degree = 5)
X_reshape = df.X.values.reshape(-1,1)
poly_feats = poly.fit_transform(X_reshape)
print(poly_feats)
poly_X = pd.DataFrame(data = poly_feats)
poly_X
pdf = pd.merge(df[['Y']], poly_X, left_index=True, right_index=True)
print(pdf)

Y = pdf['Y']
model = sm.OLS(Y,poly_feats)
results = model.fit()
print(results.summary())
print(results.params)

poly_yhat = results.predict(poly_feats)
y_pred_poly = pd.Series(poly_yhat)
df['y_pred_poly'] = y_pred_poly
print(df)

