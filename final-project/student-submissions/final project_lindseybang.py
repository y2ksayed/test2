#Perform EDA on your dataset
#Exploratory data analysis is a crucial step in any data workflow. 
#Create a Jupyter Notebook that explores your data mathematically and visually. Explore features, apply descriptive statistics, look at distributions, and determine how to handle sampling or any missing values.

#Requirements
#Create an exploratory data analysis notebook.
#Perform statistical analysis, along with any visualizations.
#Determine how to handle sampling or missing values.
#Clearly identify shortcomings, assumptions, and next steps.

import seaborn as sns
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
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
data.describe()

#total three unique values under state
data.State.unique()

#how many startups per state
data.groupby('State')['State'].count()

#plot
data.plot();

#correlation
data.corr()
plt.rcParams["figure.figsize"] = [16,9]
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
sns.scatterplot(x=data['State'], y=y,hue = color, data = data, legend=0 )
#no relationship

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
# Pandas scatterplot
data.plot(kind='scatter', x='R&D Spend', y='Profit', alpha=1);
data.plot(kind='scatter', x= 'Marketing Spend', y='Profit', alpha=1);
# Seaborn scatterplot with regression line
sns.lmplot(x='R&D Spend', y='Profit', data=data, aspect=1.5, scatter_kws={'alpha':0.2});
sns.lmplot(x='Marketing Spend', y='Profit', data=data, aspect=1.5, scatter_kws={'alpha':0.2});
#Visualizing data 
#Create feature column variables
data.columns
feature_cols = ['R&D Spend', 'Administration', 'Marketing Spend']
sns.pairplot(data, x_vars=feature_cols, y_vars='Profit', kind='reg');
sns.pairplot(data, x_vars=feature_cols, y_vars='Profit', kind='reg');
grr = pd.plotting.scatter_matrix(data[['Profit'] + feature_cols], figsize=(15, 15), alpha=0.7)
#boxplot
#data.boxplot(column='Profit', by='Total_spend');

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
data['net_profit'] = data['Profit'] - data['Total_spend']
data.head()
#top 10 profits
data.sort_values(['Profit'],ascending=[False]).head(10)
#top ROI
data.sort_values(['ROI_TTL_SPND'],ascending=[False]).head(10)
data.sort_values(['ROI_R&D_MRKG'],ascending=[False]).head(50)


#top net profit
data.sort_values(['net_profit'],ascending=[False]).head(10)

#bottom net profit
data.sort_values(['net_profit'],ascending=[True]).head(10)

data.columns
#plot
plt.rcParams["figure.figsize"] = [16,9]
sns.scatterplot(x=data.net_profit, y=data.ROI_TTL_SPND,hue = color,data = data )
#plot
data.plot();
#plt.rcParams["figure.figsize"] = [16,9]
#sns.boxplot(x=data.net_profit, y=data['ROI_R&D_MRKG'], hue = color,data = data )

"""Part 3"""

#modeling
#linear regression
#Split the columns into Dependent(Y) and Independent (X)
data.columns
x= data[['R&D Spend','Marketing Spend']]
y= data.iloc[:,4]#profit
#y_roi=data.iloc[:,7] #ROI

"""method 1"""
#linear regression using original dataset of X
from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(x, y)
#print(lr.intercept_)
#print(lr.coef_)
print('Intercept: \n', lr.intercept_)
print('Coefficients: \n', lr.coef_)

#from sklearn.linear_model import LinearRegression
#lrlr= linear_model.LinearRegression()
#lrlr.fit(x, y)
#print(lrlr.intercept_)
#print(lrlr.coef_)
#print('Intercept: \n', lrlr.intercept_)
#print('Coefficients: \n', lrlr.coef_)


#R&D
#y = 47561.43852058689 + 0.78962993X
#Marketing
#y=47561.43852058689 + 0.03064816X

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(x,y, test_size = 0.2, random_state = 1234)
lr= LinearRegression()
lr.fit(X_train, Y_train)
print('Intercept: \n', lr.intercept_)
print('Coefficients: \n', lr.coef_)

y_predict = lr.predict(X_test)
y_predict

from sklearn.metrics import mean_squared_error
import math
rmse_linear= math.sqrt(mean_squared_error(Y_test,y_predict))
rmse_linear
#R2
# higher R-squared values represent smaller differences between the observed data 
#and the fitted values.
score_linear = lr.score(X_test,Y_test)
score_linear

mse_linear = mean_squared_error(Y_test, y_predict)
mse_linear

import matplotlib.pyplot as plt
#R&D
X_train.columns
t =plt.scatter(X_train['R&D Spend' ], Y_train, color = 'red')
p =plt.scatter(X_train['R&D Spend' ], lr.predict(X_train), color = 'blue')
plt.title('R&D Spend vs Profits')
plt.xlabel('R&D Spend')
plt.ylabel('Profits')
plt.legend((t,p),('Train','Prediction'), scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=8)
plt.show()

#Makreting
X_train.columns
tt = plt.scatter(X_train['Marketing Spend' ], Y_train, color = 'red')
pp = plt.scatter(X_train['Marketing Spend'], lr.predict(X_train), color = 'blue')
plt.title('Marketing Spend vs Profits')
plt.xlabel('Marketing Spend')
plt.ylabel('Profits')
plt.legend((tt,pp),('Train','Prediction'), scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=8)
plt.show()

# Visualising the test set results in a scatter plot
#R&D
plt.scatter(X_test['R&D Spend'], Y_test, color = 'red')
plt.scatter(X_test['R&D Spend'], y_predict, color = 'blue')
plt.title('R&D Spend vs Profits (Test set)')
plt.xlabel('R&D Spend')
plt.ylabel('Profits')
plt.show()
#marketing
plt.scatter(X_test['Marketing Spend'], Y_test, color = 'red')
plt.scatter(X_test['Marketing Spend'], y_predict, color = 'blue')
plt.title('Marketing Spend vs Profits (Test set)')
plt.xlabel('Marketing Spend')
plt.ylabel('Profits')
plt.show()


"""method 2"""
""" Lasso & Ridge """

#Lasso & Ridge regression
#A larger alpha (toward the left of each diagram) results in more regularization:
#Lasso regression shrinks coefficients all the way to zero, thus removing them from the model.
#Ridge regression shrinks coefficients toward zero, but they rarely reach zero.
"""Lasso"""
#split into X (independents) and Y (predicted)
X=data[['R&D Spend','Marketing Spend']]
Y= data.iloc[:,4]#profit
#import all the regression
from sklearn.linear_model import Lasso, Ridge, LinearRegression
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size = 0.2, random_state = 1234)

#Lasso Regression
alpha_range = [0.1,0.5,1,10,20,50,100]
for i in alpha_range:
    lasso =Lasso(alpha =i)
    lasso.fit(X_train, Y_train)
    y_predict_lasso = lasso.predict(X_test)
    print('Intercept: \n', lasso.intercept_)
    print('Coefficients: \n', lasso.coef_)
    print('Alpha {}, \n mse {}'.format(i, mean_squared_error(Y_test, y_predict_lasso)))

#individual modeling
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, Y_train)
y_predict_lasso = lasso.predict(X_test)

lasso_coeff = lasso.coef_
lasso_intercept = lasso.intercept_
print('Intercept: \n', lasso.intercept_)
print('Coefficients: \n', lasso.coef_)

from sklearn.metrics import mean_squared_error
import math
rmse_lasso = math.sqrt(mean_squared_error(Y_test,y_predict_lasso))
rmse_lasso

score_lasso = lasso.score(X_test,Y_test)
score_lasso
from sklearn.metrics import mean_squared_error
mse_lasso = mean_squared_error(Y_test, y_predict_lasso)
mse_lasso

#plot
#R&D
X_train.columns
t =plt.scatter(X_train['R&D Spend' ], Y_train, color = 'red')
p =plt.scatter(X_train['R&D Spend' ], lasso.predict(X_train), color = 'blue')
plt.title('R&D Spend vs Profits')
plt.xlabel('R&D Spend')
plt.ylabel('Profits')
plt.legend((t,p),('Train','Prediction'), scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=8)
plt.show()

#Makreting
X_train.columns
tt = plt.scatter(X_train['Marketing Spend' ], Y_train, color = 'red')
pp = plt.scatter(X_train['Marketing Spend'], lasso.predict(X_train), color = 'blue')
plt.title('Marketing Spend vs Profits')
plt.xlabel('Marketing Spend')
plt.ylabel('Profits')
plt.legend((tt,pp),('Train','Prediction'), scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=8)
plt.show()

"""Ridge"""
#Ridge Regression
X=data[['R&D Spend','Marketing Spend']]
Y= data.iloc[:,4]#profit
#import all the regression
from sklearn.linear_model import Lasso, Ridge, LinearRegression
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size = 0.2, random_state = 1234)

alpha_range_ridge = [0.1,0.5,1,10,20,50,100]
for i in alpha_range:
    ridge =Ridge(alpha =i)
    ridge.fit(X_train, Y_train)
    y_predict_ridge = ridge.predict(X_test)
    print('Intercept: \n', ridge.intercept_)
    print('Coefficients: \n', ridge.coef_)
    print('Alpha {}, \n mse {}'.format(i, mean_squared_error(Y_test, y_predict_ridge)))

#single model
#alpha = 0.1
ridge = Ridge(alpha = 0.1)
ridge.fit(X_train, Y_train)
y_predict_ridge = ridge.predict(X_test)

ridge_coeff = ridge.coef_
ridge_intercept = ridge.intercept_
print('Intercept: \n', ridge.intercept_)
print('Coefficients: \n', ridge.coef_)

from sklearn.metrics import mean_squared_error
import math
rmse_ridge = math.sqrt(mean_squared_error(Y_test,y_predict_ridge))
rmse_ridge

score_ridge = ridge.score(X_test,Y_test)
score_ridge
from sklearn.metrics import mean_squared_error
mse_ridge = mean_squared_error(Y_test, y_predict_ridge)
mse_ridge
#alpha = 100
ridge = Ridge(alpha = 100)
ridge.fit(X_train, Y_train)
y_predict_ridge = ridge.predict(X_test)

ridge_coeff = ridge.coef_
ridge_intercept = ridge.intercept_
print('Intercept: \n', ridge.intercept_)
print('Coefficients: \n', ridge.coef_)

from sklearn.metrics import mean_squared_error
import math
rmse_ridge = math.sqrt(mean_squared_error(Y_test,y_predict_ridge))
rmse_ridge

score_ridge = ridge.score(X_test,Y_test)
score_ridge
from sklearn.metrics import mean_squared_error
mse_ridge = mean_squared_error(Y_test, y_predict_ridge)
mse_ridge

#find the best alpha value
# Standarize features
from sklearn.linear_model import RidgeCV
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X_train)
# Create ridge regression with three possible alpha values
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0,50.0, 100.0])
# Fit the linear regression
model_cv = regr_cv.fit(X_std, Y_train)
# View alpha
model_cv.alpha_
#after standarscaler
# Fit the linear regression
model_cv = regr_cv.fit(X_train, Y_train)
# View alpha
model_cv.alpha_

#plot
#R&D
X_train.columns
t =plt.scatter(X_train['R&D Spend' ], Y_train, color = 'red')
p =plt.scatter(X_train['R&D Spend' ], ridge.predict(X_train), color = 'blue')
plt.title('R&D Spend vs Profits')
plt.xlabel('R&D Spend')
plt.ylabel('Profits')
plt.legend((t,p),('Train','Prediction'), scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=8)
plt.show()

#Makreting
X_train.columns
tt = plt.scatter(X_train['Marketing Spend' ], Y_train, color = 'red')
pp = plt.scatter(X_train['Marketing Spend'], ridge.predict(X_train), color = 'blue')
plt.title('Marketing Spend vs Profits')
plt.xlabel('Marketing Spend')
plt.ylabel('Profits')
plt.legend((tt,pp),('Train','Prediction'), scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=8)
plt.show()


"""ridge"""
"""excercise"""
#Ridge
from sklearn import metrics
data.columns
X= data[['R&D Spend','Marketing Spend']]
y= data.iloc[:,4]#profit

X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size = 0.2, random_state = 1234)
# alpha=0 is equivalent to linear regression.
from sklearn.linear_model import Ridge
# Instantiate the model.
#(Alpha of zero has no regularization strength, essentially a basic linear regression.)
ridgereg = Ridge(alpha=0, normalize=True)
# Fit the model.
ridgereg.fit(X_train, y_train)
# Predict with fitted model.
y_pred_ridge_1 = ridgereg.predict(X_test)
rmse_4 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge_1))
# Coefficients for a non-regularized linear regression
feature_cols = ['R&D Spend','Marketing Spend']
list(zip(feature_cols, ridgereg.coef_))
# Try alpha=0.1.
ridgereg = Ridge(alpha=0.1, normalize=True)
ridgereg.fit(X_train, y_train)
y_pred_ridge_2 = ridgereg.predict(X_test)
rmse_4_1 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge_2))
rmse_4_1
# Examine the coefficients.
list(zip(feature_cols, ridgereg.coef_))
# Try alpha=1.
ridgereg = Ridge(alpha=1, normalize=True)
ridgereg.fit(X_train, y_train)
y_pred_ridge_3 = ridgereg.predict(X_test)
rmse_4_2 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge_3))
rmse_4_2
# Examine the coefficients.
list(zip(feature_cols, ridgereg.coef_))



"""method 3"""
"""random forest regressor"""
# Import RandomForestClassifier from scikit's ensemble module
# Using Skicit-learn to split data into training and testing sets
X=data[['R&D Spend','Marketing Spend']]
Y= data.iloc[:,4]

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn.linear_model import Lasso, Ridge, LinearRegression

X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size = 0.2, random_state = 1234)

regressor.fit(X_train,Y_train)  

#for loop
for n in range(10,101,1):
    regressor = RandomForestRegressor(n_estimators = n, random_state = 0)
    regressor.fit(X_train,Y_train)
    y_pred_regressor = regressor.predict(X_test) 
    mse_regressor = mean_squared_error(Y_test, y_pred_regressor)
    score_regressor =regressor.score(X_test,Y_test)

    print('n_estimators: {}, mean_squared_error {}, score {}'.format(n,
          mse_regressor,score_regressor))
    
    
#without for-loop
regressor = RandomForestRegressor(n_estimators = 74, random_state = 0) 
# fit the regressor with x and y data 
#regressor.fit(X,Y)  

X_train, X_test, y_train, y_test = \
train_test_split(X, Y, test_size = 0.2, random_state = 1234)

regressor.fit(X_train,Y_train)  
y_pred_regressor = regressor.predict(X_test) 

#regressor_coeff = regressor.coef_
#regressor_intercept = regressor.intercept_
#print('Intercept: \n', regressor.intercept_)
#print('Coefficients: \n', regressor.coef_)

from sklearn.metrics import mean_squared_error
import math
rmse_regressor = math.sqrt(mean_squared_error(Y_test,y_pred_regressor))
rmse_regressor

score_regressor = regressor.score(X_test,Y_test)
score_regressor

from sklearn.metrics import mean_squared_error
mse_regressor = mean_squared_error(Y_test, y_pred_regressor)
mse_regressor

#plot
#R&D
X_train.columns
t =plt.scatter(X_train['R&D Spend' ], Y_train, color = 'red')
p =plt.scatter(X_train['R&D Spend' ], regressor.predict(X_train), color = 'blue')
plt.title('R&D Spend vs Profits')
plt.xlabel('R&D Spend')
plt.ylabel('Profits')
plt.legend((t,p),('Train','Prediction'), scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=8)
plt.show()

#Makreting
X_train.columns
tt = plt.scatter(X_train['Marketing Spend' ], Y_train, color = 'red')
pp = plt.scatter(X_train['Marketing Spend'], regressor.predict(X_train), color = 'blue')
plt.title('Marketing Spend vs Profits')
plt.xlabel('Marketing Spend')
plt.ylabel('Profits')
plt.legend((tt,pp),('Train','Prediction'), scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=8)
plt.show()

"""random forest regreson plots"""
#random forest regression model
X=data[['R&D Spend','Marketing Spend']]
Y= data.iloc[:,4]

X_train, X_test, y_train, y_test = \
train_test_split(X, Y, test_size = 0.2, random_state = 1234)

regressor = RandomForestRegressor(n_estimators = 74, random_state = 0)
regressor.fit(X_train,Y_train) 
regressor.fit(np.array(X_train['R&D Spend']).reshape(-1,1),Y_train) 

X_grid = np.arange(min(X_train['R&D Spend']), max(X_train['R&D Spend']))
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train['R&D Spend'], Y_train, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Investment vs. Profit(Random Forest Regression Model)')
plt.xlabel('Investment')
plt.ylabel('Profit')
plt.show()

#random forest regression model
X=data[['R&D Spend','Marketing Spend']]
Y= data.iloc[:,4]

X_train, X_test, y_train, y_test = \
train_test_split(X, Y, test_size = 0.2, random_state = 1234)

regressor = RandomForestRegressor(n_estimators = 74, random_state = 0)
regressor.fit(X_train,Y_train) 
regressor.fit(np.array(X_train['Marketing Spend']).reshape(-1,1),Y_train) 

X_grid = np.arange(min(X_train['Marketing Spend']), max(X_train['Marketing Spend']))
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train['Marketing Spend'], Y_train, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Investment vs. Profit(Random Forest Regression Model)')
plt.xlabel('Investment')
plt.ylabel('Profit')
plt.show()


"""Method 5"""
"""neural networking"""
#one layer
#couple neurals

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

X=data[['R&D Spend','Marketing Spend']]
Y= data.iloc[:,4]

#split the rows
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  \
train_test_split(X,Y, test_size=0.2, random_state=1234) #stratify=Y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising the ANN
model = Sequential()
# Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 2))
# Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))
# Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))
# Adding the output layer
model.add(Dense(units = 1))
#model.add(Dense(1))
# Compiling the ANN
model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
# Fitting the ANN to the Training set
model.fit(X_train, Y_train, batch_size = 10, epochs = 100)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, 
                           verbose=1, mode='auto')

y_pred_nn = model.predict(X_test)

plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred_nn, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

"""neural netowkring test"""
#split the rows
X=data[['R&D Spend','Marketing Spend']]
Y= data.iloc[:,4]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  \
train_test_split(X,Y, test_size=0.2, random_state=1234)

#Define the keras Model
model = Sequential()
model.add(Dense(5,
                input_shape=(2,),
                activation='relu',
                kernel_initializer='RandomNormal'))

model.add(Dense(3,
                activation='relu',
                kernel_initializer='RandomNormal'))

model.add(Dense(1,
                activation='sigmoid',))



#compile the model
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics = ['mean_squared_error'])

#run the model
#10 batches 160 times
model.fit(X_train, Y_train, epochs=100,batch_size=5)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, 
                           patience=2, verbose=0, mode='auto')

y_predict_nn = model.predict(X_test)

#GEt the accuracy score to evaluate the model
#0 first one -  loss value
#1 second one - accuracy value
#accuracy_test=model.evaluate(X_test,Y_test)

#get the predicted values and predicted provabilities of Y_test
#y_predict_nn = model.predict_classes(X_test)
#Y_pred_prob = model.predict(X_test)
#plot
plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_predict_nn, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

"""different approach"""
"""neural network"""
#Variables
X=data[['R&D Spend','Marketing Spend']]
Y= data.iloc[:,4]
#Y = np.array(Y).reshape(1,-1)
#scaler_x = MinMaxScaler()
#scaler_y = MinMaxScaler()
#print(scaler_x.fit(X))
#xscale=scaler_x.transform(X)
#print(scaler_y.fit(Y))
#yscale=scaler_y.transform(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  \
train_test_split(X,Y, test_size=0.2, random_state=1234)

model = Sequential()
model.add(Dense(12, input_shape=(2,), kernel_initializer='normal', 
                activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(X_train, Y_train, epochs=150, 
                    batch_size=5, verbose=1, validation_split=0.2)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, 
                           patience=2, verbose=0, mode='auto')
print(history.history.keys())

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


"""neural network for loop"""
from tensorflow.keras import layers

def build_model():
  model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(2,)),
    layers.Dense(5, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()
model.summary()

X=data[['R&D Spend','Marketing Spend']]
Y= data.iloc[:,4]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  \
train_test_split(X,Y, test_size=0.2, random_state=1234)
example_result = model.predict(X_test)
example_result

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

EPOCHS = 1000

history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS, validation_split = 0.2, verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = "mae")
plt.ylabel('MAE [MPG]')


#Build the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_predict)

"""neural networking testing"""
#imports
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

#build our model
model = Sequential()
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='rmsprop')

X=data[['R&D Spend','Marketing Spend']]
Y= data.iloc[:,4]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  \
train_test_split(X,Y, test_size=0.2, random_state=1234)

from keras.callbacks import EarlyStopping
model.fit(X_train, Y_train, epochs=100,batch_size=10) #, shuffle=True, verbose=2)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

prediction = model.predict(X_test)
print('Prediction - {}',format(prediction))


"""Validation"""
"K-Folds"
#find out which model works the best with this dataset
"""K-Folds Cross-Validation"""
from sklearn import model_selection
kf = model_selection.KFold(n_splits=5, shuffle=True)

mse_values_lr = []
mse_values_lasso = []
mse_values_ridge = []
mse_values_random = []

scores_lr = []
scores_lasso = []
scores_ridge = []
scores_random = []

n = 0

print("~~~~ CROSS VALIDATION each fold ~~~~")
for train_index, test_index in kf.split(X, Y):
    lr = LinearRegression().fit(X.iloc[train_index], y.iloc[train_index])
    lasso = Lasso(alpha=0.1).fit(X.iloc[train_index], y.iloc[train_index])
    ridge = Ridge(alpha =0.1).fit(X.iloc[train_index], y.iloc[train_index])
    random = RandomForestRegressor(n_estimators=70,
                                   random_state=0).fit(X.iloc[train_index], 
                                                 y.iloc[train_index])
    
    mse_values_lr.append(metrics.mean_squared_error(y.iloc[test_index],
                                                    lr.predict(X.iloc[test_index])))
    mse_values_lasso.append(metrics.mean_squared_error(y.iloc[test_index],
                                                    lasso.predict(X.iloc[test_index])))
    mse_values_ridge.append(metrics.mean_squared_error(y.iloc[test_index],
                                                    ridge.predict(X.iloc[test_index])))
    mse_values_random.append(metrics.mean_squared_error(y.iloc[test_index],
                                                    random.predict(X.iloc[test_index])))
    
    scores_lr.append(lr.score(X, Y))
    scores_lasso.append(lasso.score(X, Y))
    scores_ridge.append(ridge.score(X, Y))
    scores_random.append(random.score(X, Y))

    
    n += 1
    
    print('Model_lr {}'.format(n))
    print('MSE_lr: {}'.format(mse_values_lr[n-1]))
    print('R2_lr: {}\n'.format(scores_lr[n-1]))
    print('Model_lasso {}'.format(n))
    print('MSE_lasso: {}'.format(mse_values_lasso[n-1]))
    print('R2_lasso: {}\n'.format(scores_lasso[n-1]))    
    print('Model_ridge {}'.format(n))
    print('MSE_ridge: {}'.format(mse_values_ridge[n-1]))
    print('R2_ridge: {}\n'.format(scores_ridge[n-1]))    
    print('Model_random {}'.format(n))
    print('MSE_random: {}'.format(mse_values_random[n-1]))
    print('R2_random: {}\n'.format(scores_random[n-1]))


print("~~~~ SUMMARY OF CROSS VALIDATION ~~~~")
print('Mean of MSE for all folds lr: {}'.format(np.mean(mse_values_lr)))
print('Mean of R2 for all folds lr: {}'.format(np.mean(scores_lr)))
print('Mean of MSE for all folds lasso: {}'.format(np.mean(mse_values_lasso)))
print('Mean of R2 for all folds lasso: {}'.format(np.mean(scores_lasso)))
print('Mean of MSE for all folds ridge: {}'.format(np.mean(mse_values_ridge)))
print('Mean of R2 for all folds ridge: {}'.format(np.mean(scores_ridge)))
print('Mean of MSE for all folds random: {}'.format(np.mean(mse_values_random)))
print('Mean of R2 for all folds random: {}'.format(np.mean(scores_random)))



