from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

%config InlineBackend.figure_format = 'retina'
%matplotlib inline

plt.style.use('fivethirtyeight')

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()
"""1. Clean up any data problems"""
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=['MEDV'])
boston = pd.concat([y, X], axis=1)
boston.head()

"""2. Select 3-4 variables with your dataset to perform a 50/50 test train split on"""
#Use sklearn.
X.columns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

features = ['CRIM', 'CHAS', 'RM', 'LSTAT']

X_train, X_test, y_train, y_test = train_test_split(X[features], y,
                                                    train_size=0.5, random_state=1234)


lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

#Score and plot your predictions.
y_pred = lr.predict(X_test)
sns.jointplot(y_test, y_pred)

"""Try 70/30 and 90/10"""
#Score and plot.
#How do your metrics change?
#70/30
features = ['CRIM', 'CHAS', 'RM', 'LSTAT']

X_train, X_test, y_train, y_test = train_test_split(X[features], y,
                                                    train_size=0.7, random_state=1234)


lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

#Score and plot your predictions.
y_pred = lr.predict(X_test)
sns.jointplot(y_test, y_pred)


features = ['CRIM', 'CHAS', 'RM', 'LSTAT']

X_train, X_test, y_train, y_test = train_test_split(X[features], y,
                                                    train_size=0.9, random_state=1234)


lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

#Score and plot your predictions.
y_pred = lr.predict(X_test)
sns.jointplot(y_test, y_pred)


"""4. Try K-Folds cross-validation with K between 5-10 for your regression"""
#What seems optimal?
#How do your scores change?
#What the variance of scores like?
#Try different folds to get a sense of how this impacts your score.
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

#mse_values_f=[]
scores_f = []
for fold in range(5,11):
    print('\n~~~~ CROSS VALIDATION ~~~~')
    print('K: {}'.format(fold))
    model = LinearRegression()
   
    # Perform cross-validation
    scores = cross_val_score(model, X[features], y, cv=fold)
    scores_f.append(lr.score(X, y))
    #mse_values_f.append(metrics.mean_squared_error(y.iloc[test_index], lr.predict(X.iloc[test_index])))
   
    print("Cross-validated scores: {}".format(scores))
    print("Mean of R2: {}".format(np.mean(scores)))
    print('Std of R2: {}'.format(np.std(scores)))
   
    # Make cross-validated predictions
    predictions = cross_val_predict(model, X[features], y, cv=fold)
    r2 = metrics.r2_score(y, predictions)
    print("Cross-Predicted R2: {}".format(r2))

print("~~~~ SUMMARY OF CROSS VALIDATION ~~~~")
#print('Mean of MSE for all folds: {}'.format(np.mean(mse_values)))
print('Mean of R2 for all folds: {}'.format(np.mean(scores_f)))

#
import pandas
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np

kf = model_selection.KFold(n_splits=10, shuffle=True)

mse_values = []
scores = []
n = 0

print("~~~~ CROSS VALIDATION each fold ~~~~")
for train_index, test_index in kf.split(X, y):
    lr = LinearRegression().fit(X.iloc[train_index], y.iloc[train_index])
   
    mse_values.append(metrics.mean_squared_error(y.iloc[test_index], lr.predict(X.iloc[test_index])))
    scores.append(lr.score(X, y))
   
    n += 1
   
    print('Model {}'.format(n))
    print('MSE: {}'.format(mse_values[n-1]))
    print('R2: {}\n'.format(scores[n-1]))


print("~~~~ SUMMARY OF CROSS VALIDATION ~~~~")
print('Mean of MSE for all folds: {}'.format(np.mean(mse_values)))
print('Mean of R2 for all folds: {}'.format(np.mean(scores)))

