#------------------------------------------Classification and KNN with NHL data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# web location:
local_csv = r'C:\Users\linds\.spyder-py3\NHL_Data_GA.csv'
"""1.Load the NHL data"""
nhl = pd.read_csv(local_csv)
nhl.head()
"""2. Perform any required data cleaning. Do some EDA."""
nhl.dtypes
nhl.isnull().sum(axis=0)
nhl.info()
nhl.shape
nhl.describe()

nhl.corr()
plt.cla()
sns.heatmap(nhl.corr())

fig, ax = plt.subplots(figsize=(9,7))
ax = sns.heatmap(nhl.corr(), ax=ax)
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14)
plt.show()

fig, ax = plt.subplots(figsize=(9,7))
mask = np.zeros_like(nhl.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
ax = sns.heatmap(nhl.corr(), mask=mask, ax=ax)
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14)
plt.show()

"""3. Set up the Rank variable as your target. How many classes are there?"""
nhl.Rank.nunique()
nhl.Rank.unique()
y = nhl.Rank

"""4. What is the baseline accuracy?"""
y.value_counts()/y.count()

"""5. Choose 4 features to be your predictor variables and set up your design matrix."""
feature_cols = ['CF%', 'GA', 'Sh%', 'CA']
X = nhl[feature_cols]
X.head()

"""6. Fit a KNeighborsClassifier with 1 neighbor using the target and predictors"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
y_pred_class = knn.predict(X)
print(metrics.accuracy_score(y, y_pred_class))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
"""7. Evaluate the accuracy of your model."""
y_pred_class = knn.predict(X_test)
print((metrics.accuracy_score(y_test, y_pred_class)))
"""knn=50"""
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
print((metrics.accuracy_score(y_test, y_pred_class)))

""" Create a 50-50 train-test-split of your target and predictors.""" 
"""Refit the KNN and assess the accuracy"""


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99,
                                                    test_size=0.5)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred_class = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred_class))

"""9. Evaluate the test accuracy of a KNN where K == number of rows"""
"""in the training data."""
knn = KNeighborsClassifier(n_neighbors=X_train.shape[0])
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred_class))

"""10. Fit the KNN at values of K from 1 to the number of rows in the training data."""
test_accur = []
for i in range(1, X_train.shape[0]+1):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    test_accur.append(knn.score(X_test, y_test))

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(list(range(1, X_train.shape[0]+1)), test_accur, lw=3.)
plt.show()
"""11. Fit KNN across different values of K and plot the mean cross-validated"""
""" accuracy with 5 folds"""
from sklearn.model_selection import cross_val_score
folds = 5
max_neighbors = np.floor(X.shape[0] - X.shape[0]/5.)
print(max_neighbors)

test_accur = []
for i in range(1, int(max_neighbors)):
    knn = KNeighborsClassifier(n_neighbors=i)
    test_accur.append(np.mean(cross_val_score(knn, X, y, cv=5)))

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(list(range(1, int(max_neighbors))), test_accur, lw=3.)
plt.show()

"""12. Standardize the predictor matrix and cross-validate across the different K."""

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
Xss = ss.fit_transform(X)
test_accur_std = []
for i in range(1, int(max_neighbors)):
    knn = KNeighborsClassifier(n_neighbors=i)
    test_accur_std.append(np.mean(cross_val_score(knn, Xss, y, cv=5)))


fig, ax = plt.subplots(figsize=(8,6))
ax.plot(list(range(1, int(max_neighbors))), test_accur, lw=3.)
ax.plot(list(range(1, int(max_neighbors))), test_accur_std, lw=3., color='darkred')
plt.show()


