import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from sklearn.neighbors import KNeighborsClassifier
"""1. Load the cell phone "churn" data containing some missing values."""
churn = pd.read_csv( r'C:\Users\linds\.spyder-py3\churn_missing.csv')
churn.dtypes
churn.isnull().sum(axis=0)
churn.info()
churn.shape
churn.describe()

churn.corr()
plt.cla()
sns.heatmap(churn.corr())

fig, ax = plt.subplots(figsize=(9,7))
ax = sns.heatmap(churn.corr(), ax=ax)
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14)
plt.show()

"""3. Convert the vmail_plan and intl_plan colums to binary integer columns."""
churn.intl_plan.value_counts(dropna=False)
churn.vmail_plan.value_counts(dropna=False)
churn.state.value_counts()

churn.loc[:,'vmail_plan'] = churn.vmail_plan.map(lambda x: 1 if x=='yes'
         else 0 if x=='no' else x)

churn.loc[:,'intl_plan'] = churn.intl_plan.map(lambda x: 1 if x=='yes'
         else 0 if x=='no' else x)

"""4. Create dummy coded columns for state and concatenate it to the churn dataset."""
states = pd.get_dummies(churn.state, drop_first=True)
states.head()
churn = pd.concat([churn, states], axis=1)

"""5. Create a version of the churn data that has no missing values."""
churn_nonan = churn.dropna()
churn_nonan.shape

"""6. Create a target vector and predictor matrix."""
X =churn_nonan.drop(['area_code','state','churn'], axis =1)
y = churn_nonan.churn.values
"""7. Calculate the baseline accuracy for churn."""
churn_nonan.churn.mean()
baseline = 1 - churn_nonan.churn.mean()
print(baseline)
"""8. Cross-validate a KNN model predicting churn."""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Xs = ss.fit_transform(X)
scores = cross_val_score(knn, Xs, y, cv=10)
print(scores)
print(np.mean(scores))
"""9. Iterate from k=1 to k=49 (only odd k) and """ 
"""cross-validate the accuracy of the model for each"""

k = list(range(1,50,2))
accur = []
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, Xs, y, cv=10)
    accur.append(np.mean(scores))
    
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(k, accur, lw=3)
plt.show()
print(np.max(accur))    

"""10. Imputing with KNN"""
from sklearn.neighbors import KNeighborsRegressor
missing_cols = ['vmail_plan','vmail_message']

impute_missing = churn.loc[churn.vmail_plan.isnull(), :]
impute_valid = churn.loc[~churn.vmail_plan.isnull(), :]

impute_cols = [c for c in impute_valid.columns if not c in ['state','area_code','churn']+missing_cols]
y = impute_valid.vmail_plan.values
X = impute_valid[impute_cols]

ss = StandardScaler()
Xs = ss.fit_transform(X)

X.columns

#find the best k
def find_best_k (X, y, k_min=1, k_max=51, step=2, cv=5):
    k_range = list(range(k_min, k_max+1, step))
    accs = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=cv)
        accs.append(np.mean(scores))
    print(np.max(accs), np.argmax(k_range))
    return np.argmax(k_range)

find_best_k(Xs, y)

impute_valid.vmail_plan.mean()
vmail_plan_baseline = 1 - impute_valid.vmail_plan.mean()
print(vmail_plan_baseline)


knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(Xs, y)

X_miss = impute_missing[impute_cols]
X_miss_s = ss.transform(X_miss)

vmail_plan_impute = knn.predict(X_miss_s)

vmail_plan_impute


# creating a DF copy to use to imputed missing values
churn_imputed = churn.copy()
# filling missing vmail_plan values with those predicted by KNN model
churn_imputed.loc[churn.vmail_plan.isnull(), 
                  'vmail_plan'] = vmail_plan_impute

"""11. Impute the missing values for vmail_message using the same process."""

"""12. Given the accuracy (and $R^2$) of your best imputation models"""
"""when finding the best K neighbors, do you think imputing is a good idea?"""

"""13. With the imputed dataset, cross-validate the accuracy predicting churn."""
""" Is it better? Worse? The same?"""







