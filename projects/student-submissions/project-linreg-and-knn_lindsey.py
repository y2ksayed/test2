#---------------------------------------------------------------project 3
#Linear Regression and KNN - Train/Test Split

"""Linear Regression Use Case"""
import matplotlib.pyplot as plt
%matplotlib inline

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

boston = pd.read_csv(r'C:\Users\linds\.spyder-py3\boston_housing_data.csv')

X = boston.iloc[:, :-1]    
y = boston.iloc[:,-1]

"""1. Clean Up Data and Perform Exporatory Data Analysis"""
boston.isnull().sum(axis=0)
boston.describe()
boston.info()
boston.dtypes
boston.shape

boston.apply(lambda x: x.nunique())
X.apply(lambda x: x.nunique()) #number of unique values for each column

import seaborn as sns
sns.heatmap(boston.corr())
"""2. Pick 3-4 predictors (i.e. CRIM, ZN, etc...) that you will use to predict"""
"""our target variable, MEDV"""
from sklearn.linear_model import LinearRegression
feature_cols = ["RM", "ZN", "PTRATIO", "LSTAT"]
X = X[feature_cols]
X = X[["RM", "ZN", "PTRATIO", "LSTAT"]]
y

linreg = LinearRegression()
linreg.fit(X, y)
print(linreg.intercept_)
print(linreg.coef_)
list(zip(feature_cols, linreg.coef_))
score = linreg.score(X, y)
sns.lmplot(x='RM', y='MEDV', data=boston, aspect=1.5, scatter_kws={'alpha':0.2});
sns.lmplot(x='ZN', y='MEDV', data=boston, aspect=1.5, scatter_kws={'alpha':0.2});
sns.lmplot(x='PTRATIO', y='MEDV', data=boston, aspect=1.5, scatter_kws={'alpha':0.2});
sns.lmplot(x='LSTAT', y='MEDV', data=boston, aspect=1.5, scatter_kws={'alpha':0.2});

"""3. Try 70/30 and 90/10 train/test splits (70% of the data for training"""
"""- 30% for testing, then 90% for training - 10% for testing)"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size = 0.3, random_state = 1234)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_predict = linreg.predict(X_test)
score_01 = linreg.score(X_train, y_train)

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_predict)
#from sklearn import metrics
#acc_score = metrics.accuracy_score(y_test, y_predict)
#not for continus data

X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size = 0.1, random_state = 1234)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_predict = linreg.predict(X_test)
score_02 = linreg.score(X_train, y_train)

"""4. Use k-fold cross validation varying the number of folds from 5 to 10"""
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

kf = KFold(n_splits=5, shuffle=True) #when k=5

print(np.mean(-cross_val_score(linreg, X, y, cv=kf, scoring='neg_mean_squared_error')))
print(np.mean(cross_val_score(linreg, X, y, cv=kf)))

#forloop testing#1
for i in range(5,11):
    kf=KFold(n_splits=i, shuffle=True)
    print('NFold: {}, Accuracy_socre: {}'.format(i,
          np.mean(-cross_val_score(linreg, X, y, cv=kf, scoring='neg_mean_squared_error'))))
    print('Val_Score: {}'.format(np.mean(cross_val_score(linreg, X, y, cv=kf))))

#for loop testing
for i in range(5,11):
    kf = KFold(n_splits=i, shuffle=True)
    print('NFold: {}, Accuracy_socre: {}'.format(i,np.mean(cross_val_score(linreg, X, y, cv=kf))))

#test_accur_01=[]
#print(test_accur_01)
#print(np.mean(-cross_val_score(linreg, X, y, cv=kf, scoring='neg_mean_squared_error')))
#print(np.mean(cross_val_score(linreg, X, y, cv=kf)))

"""K-Folds Cross-Validation"""
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import metrics
kf = model_selection.KFold(n_splits=5, shuffle=True)
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


"""KNN Practice"""
# Read the iris data into a DataFrame
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=col_names)

iris.head()

import matplotlib as plt

# Increase the default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 14

# Create a custom colormap
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                           
# Map each iris species to a number
# Let's use Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2 and
#create a column called 'species_num' (disctionary format)
iris['species_num'] = iris.species.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

# Create a scatterplot of PETAL LENGTH versus PETAL WIDTH and color by SPECIES
iris.plot(kind='scatter',x='petal_length', y='sepal_width', c='species_num', colormap=cmap_bold)
# Create a scatterplot of SEPAL LENGTH versus SEPAL WIDTH and color by SPECIES
iris.plot(kind='scatter',x='sepal_length', y='sepal_width', c='species_num', colormap=cmap_bold)

# ## K-nearest neighbors (KNN) classification
#Create your feature matrix "X"
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris[feature_cols]
# alternative ways to create "X"
X = iris.iloc[:, :-2]
X = iris.drop(['species', 'species_num'], axis=1)
X = iris.loc[:, 'sepal_length':'petal_width']
# Create your target vector "y"
y=iris.species_num
print(type(X))
print(type(X.values))
print(type(y))
print(type(y.values))
#Make Use of Train-Test-Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, random_state=1234)

#Import KNN From scikit-learn and Instatiate a Model With One Neighbor
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
type(knn)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
#Check the Accuracy
knn.score(X_test,y_test)
from sklearn.metrics import classification_report, accuracy_score
print(accuracy_score(y_test, y_pred ))

#Create a Model With Five Neighbors. Did it Improve?
knn = KNeighborsClassifier(n_neighbors=5)
type(knn)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

knn.score(X_test,y_test)
from sklearn.metrics import classification_report, accuracy_score
print(accuracy_score(y_test, y_pred ))

#Create a Looped Function That Will Check All Levels of Various Neighbors
#and Calculate the Accuracy
for i in range(1,20,4):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print('N_neighbors {}, Accuracy_socre {}'.format(i, accuracy_score(y_test, y_pred )))

#Bonus: According to scikit-learn Documentation,
#What is knn.predict_proba(X_new) Going to Do?
knn.predict_proba(X_test)

"""Enrichment"""
"""Using the Statsmodels Formula"""
# First, format our data in a DataFrame
df = pd.read_csv(r'C:\Users\bangli\Desktop\test\boston_data.csv')
df.head()
df.isnull().sum(axis=0)

# Set up our new statsmodel.formula handling model
import statsmodels.formula.api as smf
# You can easily swap these out to test multiple versions/different formulas
formulas = {
    "case1": "MEDV ~ RM + LSTAT + RAD + TAX + NOX + INDUS + CRIM + ZN - 1", # - 1 = remove intercept
    "case2": "MEDV ~ NOX + RM",
    "case3": "MEDV ~ RAD + TAX"
}
model = smf.ols(formula=formulas['case1'], data=df)
result = model.fit()
result.summary()

"""import patsy"""
# Add response to the core DataFrame
import patsy
from sklearn.model_selection import train_test_split #If you didn't import it earlier, do so now

# Easily change your variable predictors without reslicing your DataFrame
y, X = patsy.dmatrices("MEDV ~ AGE + RM", data=df, return_type="dataframe")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7)

from sklearn import metrics
from sklearn.linear_model import LinearRegression
# Rerun your model, iteratively changing your variables and train_size from the previous cell
lm = LinearRegression()
model = lm.fit(X_train, y_train)

predictions = model.predict(X_test)
print("R^2 Score: {}".format(metrics.r2_score(y_test, predictions)))



