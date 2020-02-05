#-----------------------------------------------Logistic Regresion Lab
"""Step 1: Read the data into Pandas"""
import pandas as pd
bank = pd.read_csv(r'C:\Users\bangli\Desktop\test\bank.csv')
bank.head()

bank.isnull().sum(axis=0)
bank.columns.values.tolist()

bank.dtypes
bank.describe()

"""Step 2: Prepare at least three features"""
import seaborn as sns
sns.heatmap(bank.corr())
#Include both numeric and categorical features
#Choose features that you think might be related to the response (based on intuition or exploration)
#Think about how to handle missing values (encoded as "unknown")
feature_cols = ["duration","previous", "nr.employed" ]
X = bank[feature_cols]
y=bank.y

"""Step 3: Model building"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
# convert selected features do dummies
# set x and y
# train test splot
x_dummies = pd.get_dummies(X["previous"], drop_first = True)
X =  pd.concat([X,x_dummies], axis=1)
X.drop(["previous"], inplace=True, axis=1)
y.nunique()
y.unique()
# set the model
X_test, X_train, y_test, y_train =  train_test_split(X, y, test_size = 0.3, random_state = 1234)
# fit model
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)
logreg.score(X_test,y_test)
from sklearn.metrics import classification_report, accuracy_score
print(accuracy_score(y_test, y_pred ))
#Get the Coefficient for each feature.
print(logreg.coef_)
print(logreg.intercept_)
#Use the Model to predict on x_test and evaluate the model using metric(s) of Choice.
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
#visualization
bank['y_pred_prob'] = logreg.predict_proba(X)[:, 1]

import seaborn as sns
import matplotlib.pyplot as plt
plt.scatter(bank["nr.employed"] , bank.y);
plt.plot(bank["nr.employed"], bank['y_pred_prob'], color='red');
plt.xlabel('nr.employed');
plt.ylabel('Y');

#cross_val_score
from sklearn.model_selection import cross_val_score
logreg1 = LogisticRegression(C=1e9)
cross_val_score(logreg1, X, y, cv=10, scoring='roc_auc').mean()

"""Model 2: Use a different combination of features"""