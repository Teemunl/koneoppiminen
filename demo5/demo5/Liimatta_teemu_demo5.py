# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:56:40 2021

@author: teemu
"""


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as  pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.inspection import permutation_importance
plt.style.use(['ggplot'])

df = pd.read_csv('Skyserver_12_30_2019 4_49_58 PM.csv') 
df.head()
df.shape
df.describe()


labels = {'STAR':1, 'GALAXY':2, 'QSO':3}
df.replace({'class':labels}, inplace = True)
print(df.head())

X = df.drop('class', axis = 1).values

y = df['class'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

rfc = RandomForestClassifier(n_estimators=100, #The number of trees in the forest.
                             criterion="gini", #The function to measure the quality of a split.
                             max_depth=None, #The maximum depth of the tree.
                             min_samples_split=2, #The minimum number of samples required to split an internal node.
                             min_samples_leaf=1, #The minimum number of samples required to be at a leaf node.
                             max_features="auto") #The number of features to consider when looking for the best split
                                                  #When "auto", max_features=sqrt(n_features)
                                                  
rfc.fit(X_train, y_train)

print('Opetusaineisto:')
y_train_preds = rfc.predict(X_train)
print(classification_report(y_train, y_train_preds))
print(print(confusion_matrix(y_train, y_train_preds)))
print()
print('Testiaineisto:')
y_test_preds = rfc.predict(X_test)
print(classification_report(y_test, y_test_preds))
print(print(confusion_matrix(y_test, y_test_preds)))


importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), df.drop('class', axis = 1).columns.values[indices], rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.show()
# Plot the permutation importances
result = permutation_importance(rfc, X_train, y_train, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=df.drop('class', axis = 1).columns[sorted_idx])
ax.set_title("Permutation Importances ")
fig.tight_layout()
plt.show()

"""
Permutation_importancen mukaan muuttuja u olisi paljon tärkeämpi kuin feature_importance antaa ymmärtää
feature_importance mukaan toisaalta, jokaisen muuttuja on keskimäärin tärkeämpi verrattuna permutation_importanceen
,joka antaa vaikuttaviksi muuttujiksi vain u:n ja redshiftin.
"""

