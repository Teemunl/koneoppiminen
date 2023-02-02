# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:27:33 2021

@author: teemu
"""

import time as time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
# Datasetin lataus 
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
    
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


"""
kNN menetelmä 
"""
#asetetaan k arvo välille 1-20

k_range = range(1, 20)
scoresUniform = []
total_time = 0
classification_reportUniform= []
#hakee parhaan tuloksen kun paino arvo on uniform
for k in k_range:
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=k,weights='uniform')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scoresUniform.append(accuracy_score(y_test, y_pred))
        total_time = time.time() - start_time
        classification_reportUniform.append(
        classification_report(y_test, y_pred))

bestkNNUniform = [scoresUniform.index(max(scoresUniform))-1,max(scoresUniform),classification_reportUniform[scoresUniform.index(max(scoresUniform))]]

print("Paras konfiguraatio, jos painoarvo uniform")
print("Luokittelutarkkuus %.4f" % bestkNNUniform[1])
print(bestkNNUniform[2])
print("Aika %.2f" % total_time)
print()
classification_reportDistance= []
scoresDistance = []
#hakee parhaan tuloksen kun paino arvo on distance
for k in k_range:
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=k,weights='distance')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scoresDistance.append(accuracy_score(y_test, y_pred))
        total_time = time.time() - start_time
        classification_reportDistance.append(
        classification_report(y_test, y_pred))
        
bestkNNDistance = [scoresDistance.index(max(scoresDistance))-1,max(scoresDistance),classification_reportDistance[scoresDistance.index(max(scoresDistance))]]

print("Paras konfiguraatio, jos painoarvo distance")
print("Luokittelutarkkuus %.4f" % bestkNNDistance[1])
print(bestkNNDistance[2])
print("Aika %.2f" % total_time)

"""
MLP menetelmä

"""

parameters = {'solver': ['sgd','adam'], 'max_iter': [15000,35000,65000 ],
              'alpha':  [0.0001,0.0002,0.0003 ], 'hidden_layer_sizes':np.arange(1, 7),
              'early_stopping': [True],'random_state':[0,1,2,3,4,5,6,7,8,9]}


scores = [ 'precision'  ,  'recall ', 'f1-score'  , 'support']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    grid = GridSearchCV(
        MLPClassifier(), parameters , scoring='%s_macro' % score, cv = 10)
        
    grid.fit(X_train, y_train)

    print("Parhaat opetusdatalla löydetyt parametrit ovat")
    print()
    print(grid.best_params_)
    print()
    print("GridSearch tulokset:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    total_time = time.time() - start_time
    
    print("Luokitteluraportti:")
    print()
    print(" opetusdatalla")
    print("Suorituskyky on arvioitu testidatalla")
    print()
    y_true, y_pred = y_test, grid.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("Sekaannusmatriisi")
    print()    
    print(print(confusion_matrix(y_true, y_pred)))
    print()
    print("Laskenta-aika")
    print(total_time)
    
