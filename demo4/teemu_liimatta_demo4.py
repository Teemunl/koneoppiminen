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
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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

parameters = [{'solver': ['sgd'], 'max_iter': [125000],
              'alpha':  [(0.0001)], 'hidden_layer_sizes':  [(10)],
              'early_stopping': [True]},
              {'solver': ['sgd'], 'max_iter': [25000],
              'alpha':  [(0.0001)], 'hidden_layer_sizes':  [(10)],
              'early_stopping': [False]},
              {'solver': ['sgd'], 'max_iter': [125000],
              'alpha':  [(0.0001)], 'hidden_layer_sizes':  [(5,5)],
              'early_stopping': [True]},
              {'solver': ['sgd'], 'max_iter': [35000],
              'alpha':  [(0.0001)], 'hidden_layer_sizes':  [(3,2,5)],
              'early_stopping': [False]},
              {'solver': ['sgd'], 'max_iter': [35000],
              'alpha':  [(0.00001)], 'hidden_layer_sizes':  [(5,5)],
              'early_stopping': [True]},
              {'solver': ['adam'], 'max_iter': [35000],
              'alpha':  [(0.00001)], 'hidden_layer_sizes':  [(5,5)],
              'early_stopping': [True]},
              {'solver': ['adam'], 'max_iter': [25000],
              'alpha':  [(0.00001)], 'hidden_layer_sizes':  [(3,2,5)],
              'early_stopping': [False]},]


scores = [ 'precision' , 'recall']

start_time = time.time()

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    X_scaled = StandardScaler().fit_transform(X_train)
    grid = GridSearchCV(
        MLPClassifier(), parameters , scoring='%s_macro' % score, cv = 10)
        
    grid.fit(X_scaled, y_train)

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
    print()
    print("Laskenta-aika")
    print(total_time)
    
"""
SVC esimerkki
"""

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

start_time = time.time()
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    total_time = time.time() - start_time
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, grid.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    print("Laskenta-aika")
    print(total_time)

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.

"""
Precision kertoo luokittelijan todellisten positiivisten (oikeiden ennusteiden) suhteen
positiivisiin ennusteisiin. 
recall kertoo luokittelijan todellisten positiivisten suhteen kaikkiin tapauksiin
f1-score on harmoninen keskiarvo precisionista ja recallista 
Tarkin ja nopein luokitin oli kNN luokitin, jolla sain luokittelutarkuudeksi 0.9933
toiseksi tuli SVC menetelmä, joka oli hieman hitaampi kuin kNN, viimeisenä oli MLP, jolla sain
huonoimman ja hitaimman tuloksen
Luokat, jotka menivät usemmin sekaisin olivat 9,8 ja 5
"""