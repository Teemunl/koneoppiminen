# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:51:43 2021

@author: teemu
"""

# Tarvittavia moduuleja
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor

"""
#teht.1
"""

X,y = datasets.load_diabetes(return_X_y=True)
"""
#teht2
"""

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.33,random_state = 0)
"""
#teht3
"""

print("#teht3 ,Opetusdatan syöte- ja vastematriisin dimensiot ", X_train.shape,y_train.shape )
print("Testidatan syöte- ja vastematriisin dimensiot ",X_test.shape,y_test.shape)
"""
#teht4 ja 5
"""
scaler = MinMaxScaler()
scaledX = scaler.fit_transform( X_train)
scaledTest = scaler.transform(X_test)
print("teht4 ,Skaalauksen jälkeisen opetusdatan sarakkaiden min ja max arvot",scaledX.max(),scaledX.min())
print("teht5 ,Skaalauksen jälkeisen testidatan sarakkaiden min ja max arvot",scaledTest.max(),scaledTest.min())

"""
#teht 6
"""

reg = SGDRegressor(alpha=0.0, tol=1e-3, max_iter=1000)
reg.fit(scaledX,y_train)
"""
#teht7
"""
y_train_pred = reg.predict(X_train)

train_mse = mean_squared_error(y_train, y_train_pred)
print("teht7 ,train mse ", train_mse, "alpha = 0.0")

"""
#teht8
"""
y_test_pred = reg.predict(X_test)

test_mse = mean_squared_error(y_test,y_test_pred)

print("teht8 ,test mse", test_mse, "alpha = 0.0")

"""
teht 9,10
"""

reg = SGDRegressor(alpha=2.5, tol=1e-3, max_iter=1000)
reg.fit(scaledX,y_train)
y_train_pred = reg.predict(X_train)

train_mse = mean_squared_error(y_train, y_train_pred)
print("teht10 ,train mse ", train_mse, "alpha = 1.5")
y_test_pred = reg.predict(X_test)

test_mse = mean_squared_error(y_test,y_test_pred)

print("teht10 ,test mse", test_mse, "alpha = 1.5")

"""
Alphan etsiminen olisi varmaan hyvä toteuttaa jollain funktiolla, joka etsisi
alphan, jolla saadaan tarkin arvo,mikä päivittyisi datan lisääntyessä. 
"""
plt.scatter(X_train,y_test,  color='black')
plt.plot(X_test, y_test_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
