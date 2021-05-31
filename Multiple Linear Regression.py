"""
Created on Mon Dec 21 23:41:27 2020

@author: Emmanuel Hammond
"""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset and defining variables
dataset = pd.read_csv("C:/Users/User/Desktop/HashAnalytic Internship/Data Science 1/5. Regression  Multiple Linear Regression/3.2 Org_data.csv.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

#Enoding Indepenent variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [("onehotencoder", OneHotEncoder(categories= 'auto'),[3])],
                       remainder = "passthrough")
X = np.array(ct.fit_transform(X), dtype = np.float)

# Avoiding dummy variable trap
X = X[:, 1:]

# Splitting data into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# Fitting model using the train
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
regressor.score(X_train, Y_train)#(returns the coefficient of determination)

#Predicting the fitted model using the test set
Y_pred = regressor.predict(X_test)

#Plotting fiited values against test set
x1 = np.arange(1,11)
plt.plot(x1,Y_test,'b',x1,Y_pred,'r')


