import pickle
from sklearn.datasets import load_diabetes
from sklearn import svm, ensemble
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import inspect
from inspect import signature
import os
import sys
from utils import *

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = base_dir+"\\data\\data_class.xlsx"     #might need to change this later to be more flexible
output_dir = base_dir + "\\data\\model_output.xlsx"
model_dir = base_dir+"\\models\\model.p"

X,y = excel_to_data(data_dir) #get the data


models =    {
                'Linear Regression' : LinearRegression,
                'SVR' : svm.SVR,
                'RFG' : ensemble.RandomForestRegressor,
                'Logistic Regression' : LogisticRegression,
                'Decision Tree' : DecisionTreeClassifier
            }


print("Which algorithm would you like to use?\n")
for model in models:
    print(model)

while(True): #asks user for choice of model
    choice = input("\n")
    model = models.get(choice)
    if model is not None:
        print("You chose" , model)
        break
    if choice == "quit":
        sys.exit()
    print("You did not select a valid option, please enter again, to quit enter \"quit\"")

'''
sig = signature(model)

for param in sig.parameters.values():
    print(param)

Add parameter support here^^^ with the signature module
'''

model = model()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model.fit(X_train,y_train) #this is where we train the model

pickle.dump(model, open(model_dir,"wb")) #save the model with pickle
