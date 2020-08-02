import pickle
from sklearn import svm, ensemble
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from utils import *

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir = base_dir+"\\data\\data_class.xlsx"     #might need to change this later to be more flexible
model_dir = base_dir+"\\models\\model.p"
output_dir = base_dir + "\\data\\model_output.xlsx"


model = pickle.load(open(model_dir, 'rb')) #load the model with pickle
X,y = excel_to_data(data_dir)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) #random state is same as in train_model.py
prediction = model.predict(X_test) #testing the model

df_test = pd.DataFrame(X_test)
df_test['target'] = y_test
df_test['prediction'] = prediction
with pd.ExcelWriter(output_dir) as writer:
    df_test.to_excel(writer,sheet_name = 'testing data and prediction', index = False)
