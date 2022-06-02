import os
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
from requests import head
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

filename = input('Input the csv file name: ')
input_directory = os.path.abspath("../csv") + '/'
datasets = pd.read_csv(input_directory + filename)
output_directory = os.path.abspath('./regressorOutput/') 
if not os.path.isdir(output_directory):
  os.makedirs(output_directory)

CurrentCustomers=datasets.head(2000)
NewCustomers=datasets.tail(939)

attributes=CurrentCustomers.drop(['data','diff','result'],axis=1)
label=CurrentCustomers['result']
RFRegressor = RandomForestRegressor(n_estimators=501, criterion='squared_error',n_jobs=-1)
print(RFRegressor)
n_score = cross_val_score(RFRegressor,attributes,label,scoring='neg_mean_squared_error',cv=10,n_jobs=-1)
print("MSE: %.3f (%.3f)"%(mean(n_score),std(n_score)))
learned_model = RFRegressor.fit(attributes,label)
test_attributes = NewCustomers.drop(['data','diff','result'],axis=1)
test_label=NewCustomers['result']
y_prediction = learned_model.predict(test_attributes)
print("Mean Absolute Error:",metrics.mean_absolute_error(test_label,y_prediction))
print("Mean Squared Error:",metrics.mean_squared_error(test_label,y_prediction))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(test_label,y_prediction)))
prediction_result = pd.DataFrame(NewCustomers)
prediction_result['Prediction_Result']= y_prediction
output_filename = output_directory + "/RandomForestRegression_" + filename
prediction_result.to_csv(output_filename,mode='a' ,header = True, index = False)
