from cgi import test
import os
from xml.dom.minidom import TypeInfo
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
from requests import head
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import math
def normalize(data):
  norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
  return norm

def denormalize(original_data, scaled_data):
  denorm = scaled_data.apply(lambda x: x*(np.max(original_data)-np.min(original_data))+np.min(original_data))
  return denorm
# def mathew(original, predict):
#     tn, fp, fn, tp = confusion_matrix(original,predict).ravel()
#     return (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))



stock_list = [1210,1231,2344,2449,2603,2633,3596,1215,1232,2345,2454,2607,2634,3682,1216,1434,2379,2455,2609,2637,4904,1218,1702,2408,2459,2610,3034,5388,1227,2330,2412,2468,2615,3035,1229,2337,2439,2498,2618,3045]
for stock in stock_list:
  input_file = "../csv/" + str(stock) + ".csv"
  output_file = "outputr/" + str(stock) + "_result.txt"
  csv_file = "../csv/pridiction_result/" + str(stock) + ".csv"
  df = pd.read_csv(input_file)
  f = open(output_file, 'w',encoding='utf-8')

  HistoricalPrice = df.head(2900)
  NewPrice = df.tail(53)
  attributes = HistoricalPrice.drop(['X','data','diff','result'],axis=1)
  attributes = attributes.drop('close',axis=1)
  label = HistoricalPrice['close']
  RFRegressor = RandomForestRegressor(n_estimators=501, criterion='squared_error',n_jobs=-1)
  print(RFRegressor)
  n_score = cross_val_score(RFRegressor,attributes,label,scoring='neg_mean_squared_error',cv=10,n_jobs=-1)
  print("MSE: %.3f (%.3f)"%(mean(n_score),std(n_score)))
  learned_model = RFRegressor.fit(attributes,label)
  test_attributes = NewPrice.drop('close',axis=1)
  test_attributes = test_attributes.drop(['X','data','diff','result'],axis=1)
  test_lable = NewPrice['close']
  y_prediction = learned_model.predict(test_attributes)
  print("Mean Absolute Error:",metrics.mean_absolute_error(test_lable,y_prediction))
  print("Mean Squared Error:",metrics.mean_squared_error(test_lable,y_prediction))
  print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(test_lable,y_prediction)))
  prediction_result = pd.DataFrame(NewPrice)
  #print(y_prediction)
  prediction_result['Prediction_Result']= y_prediction
  output_filename = "outputr/RandomForestR" + str(stock) +".csv"
  prediction_result.to_csv(output_filename,mode='w' ,header = True, index = False)
