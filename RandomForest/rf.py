
from numpy import mean
from numpy import std
import os
import math
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np

def normalize(data):
  norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
  return norm
def mathew(original, predict):
    tn, fp, fn, tp = confusion_matrix(original,predict).ravel()
    return (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
stock_list = [1210,1231,2344,2449,2603,2633,3596,1215,1232,2345,2454,2607,2634,3682,1216,1434,2379,2455,2609,2637,4904,1218,1702,2408,2459,2610,3034,5388,1227,2330,2412,2468,2615,3035,1229,2337,2439,2498,2618,3045]
for stock in stock_list:
  print(stock)
  input_file = "../csv/" + str(stock) + ".csv"
  output_file = "output/" + str(stock) + "_result.txt"
  csv_file = "../csv/pridiction_result/" + str(stock) + ".csv"
  df = pd.read_csv(input_file)
  f = open(output_file, 'w',encoding='utf-8')

  CurrentCustomers=df.head(2000)
  NewCustomers=df.tail(52)
  NewCustomers.shape

  attributes=CurrentCustomers.drop(['data','diff','result'],axis=1)
  label=CurrentCustomers['result']
  #attributes = normalize(attributes)
  RFClassfier = RandomForestClassifier(criterion='gini',n_estimators=500,n_jobs=-1)
  print(RFClassfier)
  n_score = cross_val_score(RFClassfier,attributes,label,scoring='f1_macro',cv=10,n_jobs=-1)
  print('F-Score: %.3f (%.3f)'%(mean(n_score),std(n_score)))

  learned_model=RFClassfier.fit(attributes,label)

  test_attributes = NewCustomers.drop(['data','diff','result'],axis=1)
  test_label=NewCustomers['result']
  original = test_attributes
  #test_attributes = normalize(test_attributes)
  y_prediction = learned_model.predict(test_attributes)
  from sklearn.metrics import classification_report,confusion_matrix
  print(confusion_matrix(test_label,y_prediction))
  print(classification_report(test_label,y_prediction))
  m=mathew(test_label,y_prediction)
  Predict_result = pd.DataFrame(original)
  output_filename = "output/RandomForest" + str(stock) +".csv"
  Predict_result.to_csv(output_filename,mode='a' ,header = True, index = False)
  f.write(str(confusion_matrix(test_label,y_prediction)))
  f.write('\n')
  f.write(str(classification_report(test_label,y_prediction)))
  f.write('\n')
  f.write("matthews: %.4f" % (m))
