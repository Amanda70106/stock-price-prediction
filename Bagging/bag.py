
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
def normalize(data):
  norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
  return norm
import math
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
  NewCustomers=df.tail(939)

  NewCustomers.shape
 # print(CurrentCustomers)
  attributes=CurrentCustomers.drop(['X','data','diff','result'],axis=1)

  #the result with normalization is worse than the one without normalization
  attributes = normalize(attributes)
  label=CurrentCustomers['result']
  model = BaggingClassifier(base_estimator=None,n_estimators=500,n_jobs=-1)
  n_score=cross_val_score(model,attributes,label,scoring='f1_macro',cv=10,n_jobs=-1,error_score='raise')
  print(n_score)
  print('F-Score: %.3f (%.3f)'%(mean(n_score),std(n_score)))
  learned_model=model.fit(attributes,label)
  test_attributes = NewCustomers.drop(['X','data','diff','result'],axis=1)
  original = test_attributes
  test_attributes = normalize(test_attributes)
  test_label=NewCustomers['result']
  y_prediction = learned_model.predict(test_attributes)
  from sklearn.metrics import classification_report,confusion_matrix
  print(confusion_matrix(test_label,y_prediction))
  print(classification_report(test_label,y_prediction))
  m=mathew(test_label,y_prediction)
  Predict_result = pd.DataFrame(original)
  Predict_result["Prediction_Result"] = y_prediction
  output_filename = "output/Bagging_" + str(stock) +".csv"
  print(output_file)
  Predict_result.to_csv(output_filename,mode='w' ,header = True, index = False)
  f.write(str(confusion_matrix(test_label,y_prediction)))
  f.write('\n')
  f.write(str(classification_report(test_label,y_prediction)))
  f.write('\n')
  f.write("matthews: %.4f" % (m))
