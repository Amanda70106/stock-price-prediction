
from numpy import mean
from numpy import std
import os
import math
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def normalize(data):
  norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
  return norm
def mathew(original, predict):
    tn, fp, fn, tp = confusion_matrix(original,predict).ravel()
    return (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
def smoothCut(df,days):
    #moving average
    df['close']=df['close'].rolling(days).mean()
    #drop empty value
    df=df.drop(index=[0,1,2,3,4,5,6,7,8],axis=0)
    #calculate diff
    r, c = df.shape
    for i in range(r-1):
        df.iloc[i,8] =df.iloc[i+1,7]-df.iloc[i,7]
    for i in range(r-1):
        if df.iloc[i,8]>0:
            df.iloc[i,10]=1
        else:
            df.iloc[i,10]=0
    df.to_csv("middle.csv")
    return df
def profit(df,signal):#df is original input data (from csv file and make close more smooth)
                                   #y_prediction is predicted data (numpy array)
    #signal is numpy array
    signal = pd.DataFrame(signal)
    signal = signal.reset_index()
    signal = signal.iloc[:,1]
    df = df.tail(len(signal))
    df = df.reset_index()
    df = df['close']
    current = signal.iloc[0]
    money=0
    previous=current
    #print(signal)
    #print(df)
    for i in range(1,len(df)):# i->tomorrow
        if current!=signal.iloc[i] and previous==signal.iloc[i]:
            if current==0:#buy
                money = money-df.iloc[i-1]#扣今天的錢
                current=1
            else:   #sell
                money = money+df.iloc[i-1]
                current=0
        previous=signal.iloc[i]#前一天是要買還是賣
    return money
money = []

stock_list = [1210,1231,2344,2449,2603,2633,3596,1215,1232,2345,2454,2607,2634,3682,1216,1434,2379,2455,2609,2637,4904,1218,1702,2408,2459,2610,3034,5388,1227,2330,2412,2468,2615,3035,1229,2337,2439,2498,2618,3045]
for stock in stock_list:
  print(stock)
  input_file = "../csv/" + str(stock) + ".csv"
  output_file = "output/" + str(stock) + "_result.txt"
  csv_file = "../csv/pridiction_result/" + str(stock) + ".csv"
  df = pd.read_csv(input_file)
  f = open(output_file, 'w',encoding='utf-8')
  original = df
  df = smoothCut(df,10)
  CurrentCustomers=df.head(2900)


  NewCustomers=df.tail(52)
  #NewCustomers.shape

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
  # print(confusion_matrix(test_label,y_prediction))
  # print(classification_report(test_label,y_prediction))
  m=mathew(test_label,y_prediction)
  money.append(profit(original,y_prediction))
  Predict_result = pd.DataFrame(original)
  output_filename = "output/RandomForest" + str(stock) +".csv"
  Predict_result.to_csv(output_filename,mode='a' ,header = True, index = False)
  f.write(str(confusion_matrix(test_label,y_prediction)))
  f.write('\n')
  f.write(str(classification_report(test_label,y_prediction)))
  f.write('\n')
  f.write("matthews: %.4f" % (m))
d = {'stockID': stock_list, 'profit': money}
result = pd.DataFrame(data=d)
result.to_csv("output/profit.csv",header = True, index = False,mode='a')