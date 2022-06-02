from cProfile import label
from attr import attr
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
from requests import head
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

datasets = pd.read_csv('2308_TrainingData.csv',encoding='unicode_escape')
HistoricalPrice = datasets.tail(2600)
NewPrice = datasets.head(32)
attributes = HistoricalPrice#.drop('StockclusterLabel',axis=1)
attributes = attributes.drop('X10',axis=1)
label = HistoricalPrice['X10']
RFRegressor = RandomForestRegressor(n_estimators=501, criterion='squared_error',n_jobs=-1)
print(RFRegressor)
n_score = cross_val_score(RFRegressor,attributes,label,scoring='neg_mean_squared_error',cv=10,n_jobs=-1)
print("MSE: %.3f (%.3f)"%(mean(n_score),std(n_score)))
learned_model = RFRegressor.fit(attributes,label)
test_attributes = NewPrice.drop('X10',axis=1)
test_attributes = test_attributes.drop('StockclusterLabel',axis=1)
test_lable = NewPrice['X10']
y_prediction = learned_model.predict(test_attributes)
print("Mean Absolute Error:",metrics.mean_absolute_error(test_lable,y_prediction))
print("Mean Squared Error:",metrics.mean_squared_error(test_lable,y_prediction))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(test_lable,y_prediction)))
prediction_result = pd.DataFrame(NewPrice)
prediction_result['Prediction_Result']= y_prediction
prediction_result.to_csv('Stock_Price_Prediction_RF.csv',mode = 'a', header=True)
