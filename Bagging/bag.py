from cProfile import label
import imp
from numpy import mean
from numpy import std
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier

datasets = pd.read_csv('UCI_Credit_Card.csv')
CurrentCustomers=datasets.head(25000)
NewCustomers=datasets.tail(5000)

NewCustomers.shape

attributes=CurrentCustomers.drop('default.payment.next.month',axis=1)
label=CurrentCustomers['default.payment.next.month']
model = BaggingClassifier(base_estimator=None,n_estimators=500,n_jobs=-1)
n_score=cross_val_score(model,attributes,label,scoring='f1_macro',cv=10,n_jobs=-1,error_score='raise')
print(n_score)
print('F-Score: %.3f (%.3f)'%(mean(n_score),std(n_score)))
learned_model=model.fit(attributes,label)
test_attributes = NewCustomers.drop('default.payment.next.month',axis=1)
test_label=NewCustomers['default.payment.next.month']
y_prediction = learned_model.predict(test_attributes)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(test_label,y_prediction))
print(classification_report(test_label,y_prediction))
Predict_result = pd.DataFrame(test_attributes)
Predict_result["Prediction_Result"] = y_prediction
Predict_result.to_csv('prediction_Result_Bagging_Esamble.csv',mode='a',header=True)
