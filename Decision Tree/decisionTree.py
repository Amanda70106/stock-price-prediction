import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import metrics
stock_list = [1210,1231,2344,2449,2603,2633,3596,1215,1232,2345,2454,2607,2634,3682,1216,1434,2379,2455,2609,2637,4904,1218,1702,2408,2459,2610,3034,5388,1227,2330,2412,2468,2615,3035,1229,2337,2439,2498,2618,3045]
for stock in stock_list:
    print(stock)
    input_file = "../csv/" + str(stock) + ".csv"
    output_file = "./output/" + str(stock) + "_result.txt"
    csv_file = "./pridiction_result/" + str(stock) + ".csv"
    df = pd.read_csv(input_file)
    df = df.drop(['data','diff', 'X'], axis=1)
    f = open(output_file, 'w',encoding='utf-8')
    # print(df)

    CurrentCustomers = df.head(2900)
    NewCustomers = df.tail(39)
    attributes = CurrentCustomers.drop('result',axis=1)
    label = CurrentCustomers['result']

    DT = DecisionTreeClassifier(criterion='entropy')
    #print(DT)

    scores = cross_val_score(DT, attributes, label, cv=7, scoring='f1_macro',n_jobs=1)
    #print(scores)
    #print("F-score: %0.2f (+/= % 0.2f)" % (scores.mean(),scores.std()*2))

    DT_Model = DT.fit(attributes,label)

    feature_names = attributes.columns[:13]
    # fig = plt.figure(figsize=(50,200))
    # _ = tree.plot_tree(DT_Model, feature_names=feature_names, class_names='result', filled = True)


    test_attributes = NewCustomers.drop('result',axis=1)
    test_label = NewCustomers['result']
    y_prediction = DT_Model.predict(test_attributes)
    f.write(str(confusion_matrix(test_label,y_prediction)))
    f.write('\n')
    f.write(str(classification_report(test_label,y_prediction)))
    # print(confusion_matrix(test_label,y_prediction))
    # print(classification_report(test_label,y_prediction))

    Prediction_result = pd.DataFrame(test_attributes)
    Prediction_result["Prediction_Result"]=y_prediction

    Prediction_result.to_csv(csv_file,mode ='w', header = True)
    #plt.show()
