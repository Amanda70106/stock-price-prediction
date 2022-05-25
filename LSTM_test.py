import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix



def normalize(data):
  norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
  return norm

def denormalize(original_data, scaled_data):
  denorm = scaled_data.apply(lambda x: x*(np.max(original_data)-np.min(original_data))+np.min(original_data))
  return denorm

df = pd.read_csv('csv/2454.csv')
df = df.drop(['dividend', 'PE', 'netWorth', 'date', 'diff'], axis=1)
split_boundary = 2000

train_data = df.head(split_boundary)
test_data = df.tail(df.shape[0] - split_boundary)

X_test = []  
Y_test = np.empty(shape=[0, 1])
predict = np.empty(shape=[0, 1])
test_data_scaled = normalize(test_data)

for i in range(60, test_data.shape[0]):
    X_test.append(test_data_scaled.iloc[i-60:i, :-1])
    Y_test = np.append(Y_test, test_data.iloc[i:i+1, 5:6], axis=0)
    predict = np.append(predict, test_data_scaled.iloc[i:i+1, 5:6], axis=0)

X_test = np.array(X_test)

predict_df = pd.DataFrame(predict)
print(np.max(Y_test))
print(np.min(Y_test))

Y_denormalized = denormalize(Y_test, predict_df)
print(predict)
# print(Y_denormalized)