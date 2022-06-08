import pandas as pd
import numpy as np
stock_list = [1210, 1231, 2344, 2449, 2603, 2633, 3596, 1215, 1232, 2345, 2454, 2607, 2634, 3682, 1216, 1434, 2379, 2455, 2609,
              2637, 4904, 1218, 1702, 2408, 2459, 2610, 3034, 5388, 1227, 2330, 2412, 2468, 2615, 3035, 1229, 2337, 2439, 2498, 2618, 3045]
for stock in stock_list:
    algorithm = input('Input the algorithm name: ')
    input_file = "../" + algorithm + "/output/" + \
        algorithm + str(stock) + ".csv"
    output_file = "backtest/" + algorithm + "/" + str(stock) + ".csv"
    df = pd.read_csv(input_file)
    f = open(output_file, 'w', encoding='utf-8')

    profit = []
    borrowsell_mark = []
    borrowbuy_mark = []
    buy_mark = []
    sell_mark = []
    tag = 2
    for i in range(len(df)-1):
        if df["Prediction_Result"][i] == 1 and df["Prediction_Result"][i+1] == 1:
            if tag != 1:
                sell_mark.append(np.nan)
                sell_mark.append(np.nan)
                buy_mark.append(np.nan)
                buy_mark.append(df["close"][i+1])
                tag = 1
            else:
                sell_mark.append(np.nan)
                buy_mark.append(np.nan)
        elif df["Prediction_Result"][i] == 0 and df["Prediction_Result"][i+1] == 0:
            if tag == 1:
                buy_mark.append(np.nan)
                buy_mark.append(np.nan)
                sell_mark.append(np.nan)
                sell_mark.append(df["close"][i+1])
            else:
                buy_mark.append(np.nan)
                sell_mark.append(np.nan)
                tag = 0

