# facc = open('./precisionRate_ann.csv','r')

from unittest import result
from more_itertools import bucket
from math import floor

def get_result_and_close(stock,pri_stock_path):   
    close_stock_path = "../csv/index/" + str(stock) + "_index.csv"
    fclo = open(close_stock_path,'r')
    fpri = open(pri_stock_path,'r')
    close = []
    result = []  
    fclo_lines = fclo.readline()
    fclo_lines = fclo.readline()  
    lines = fpri.readline()
    line = fpri.readline().replace('\n','')  
    tokens = line.split(',')
    last = int(tokens[-1])
    now = int(tokens[-1])
    flag = 0
    lines = fpri.readlines()
    fclo_lines = fclo.readlines()
    cnt = 0
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n','')
        tokens = lines[i].split(',')
        flag = int(tokens[-1]) ^ last
        now = now if flag == 1 else int(tokens[-1])
        toks = fclo_lines[i].split(',')
        toks = float(toks[7])
        # print(toks)
        close.append(toks)
        result.append(int(now))
        last = int(tokens[-1])
    return result, close


def simple_cnt(result, close, take_0, take_1):
    bucket = 0.0
    flag_1_start = 0.0
    flag_1_end = 0.0
    flag_0_start = 0.0
    flag_0_end = 0.0
    target_price = 10000  
    for i in range(len(result)):
        if(result[i] == 1):
            flag_1_start = close[i] if flag_1_start == 0 else flag_1_start
            flag_1_end = close[i]
            vol = 0
            if take_0:
                if flag_0_start != 0:
                    vol = floor(target_price/flag_0_start)
                # bucket += flag_0_start * vol - flag_0_end * vol 
                bucket += flag_0_start * vol - flag_0_end * vol - 0.005225 * flag_0_start * vol - 0.001425 * flag_0_end * vol
            flag_0_end = 0
            flag_0_start = 0
        elif(result[i] == 0):
            flag_0_start = close[i] if flag_0_start == 0 else flag_0_start
            flag_0_end = close[i]
            vol = 0
            if take_1:
                if flag_1_end != 0:
                    vol = floor(target_price/flag_1_start)
                bucket += flag_1_end * vol - flag_1_start * vol - 0.001425 * flag_1_start - 0.004425 * flag_1_end * vol
                # bucket += flag_1_end * vol - flag_1_start * vol 
            flag_1_start = 0
            flag_1_end = 0
    return bucket



stock_list = [1210,1231,2344,2449,2603,2633,3596,1215,1232,2345,2454,2607,2634,3682,1216,1434,2379,2455,2609,2637,4904,1218,1702,2408,2459,2610,3034,5388,1227,2330,2412,2468,2615,3035,1229,2337,2439,2498,2618,3045]
total = 0.0
acc_path = "./precisionRate_ann.csv"
facc = open(acc_path,'r')
line = facc.readline()
f = open('ann_profit.txt','a')
for stock in stock_list:
    res = []
    close = []
    pri_stock_path = "./pridiction_result/" + str(stock) + ".csv"   
    tokens = facc.readline().split(',')
    pre_0 = float(tokens[1])
    pre_1 = float(tokens[2])
    pre_theshold = 0    
    take_0 = pre_0 > pre_theshold
    take_1 = pre_1 > pre_theshold
    # print(take_0)
    # print(take_1)
    res, close = get_result_and_close(stock,pri_stock_path)
    buc = simple_cnt(res,close,take_0,take_1)
    total += buc

print(total)
# print(close)

        

