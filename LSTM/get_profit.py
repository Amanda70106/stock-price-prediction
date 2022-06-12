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
    last = int(tokens[-1].replace('.0',''))
    now = int(tokens[-1].replace('.0',''))
    flag = 0
    lines = fpri.readlines()
    fclo_lines = fclo.readlines()
    cnt = 0
    for i in range(len(lines)):
        cnt += 1
        # if(cnt == 20):
        #     break
        lines[i] = lines[i].replace('\n','')
        tokens = lines[i].split(',')
        flag = int(tokens[-1].replace('.0','')) ^ last
        now = now if flag == 1 else int(tokens[-1].replace('.0',''))
        
        toks = fclo_lines[i].split(',')
        toks = float(toks[7])
        # print(toks)
        close.append(toks)
        result.append(int(now))
        
        last = int(tokens[-1].replace('.0',''))
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
                bucket += flag_0_start * vol - flag_0_end * vol - 0.005225 * flag_0_start * vol - 0.001425 * flag_0_end * vol
                # bucket += flag_0_start * vol - flag_0_end * vol
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



stock_list = [1210,1215,1216,1218,1227,1229,1231,1232,1434,1702,2330,2337,2344,2345,2379,2408,2412,2412,2439,2449,2454,2455,2459,2468,2498,2603,2607,2609,2610,2615,2618,2633,2634,2637,3034,3035,3045,3596,3682,4904]
# stock_list = [1210]


total = 0.0
acc_path = "./percisionRate_LSTM.csv"
facc = open(acc_path,'r')
line = facc.readline()
f = open('LSTM_profit.txt','a')

for stock in stock_list:
    res = []
    close = []
    pri_stock_path = "./output/30_" + str(stock) + "_index.csv"
    
    tokens = facc.readline().split(',')
    pre_0 = float(tokens[1])
    pre_1 = float(tokens[2])
    pre_theshold = 0.9
    take_0 = pre_0 > pre_theshold
    take_1 = pre_1 > pre_theshold
    res, close = get_result_and_close(stock,pri_stock_path)
    buc = simple_cnt(res,close,take_0,take_1)
    total += buc
    # print(str(buc) + '\t' + str(total))
print(total)
f.write(str(pre_theshold) + " :\t" + str(total) + "\n")

    
    # print(result)
    # print(close)
    
          

