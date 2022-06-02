stock = input("input the stock name:")
stock_pos = "csv/" + stock + "_all.csv"
f = open(stock_pos,'r',encoding='utf-8')

lines = f.readline()
lines = f.readlines()
line_cnt = 0
float_set = [3,4,5,6]
int_set = [1,2,8]
error_cnt = 0
for line in lines:
    line_cnt += 1 
    tokens = line.split(',')
    if(len(tokens)!=10):
        print("error tokens number in lines " + str(line_cnt))
        error_cnt += 1
    dates = tokens[0].split('/')
    for date in dates:
        if(not date.isdigit()):
            print("error date in lines " + str(line_cnt))
            error_cnt += 1
    for i in float_set:
        if(not tokens[i].replace('.','').isdigit()):
            print("error " + str(i) + "th data in lines " + str(line_cnt))
            error_cnt += 1
    for i in int_set:
        if(not tokens[i].isdigit()):
            print("error " + str(i) + "th data in lines " + str(line_cnt))
            error_cnt += 1
    if(not (tokens[9].replace('\n','') == "1" or tokens[9].replace('\n','') == "0")):
        print("error at result in lines " + str(line_cnt))
        error_cnt += 1
if(error_cnt == 0):
    print("this csv file is correct")
else:
    print("tish csv file has " + str(error_cnt) + "errors.")