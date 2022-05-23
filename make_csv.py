stock = input("input the stock number:")

f1name = "csv/" + stock + "_all.csv"
f2name = "csv/" + stock + "_all_2.csv"
foutname = "csv/" + stock + ".csv"
# f1name = str(stock) + "_all.csv"
# f2name = str(stock) + "_all_2.csv"

f1 = open(f1name,'r',encoding='utf-8')
f2 = open(f2name,'r',encoding='utf-8')

f = open(foutname, 'w', encoding='utf-8')

f.write("date,dividend,PE,netWorth,volume,price,open,high,low,close,diff,times,result\n")

f1lines = f1.readline()
f1lines = f1.readlines()
f2lines = f2.readline()
f2lines = f2.readlines()
for i in range(len(f2lines)):
    pos = f1lines[i].find(',')
    f.write(f1lines[i][:pos].replace('-','0'))
    f.write(f2lines[i][10:-2].replace('-','0'))
    f.write(f1lines[i][pos+1:])