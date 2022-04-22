# from sklearn import tree
# from sklearn import datasets
# import pydotplus

f = open('3035.txt','r') # need to modify the name
fout = open('3035.csv','w',encoding='utf-8') # need to modify the name

fl = f.readlines()

for line in fl:
    line = line.replace(',','')
    if(line.find('+')!=-1):
        line = line.replace('\n',',1\n')
    else:
        line = line.replace('\n',',0\n')
    line = line.replace(' ',',')
    fout.write(line)
