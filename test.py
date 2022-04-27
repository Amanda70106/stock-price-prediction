from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
 
 
# options = Options()
# options.add_argument("--disable-notifications")
 
chrome = webdriver.Chrome('./chromedriver.exe') 
chrome.get("https://www.twse.com.tw/zh/page/trading/exchange/STOCK_DAY.html")
time.sleep(10)
stock_list = ['2317','2330'] # can adjust the stock list
for stock in stock_list:
	file = stock + '.txt'
	f = open(file,'w',encoding='utf-8')

	element = chrome.find_element_by_class_name("stock-code-autocomplete")
	element.send_keys(stock)
	for k in range(1):
		year_xpath = "//*[@id=\"d1\"]/select[1]/option["+str(13-k)+"]"
		select_year = chrome.find_element_by_xpath(year_xpath).click()
		for i in range(1):
			month_xpath ="//*[@id=\"d1\"]/select[2]/option["+str(i+1)+"]"
			select_month = chrome.find_element_by_xpath(month_xpath).click()
			time.sleep(10)
			chrome.find_element_by_link_text("查詢").click()
			time.sleep(5)
			for j in range(28):
				line = ""
				try:
					xpath = "//*[@id=\"report-table\"]/tbody/tr[" + str(j+1) + "]"
					line += chrome.find_element_by_xpath(xpath).text
					line += "\n"
					line = line.replace('.','')
					if(line.find('+')!=-1):
						line = line.replace('\n',',1\n')
					else:
						line = line.replace('\n',',0\n')
					line = line.replace(' ',',')
					fout.write(line)
					# f.write(chrome.find_element_by_xpath(xpath).text)
					# f.write('\n')
				except:
					continue
	f.close()
	element.clear()
				
chrome.close()
