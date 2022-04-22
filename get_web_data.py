from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
 
 
# options = Options()
# options.add_argument("--disable-notifications")
 
chrome = webdriver.Chrome('./chromedriver.exe')
chrome.get("https://www.twse.com.tw/zh/page/trading/exchange/STOCK_DAY.html")

stock_list = ['2317','2330']
for stock in stock_list:
    file = stock + '.txt'
    f = open(file,'w',encoding='utf-8')

    element = chrome.find_element_by_class_name("stock-code-autocomplete")
    element.send_keys(stock)
    for k in range(12):
        year_xpath = "//*[@id=\"d1\"]/select[1]/option["+str(13-k)+"]"
        select_year = chrome.find_element_by_xpath(year_xpath).click()
        for i in range(12):
            month_xpath ="//*[@id=\"d1\"]/select[2]/option["+str(i+1)+"]"
            select_month = chrome.find_element_by_xpath(month_xpath).click()
            time.sleep(0.5)
            chrome.find_element_by_link_text("查詢").click()
            time.sleep(2)
            for j in range(28):
                try:
                    xpath = "//*[@id=\"report-table\"]/tbody/tr[" + str(j+1) + "]"
                    f.write(chrome.find_element_by_xpath(xpath).text)
                    f.write('\n')
                except:
                    continue
    f.close()
    element.clear()
                
chrome.close()
