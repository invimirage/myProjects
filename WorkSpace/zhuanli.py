#!/usr/bin/env python
# coding=utf-8
from selenium import webdriver
import time

import re
import dbUtil

url = 'http://cpquery.sipo.gov.cn'  # 需要解析的网页
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('X-Frame-Options = SAMEORIGIN')
chrome_options.add_argument('X-Xss-Protection = 1; mode=block')
chrome_options.add_argument('lang=zh_CN.UTF-8')
driver = webdriver.Chrome(options=chrome_options)
# driver = webdriver.Chrome()
# driver.maximize_window()
url3=''
k=0

# 登陆网页


def login(ul1,name,pwd):
    driver.get(ul1)
    time.sleep(10)
    driver.find_element_by_id("username1").send_keys(name)
    driver.find_element_by_id('password1').send_keys(pwd)
    k=input("继续")
def getmesg(qiye):
    msgall=[]
    driver.get(url)
    time.sleep(1)
    driver.find_element_by_id("select-key:shenqingr_from").click()
    driver.find_element_by_id("select-key:shenqingr_from").send_keys("2015-1-1")
    time.sleep(0.5)
    driver.find_element_by_id("select-key:shenqingr_to").click()
    driver.find_element_by_id("select-key:shenqingr_to").send_keys("2019-6-30")
    time.sleep(0.5)
    driver.find_element_by_id("select-key:shenqingrxm").send_keys(qiye)
    code=input("输入验证码")
    driver.find_element_by_id('very-code').send_keys(code)
    driver.find_element_by_id("query").click()


    # time.sleep(8)

    # data - totalpage = "1030"
    text = driver.page_source
    # print(text)
    n=0
    num = re.findall('data-totalpage="(.*?)"', text, re.S)
    if(len(num)==0):
        return msgall
    sd=int(num[0])
    if(sd==0):
        return msgall

# 抓取数据

    while n<sd:
        print("第{}页".format(n+1))

        types=driver.find_elements_by_name("record:zhuanlilx")
        sqrs=driver.find_elements_by_name("record:shenqingr")
        sqgg=driver.find_elements_by_name("record:shouquanggr")
        names=driver.find_elements_by_name("record:shenqingrxm")
        zflh = driver.find_elements_by_name("record:zhufenlh")


        value = re.findall("javascript:jbxxAction(.*?);", text, re.S)
        i=0
        while i<len(types):
            msg=[types[i].text,names[i].text,sqrs[i].text,sqgg[i].text,zflh[i].text]
            # print(msg)
            msgall.append(msg)
            i += 1
        n+=1
        if n<sd:
           
            driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/ul/li[4]/a').click()

    #
    # # 获取当前页面
    # handle = driver.current_window_handle  # 获取当前页面标识
    # text = driver.page_source
    # # //获取页面地址
    # # 打印当前页面的URL地址
    #
    #
    # # print(driver.current_url)
    # for m in msgall:
    #     v = m[5]
    #     # print(url3)
    #     url4 = url3.replace(v, '{}').format(v)
    #     # print(url4)
    #     driver.get(url4)
    #     tt = driver.page_source
    #     value = re.findall('record_fmr:famingrxm" title="(.*?)">', tt, re.S)
    #     # print(value)
    #     m.append(value[0])
    return msgall





def getmesg2():
    driver.get(url)
    time.sleep(3)
    driver.find_element_by_id("select-key:shenqingr_from").send_keys("2015-1-1") #时间范围
    driver.find_element_by_id("select-key:shenqingr_to").send_keys("2019-6-30")
    driver.find_element_by_id("select-key:shenqingrxm").send_keys('神州高铁技术股份有限公司')
    code=input("输入验证码")
    driver.find_element_by_id('very-code').send_keys(code)
    driver.find_element_by_id("query").click()

    text = driver.page_source
    value = re.findall("javascript:jbxxAction(.*?);", text, re.S)

    driver.find_elements_by_class_name("bi_icon")[0].click()
    v = value[0].replace("('", '').replace("')", '')
    url2 = str(driver.current_url)
    global url3
    url3 = url2.replace(v, '{}')

# 账号和密码

if __name__ == '__main__':
    url1 = 'http://cpquery.sipo.gov.cn'
    name = '18811573238'
    pwd = 'Qingzhu981201!'
    sql = 'select name from qiye'
    # data = dbUtil.getdata3(sql)
    login(url1, name, pwd)
    # if k==0:
    #     getmesg2()
    #     k=1
    # if data != '':
    #     for d in data:
    #         for d in data:
    #             mess = getmesg(str(d[0], 'utf-8'))
    #             if len(mess)==0:
    #                 continue
    #             print(mess)
    #             sql = "INSERT INTO patent(a,b,c,d,e) VALUES " \
    #                   "(%s,%s,%s,%s,%s);"
    #             dbUtil.save3(sql, [mess])
    #             print(mess)


