# 导入pymysql模块
import pymysql
import time
# 连接database

def save(sql,data):
    conn = pymysql.connect(host='localhost', user='root', password='981201', database='patent', charset='utf8')

    # 得到一个可以执行SQL语句的光标对象
    cursor = conn.cursor()

    for i in data:

        cursor.executemany(sql, i)
    # 提交事务
    conn.commit()
    print('数据存入成功')
    cursor.close()
    conn.close()

def getdata(sql,list):
    conn = pymysql.connect(host='localhost', user='root', password='981201', database='patent', charset='utf8')

    try:
        cur = conn.cursor()
        # 查询数据
        cur.execute(sql,list)
        # 获取数据
        data = cur.fetchall();
    except Exception as e:
        print(e)
        cur = conn.cursor()
        # 查询数据
        cur.execute(sql, list)
        # 获取数据
        data = cur.fetchall();
    return data


def getdata2(sql):
    conn = pymysql.connect(host='localhost', user='root', password='981201', database='patent', charset='utf8')

    try:

        cur = conn.cursor()
        # 查询数据
        cur.execute(sql)
        # 获取数据
        data = cur.fetchall();
    except Exception as e:
        cur = conn.cursor()
        # 查询数据
        cur.execute(sql)

    return data
def getdata3(sql):
    conn = pymysql.connect(host='localhost', user='root', password='981201', database='patent', charset='utf8')

    try:

        cur = conn.cursor()
        # 查询数据
        cur.execute(sql)
        # 获取数据
        data = cur.fetchall();
    except Exception as e:
        cur = conn.cursor()
        # 查询数据
        cur.execute(sql)

    return data

def save3(sql,data):
    conn = pymysql.connect(host='localhost', user='root', password='981201', database='patent', charset='utf8')

    # 得到一个可以执行SQL语句的光标对象
    cursor = conn.cursor()

    for i in data:

        cursor.executemany(sql, i)
    # 提交事务
    conn.commit()
    print('数据存入成功')
    cursor.close()
    conn.close()

def updatedata(sql):
    conn = pymysql.connect(host='localhost', user='root', password='981201', database='patent', charset='utf8')

    try:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        print('数据更新成功')
    except Exception as e:
        print(e)
        conn.rollback()

    conn.close()

if __name__ == '__main__':
    sql = "INSERT INTO patent(file1,file2) VALUES ( %s,%s);"
    data = [("1","2")]
    save(sql, data)