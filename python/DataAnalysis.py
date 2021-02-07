import cx_Oracle
import configparser
import sweetviz as sv
import pandas as pd
import os
import sys
import datetime as dt
import impala.dbapi as idb


# 数据库连接函数
def Cur():
    conf = configparser.ConfigParser()
    conf.read('C:/jc/dbconfig_jc.conf')
    db_nickname = 'ORACLE'
    uid = conf.get(db_nickname, 'uid')
    pwd = conf.get(db_nickname, 'pwd')
    host = conf.get(db_nickname, 'ip')
    port = conf.get(db_nickname, 'port')
    service = conf.get(db_nickname, 'service')
    databasename = conf.get(db_nickname, 'databasename')
    cur = 0
    try:
        dsn = cx_Oracle.makedsn(host, port, service)
        conn = cx_Oracle.connect(uid, pwd, dsn)
        print('数据库连接成功！')
    except Exception as e:
        print(e)
        return cur
    cur = conn.cursor()
    return cur

# 读取数据
def FetchData(username,tablename):
    conf = configparser.ConfigParser()
    conf.read('C:/jc/dbconfig_jc.conf')
    db_nickname='ORACLE'
    uid = conf.get(db_nickname, 'uid')
    pwd = conf.get(db_nickname, 'pwd')
    host = conf.get(db_nickname, 'ip')
    port = conf.get(db_nickname, 'port')
    service = conf.get(db_nickname, 'service')
    databasename=conf.get(db_nickname, 'databasename')
    data=0
    try:
        dsn=cx_Oracle.makedsn(host,port,service)
        conn=cx_Oracle.connect(uid,pwd,dsn)
        print('数据库连接成功！')
    except Exception as e:
        print(e)
        return data
    cur=conn.cursor()
    sql='''select * from %s.%s where y=1 or rownum<=100000'''%(username,tablename)
    try:
        cur.execute(sql)
        data=pd.DataFrame(columns=[des[0] for des in cur.description],data=cur.fetchall())
        print('读取数据%s成功！'%tablename)
    except Exception as e:
        print(e)
        return data
    return data

#%%
# 数据探索和分析
def Distribution(username,tablename):
    conf = configparser.ConfigParser()
    conf.read('C:/jc/dbconfig_jc.conf')
    db_nickname = 'ORACLE'
    uid = conf.get(db_nickname, 'uid')
    pwd = conf.get(db_nickname, 'pwd')
    host = conf.get(db_nickname, 'ip')
    port = conf.get(db_nickname, 'port')
    service = conf.get(db_nickname, 'service')
    databasename = conf.get(db_nickname, 'databasename')
    zbb=pd.read_excel(r'.\dic\EFS.xlsx',sheet_name=0)
    # zbb = openpyxl.load_workbook(r'.\dic\EFS.xlsx')
    # sheetnames=zbb.get_sheet_names()
    # ws=zbb.get_sheet_by_name(sheetnames[1])
    writer = pd.ExcelWriter(r'''.\dic\%s.xlsx'''%bmc,engine='openpyxl')
    tag=0
    try:
        dsn=cx_Oracle.makedsn(host,port,service)
        conn=cx_Oracle.connect(uid,pwd,dsn)
        print('数据库连接成功！')
    except Exception as e:
        print(e)
        return tag
    cur=conn.cursor()
    zbb=zbb.loc[zbb['表英文名']==tablename]
    zbb=zbb.reset_index(drop=True)
    for i in range(zbb.shape[0]):
        ename=zbb['字段英文名'][i].upper()
        cname=zbb['字段注释'][i]
        sql = '''select %s,count(*) from %s.%s group by %s order by %s'''%(ename,username,tablename,ename,ename)
        try:
            cur.execute(sql)
            distribution = pd.DataFrame(columns=[des[0] for des in cur.description], data=cur.fetchall())
            print('读取数据%d  %s成功！  %s'%(i,ename,dt.datetime.now()))
        except Exception as e:
            print(e)
            return tag
        distribution.to_excel(excel_writer=writer,sheet_name=cname,index=False)
    writer.save()
    print('写入成功！')
    tag=1
    return tag

def FetchImpala():
    host='10.116.11.203'
    port=21050
    databasename='dagroup'
    auth_mechanism='LDAP'
    user='dtusr'
    password='dt123'
    data=0
    try:
        conn=idb.connect(host=host,port=port,database=databasename,auth_mechanism=auth_mechanism,user=user,password=password)
        print('连接数据库成功！')
    except Exception as e:
        print(e)
        return data
    cur=conn.cursor()
    tablename='CreditCardTranDetailInfo'
    sql='''select * from %s'''%tablename
    try:
        cur.execute(sql)
        data=pd.DataFrame(columns=[des[0] for des in cur.description],data=cur.fetchall())
        print('读取数据成功！')
        return data
    except Exception as e:
        print(e)
        return data


#%%
#

if __name__=='__main__':
    os.chdir(os.path.dirname(sys.path[0]))
    # FetchImpala()
    l=['RT_APPLYER']
    for bmc in l:
        Distribution('EFS',bmc)
