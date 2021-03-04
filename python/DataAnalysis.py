import cx_Oracle
import configparser
import sweetviz as sv
import pandas as pd
import os
import sys
import datetime as dt
import impala.dbapi as idb
import numpy as np
from sklearn.tree import DecisionTreeClassifier as dtc


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
def FetchData(fetch_sql:str):
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
    sql=fetch_sql
    try:
        cur.execute(sql)
        data=pd.DataFrame(columns=[des[0] for des in cur.description],data=cur.fetchall())
        print('读取数据成功！')
    except Exception as e:
        print(e)
        return data
    return data

# 创建表格
def CreateTable(create_sql:str):
    conf = configparser.ConfigParser()
    conf.read('C:/jc/dbconfig_jc.conf')
    db_nickname = 'ORACLE'
    uid = conf.get(db_nickname, 'uid')
    pwd = conf.get(db_nickname, 'pwd')
    host = conf.get(db_nickname, 'ip')
    port = conf.get(db_nickname, 'port')
    service = conf.get(db_nickname, 'service')
    databasename = conf.get(db_nickname, 'databasename')
    tag = 0
    try:
        dsn = cx_Oracle.makedsn(host, port, service)
        conn = cx_Oracle.connect(uid, pwd, dsn)
        print('数据库连接成功！')
    except Exception as e:
        print(e)
        return tag
    cur = conn.cursor()
    sql = create_sql
    try:
        cur.execute(sql)
        print('建表成功！')
    except Exception as e:
        print(e)
        return tag
    tag = 1
    return tag

# 插入数据
def InsertData(insert_sql:str,data:pd.DataFrame):
    conf = configparser.ConfigParser()
    conf.read('C:/jc/dbconfig_jc.conf')
    db_nickname='ORACLE'
    uid = conf.get(db_nickname, 'uid')
    pwd = conf.get(db_nickname, 'pwd')
    host = conf.get(db_nickname, 'ip')
    port = conf.get(db_nickname, 'port')
    service = conf.get(db_nickname, 'service')
    databasename=conf.get(db_nickname, 'databasename')
    tag=0
    try:
        dsn=cx_Oracle.makedsn(host,port,service)
        conn=cx_Oracle.connect(uid,pwd,dsn)
        print('数据库连接成功！')
    except Exception as e:
        print(e)
        return tag
    cur=conn.cursor()
    sql=insert_sql
    param = [tuple(data.iloc[i]) for i in range(data.shape[0])]
    try:
        cur.executemany(sql,param)
        print('写入数据到表成功！')
        conn.commit()
    except Exception as e:
        print(e)
        return tag
    tag=1
    return tag

#%%
# 数据探索和分析
def Distribution(username:str,tablename:str,zb_ename:list,zb_cname:list)->pd.DataFrame:
    conf = configparser.ConfigParser()
    conf.read('C:/jc/dbconfig_jc.conf')
    db_nickname = 'ORACLE'
    uid = conf.get(db_nickname, 'uid')
    pwd = conf.get(db_nickname, 'pwd')
    host = conf.get(db_nickname, 'ip')
    port = conf.get(db_nickname, 'port')
    service = conf.get(db_nickname, 'service')
    databasename = conf.get(db_nickname, 'databasename')
    # zbb = openpyxl.load_workbook(r'.\dic\EFS.xlsx')
    # sheetnames=zbb.get_sheet_names()
    # ws=zbb.get_sheet_by_name(sheetnames[1])
    writer = pd.ExcelWriter(r'''.\data\%s.xlsx'''%tablename,engine='openpyxl')
    tag=0
    try:
        dsn=cx_Oracle.makedsn(host,port,service)
        conn=cx_Oracle.connect(uid,pwd,dsn)
        print('数据库连接成功！')
    except Exception as e:
        print(e)
        return tag
    cur=conn.cursor()
    for i in range(len(zb_cname)):
        ename=zb_ename[i]
        cname=zb_cname[i]
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
    print('写入成功！表位置在%s\data\%s.xlsx'%(os.path.dirname(sys.path[0]),tablename))
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

# 决策树划分边界
def DecisionTreeBoundary(x:pd.Series,y:pd.Series):
    boundary=[]
    x=x.fillna(-999).values
    y=y.values
    clf=dtc(criterion='gini',max_leaf_nodes=6,min_samples_leaf=0.05)
    clf.fit(x.reshape(-1,1),y)
    n_node=clf.tree_.node_count
    children_left=clf.tree_.children_left
    children_right=clf.tree_.children_right
    threshold=clf.tree_.threshold
    for i in range(n_node):
        if children_left[i]!=children_right[i]:
            boundary.append(threshold[i])
    boundary.sort()
    min_x=x.min()
    max_x=x.max()+0.1
    boundary=[min_x]+boundary+[max_x]
    return boundary

# 分箱函数并计算IV
def FeatureWoeIv(data:pd.DataFrame,col_list_str:list,col_list_num:list)->pd.DataFrame:
    # 对字符型变量（类别变量）先计算其bad_rate/good_rate将类别转化为数值，为决策树分箱做准备
    total_sum = data.shape[0]
    bad_sum = sum(data['Y'])
    good_sum = total_sum - bad_sum
    tag=0
    for col in col_list_str:   # 最后一列是y值
        data[col+'_WOE']=''
        df=data[[col,'Y']]
        # 计算woe
        boundary=data[col].unique()
        for category in boundary:
            if pd.isnull(category):
                df_category=df[df.T.isnull().any()]
            else:
                df_category=df[df[col]==category]
            total_num=df_category.shape[0]
            bad_num=sum(df_category['Y'])
            good_num=total_num-bad_num
            bad_rate=(bad_num+0.5)/(bad_sum+0.5)
            good_rate=(good_num+0.5)/(good_sum+0.5)
            woe=np.log(bad_rate/good_rate)
            data.loc[df_category.index,col+'_WOE']=woe
        tag+=1
        print('完成第%d个字符指标%s的初次woe值计算！  %s'%(tag,col,dt.datetime.now()))
    # 进行决策树分箱并计算每一箱的IV
    woe_iv=pd.DataFrame(columns=['var','bin','bad_num','good_num','total_num','bad_rate','good_rate','woe','iv'])
    y=data['Y']
    tag1=0
    for col in col_list_num:
        x=data[col]
        tag1+=1
        boundary=DecisionTreeBoundary(x,y)
        df=pd.concat([x,y],axis=1)
        df.columns = ['X', 'Y']  # 特征变量、目标变量字段的重命名
        df['bins'] = pd.cut(x=x.fillna(-999),bins=boundary, right=False)  # 获得每个x值所在的分箱区间
        woe_info=pd.DataFrame(columns=['var','bin','bad_num','good_num','total_num','bad_rate','good_rate','woe','iv'],data=None)
        tag2=0
        IV=0
        for bin in df['bins'].unique().categories:
            df_bin=df[df['bins']==bin]
            tag2+=1
            total_num = df_bin.shape[0]
            bad_num = sum(df_bin['Y'])
            good_num = total_num - bad_num
            bad_rate = (bad_num + 0.5) / (bad_sum + 0.5)
            good_rate = (good_num + 0.5) / (good_sum + 0.5)
            woe = np.log(bad_rate / good_rate)
            data.loc[df_bin.index, col + '_WOE'] = woe
            iv=(bad_rate-good_rate)*woe
            data.loc[df_bin.index,col+'_IV']=iv
            woe_info.loc[tag2]=[col,bin,bad_num,good_num,total_num,bad_rate,good_rate,woe,iv]
            IV+=iv
        woe_info['IV']=IV
        woe_iv=pd.concat([woe_iv, woe_info], ignore_index=True)
        print('第%d个数值型指标%s的woe分箱已完成, IV=%.3f   %s'%(tag1,col,IV,dt.datetime.now()))
        print(woe_info)
    tag1=0
    for col in col_list_str:
        x = data[col+'_WOE']
        tag1 += 1
        boundary = DecisionTreeBoundary(x, y)
        df = pd.concat([x, y], axis=1)
        df.columns = ['X', 'Y']  # 特征变量、目标变量字段的重命名
        df['bins'] = pd.cut(x=x.fillna(-999), bins=boundary, right=False)  # 获得每个x值所在的分箱区间
        woe_info = pd.DataFrame(columns=['var','bin', 'bad_num', 'good_num', 'total_num', 'bad_rate', 'good_rate', 'woe', 'iv'], data=None)
        tag2 = 0
        IV = 0
        for bin in df['bins'].unique().categories:
            df_bin = df[df['bins'] == bin]
            tag2 += 1
            total_num = df_bin.shape[0]
            bad_num = sum(df_bin['Y'])
            good_num = total_num - bad_num
            bad_rate = (bad_num + 0.5) / (bad_sum + 0.5)
            good_rate = (good_num + 0.5) / (good_sum + 0.5)
            woe = np.log(bad_rate / good_rate)
            data.loc[df_bin.index, col + '_WOE'] = woe
            iv = (bad_rate - good_rate) * woe
            data.loc[df_bin.index, col + '_IV'] = iv
            woe_info.loc[tag2] = [col,bin, bad_num, good_num, total_num, bad_rate, good_rate, woe, iv]
            IV += iv
        woe_info['IV'] = IV
        woe_iv=pd.concat([woe_iv, woe_info], ignore_index=True)
        print('第%d个字符型指标%s的woe分箱已完成, IV=%.3f   %s' % (tag1, col, IV, dt.datetime.now()))
        print(woe_info)
    return data,woe_iv
        # grouped = df.groupby('bins')['y']  # 统计各分箱区间的好、坏、总客户数量
        # result_df = grouped.agg([('good', lambda y: (y == 0).sum()), ('bad', lambda y: (y == 1).sum()), ('total', 'count')])
        # print(result_df)
        # result_df['rank'] = range(len(grouped))
        # result_df['good_rate'] = result_df['good'] / good_sum # 好客户占比
        # result_df['bad_rate'] = result_df['bad'] / bad_sum  # 坏客户占比
        # result_df['total_rate'] = result_df['total'] / total_sum  # 总客户占比
        # result_df['woe'] = np.log(result_df['bad_rate'] / result_df['good_rate'])  # WOE
        # result_df['badcumsum'] = result_df['bad'].cumsum()
        # result_df['goodcumsum'] = result_df['good'].cumsum()
        # result_df['ks'] = max(result_df['goodcumsum'] / good_sum - result_df['badcumsum'] / bad_sum)
        # result_df['iv'] = (result_df['bad_rate'] - result_df['good_rate']) * result_df['woe']  # IV
        # result_df['IV'] = result_df['iv'].sum()
        # result_df['var_name'] = x[col].name
        # result_df.reset_index(inplace=True)
        # print(f"该变量IV = {result_df['iv'].sum()}")
        # all_iv_detail = pd.concat([all_iv_detail, result_df], axis=0)
        # print(all_iv_detail)

#%%
#
if __name__=='__main__':
    os.chdir(os.path.dirname(sys.path[0]))
    username='JC'
    tablename='CAR_PICK_2016'
    zbb=pd.read_excel(r'.\data\dic.xlsx',sheet_name=0)
    zb_ename=zbb['字段英文名'].values
    zb_cname=zbb['字段注释'].values
    Distribution(username,tablename,zb_ename,zb_cname)
    username = 'jc'
    tablename = 'creditunionreportmodel'
    data = pd.read_csv(r'C:\jc\AntiFraud\data\creditunionreportmodel.csv')
    create_sql = '''create table %s.%s(
        apply_id varchar (50),
        reportdate varchar (50),
        residenceloannumber number,
        residenceloanforbiznumber number,
        otherloannumber number,
        holdcreditcardnumbers number,
        holdquasicreditcardnumbers number,
        loanoverduenumber number,
        loanoverduedatenumber number,
        loanoverdueamount number,
        loanoverduemonths number,
        creditcardoverduenumber number,
        creditcardoverduedatenumber number,
        creditcardoverdueamount number ,
        creditcardoverduemonths number,
        quasicreditcardoverduenumber number,
        quasicreditcardoverduedatenumber number,
        quasicreditcardoverduemonths number,
        lendercorporatenumbers number,
        lenderorgnumbers number,
        loannumbers number,
        loantotalamount number,
        loantotalbalance number,
        avgrepaymentoflastsixmonth number,
        issueQuasiCreditcardCorporateNumbers number,
        issueQuasiCreditcardOrgNumbers number,
        creditcardnumbers number,
        creditcardtotalamount number,
        singlecreditcardmaxamount number,
        singlecreditcardminamount number,
        usedcreditcardbalance number ,
        avgcreditcardrepaymentoflastsixmonth number,
        guaranteenumbers number,
        guaranteeamount number,
        guaranteeamountbalance number,
        assessscore number,
        baddebtscnt number,
        baddebtsamt number,
        assetsdisposalcnt number,
        assetsdisposalamt number,
        guarantorpaycnt number,
        guarantorpayamt number,
        externalguaranteecnt number,
        externalguaranteeamt number,
        loanAuditOrgCntOfLastMonth number,
        cardAuditOrgCntOfLastMonth number,
        loanAuditCntOfLastMonth number,
        cardAuditCntOfLastMonth number,
        personalQueryCnt number,
        loanManagerCntOfLast2Years number,
        guaranteeAuditCntOfLast2Years number,
        merchantAduitCntOfLast2Years number
        )''' % (username, tablename)
    insert_sql = '''
        insert into %s.%s　
        (apply_id,reportdate,residenceloannumber,
        residenceloanforbiznumber,otherloannumber,holdcreditcardnumbers,holdquasicreditcardnumbers,loanoverduenumber,
        loanoverduedatenumber,loanoverdueamount,loanoverduemonths,creditcardoverduenumber,creditcardoverduedatenumber,
        creditcardoverdueamount,creditcardoverduemonths,quasicreditcardoverduenumber,quasicreditcardoverduedatenumber,
        quasicreditcardoverduemonths,lendercorporatenumbers,lenderorgnumbers,loannumbers,loantotalamount,loantotalbalance,
        avgrepaymentoflastsixmonth,issueQuasiCreditcardCorporateNumbers,issueQuasiCreditcardOrgNumbers,creditcardnumbers,creditcardtotalamount,singlecreditcardmaxamount,
        singlecreditcardminamount,usedcreditcardbalance,avgcreditcardrepaymentoflastsixmonth,guaranteenumbers,
        guaranteeamount,guaranteeamountbalance,assessscore,baddebtscnt,baddebtsamt,assetsdisposalcnt,
        assetsdisposalamt,guarantorpaycnt,guarantorpayamt,externalguaranteecnt,externalguaranteeamt,
        loanAuditOrgCntOfLastMonth,cardAuditOrgCntOfLastMonth,loanAuditCntOfLastMonth,cardAuditCntOfLastMonth,
        personalQueryCnt,loanManagerCntOfLast2Years,guaranteeAuditCntOfLast2Years,merchantAduitCntOfLast2Years )
        values (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12,:13,:14,:15,:16,:17,:18,:19,:20,:21,:22,:23,:24,:25,:26,:27,:28,:29,:30,
        :31,:32,:33,:34,:35,:36,:37,:38,:39,:40,:41,:42,:43,:44,:45,:46,:47,:48,:49,:50,:51)
        ''' % (username, tablename)
    # CsvToOracle(username,tablename,data,create_sql,insert_sql)
    # 导入loantrandetailinfo
    username = 'JC'
    tablename = 'loantrandetailinfo'
    data = pd.read_csv(r'C:\jc\AntiFraud\data\loantrandetailinfo.csv')
    create_sql = '''create table %s.%s(
            apply_id varchar (50),
            reportdate varchar (50),
            loanSerialNo varchar (50),
            loanAmount number,
            loanType varchar (50),
            guaranteeMode varchar (50),
            loanPhase varchar (50),
            paymentType varchar (50),
            lastPaymentDate varchar (50),
            acctStatus varchar (50),
            classLevel varchar (50),
            loanAmt number,
            remainPhase number,
            paymentAmount number,
            paymentDate varchar (50),
            currentPaymentAmount number,
            overduePhase number,
            overdueAmount number,
            overduePhase1 number,
            overduePhase2 number,
            overduePhase3 number,
            overduePhase4 number,
            last2yearsRepaymentRecord varchar (50)
        )''' % (username, tablename)
    insert_sql = '''insert into %s.%s　
            (apply_id,reportdate,loanSerialNo,loanAmount,loanType,guaranteeMode,loanPhase,
            paymentType,lastPaymentDate,acctStatus,classLevel,loanAmt,remainPhase,paymentAmount,paymentDate,currentPaymentAmount,
            overduePhase,overdueAmount,overduePhase1,overduePhase2,overduePhase3,overduePhase4,last2yearsRepaymentRecord)
            values (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12,:13,:14,:15,:16,:17,:18,:19,:20,:21,:22)
            ''' % (username, tablename)
    # CsvToOracle(username, tablename, data, create_sql, insert_sql)
    # 导入creditcardtrandetailinfo
    username = 'JC'
    tablename = 'creditcardtrandetailinfo'
    data = pd.read_csv(r'C:\jc\AntiFraud\data\creditcardtrandetailinfo.csv')
    create_sql = '''create table %s.%s(
            apply_id varchar (50),
            reportdate varchar (50),
            creditSerialNo number ,
            currency varchar (50),
            guaranteeMode varchar (50),
            acctStatus varchar (50),
            creditLimit number,
            usedCreditLimit number,
            avgLimit number,
            maxUseLimit number,
            totalStatementBalance number,
            paymentAmount number,
            overduePhaseCnt number,
            overduePhaseAmount number,
            last2yearsRepaymentRecord varchar (50)
        )''' % (username, tablename)
    insert_sql = '''insert into %s.%s　
            (apply_id,reportdate,creditSerialNo,currency,guaranteeMode,acctStatus,creditLimit,usedCreditLimit,
            avgLimit,maxUseLimit,totalStatementBalance,paymentAmount,overduePhaseCnt,overduePhaseAmount,last2yearsRepaymentRecord)
            values (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12,:13,:14,:15)
        ''' % (username, tablename)
    CsvToOracle(username, tablename, data, create_sql, insert_sql)

    zbb = pd.read_excel(r'.\data\dic.xlsx', sheet_name=0)
    zb_ename = zbb['字段英文名'].values
    zb_cname = zbb['字段注释'].values
    Distribution(username, tablename, zb_ename, zb_cname)