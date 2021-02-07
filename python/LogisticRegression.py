import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as dtc
import datetime as dt
import math
import os
import sys

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

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

if __name__=='__main__':
    os.chdir(os.path.dirname(sys.path[0]))
    per_info = pd.read_csv(r'data\basicinfo.csv')
    col_list_num=['AGE','INCOME_YEAR','JOB_MONTH']
    col_list_str=list(per_info.columns[1:-1])   # 不包含第0列，因为不是指标,也不包含最后一列,因为是y
    for col in col_list_num:
        col_list_str.remove(col)
    data,woe_iv=FeatureWoeIv(per_info,col_list_str,col_list_num)
    # 进行onehot


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
sc=StandardScaler()
mms=MinMaxScaler()
sc.fit(x_train)
mms.fit(x_train)
x_train_std=sc.transform(x_train)
x_test_std=sc.transform(x_test)
x_train_mms=mms.transform(x_train)
x_test_mms=mms.transform(x_test)
lr=LogisticRegression(max_iter=10000,random_state=1)
lr.fit(x_train_std, y_train)
r = lr.score(x_train, y_train)
y_predict = lr.predict(x_test)
y_prob = lr.predict_proba(x_test)[:,1]
y_predict_ = lr.predict(x_train)
y_prob_ = lr.predict_proba(x_train)[:,1]


print('-----------------测试-----------------')
train_right_counts = 0
for i in range(x_train_std.shape[0]):
    original_val = y_train[i]
    predict_val = lr.predict(x_train_std[i,:].reshape(1, -1))
    if original_val == predict_val:
        train_right_counts += 1
print("训练集准确率：", ((train_right_counts * 1.0) / x_train_std.shape[0]))
test_right_counts = 0
for i in range(x_test_std.shape[0]):
    original_val = y_test[i]
    predict_val = lr.predict(x_test_std[i, :].reshape(1, -1))
    if original_val == predict_val:
        test_right_counts += 1
print("测试集准确率：", ((test_right_counts * 1.0) / x_test_std.shape[0]))

print(metrics.accuracy_score(y_train, y_predict_, normalize=True, sample_weight=None))
cfm = metrics.confusion_matrix(y_train, y_predict_, labels=None, sample_weight=None)
print(cfm)
print(metrics.classification_report(y_train, y_predict_, labels=None, sample_weight=None))

print(metrics.accuracy_score(y_test, y_predict, normalize=True, sample_weight=None))
cfm = metrics.confusion_matrix(y_test, y_predict, labels=None, sample_weight=None)
print(cfm)
print(metrics.classification_report(y_test, y_predict, labels=None, sample_weight=None))
row_sums = np.sum(cfm, axis=1)  # 求出混淆矩阵每一行的和
error_matrix = cfm / row_sums  # 求出每一行中每一个元素所占这一行的百分比
# 将矩阵中对角线上的元素都定位0
# np.fill_diagonal(error_matrix, 0)
plt.matshow(error_matrix, cmap=plt.cm.gray)
plt.show()
print("RMSE = ",math.sqrt(sum((y_test - y_prob) ** 2)/len(y_prob)))

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob, pos_label=1)
fpr_, tpr_, thresholds_ = metrics.roc_curve(y_train, y_prob_, pos_label=1)

for i in range(tpr.shape[0]):
    if tpr[i] > 0.5:
        print(tpr[i], 1 - fpr[i], thresholds[i])
    break

# train
roc_auc_ = metrics.auc(fpr_, tpr_)
plt.figure(figsize=(10, 10))
plt.plot(fpr_, tpr_, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc_)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example(train)')
plt.legend(loc="lower right")
plt.savefig("ROC（train）.png")
plt.show()
plt.plot(tpr_, lw=2, label='tpr')
plt.plot(fpr_, lw=2, label='fpr')
plt.plot(tpr_ - fpr_, label='ks')
plt.title('KS = %0.2f(train)' % max(tpr_ - fpr_))
plt.legend(loc="lower right")
plt.savefig("KS(train).png")
plt.show()

# test
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
     lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example(test)')
plt.legend(loc="lower right")
plt.savefig("ROC(test).png")
plt.show()
plt.plot(tpr,lw=2, label='tpr')
plt.plot(fpr,lw=2, label='fpr')
plt.plot(tpr - fpr, label = 'ks')
plt.title('KS = %0.2f(test)' % max(tpr-fpr))
plt.legend(loc="lower right")
plt.savefig("KS(test).png")
plt.show()

# 绘制测试集结果验证
# test_validate(x_test=x_test, y_test=y_test, y_predict=y_predict, classifier=lr)
