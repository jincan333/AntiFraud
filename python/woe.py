# -*- coding: utf-8 -*-
import pandas as pd

import numpy as np


from sklearn.tree import DecisionTreeClassifier


def _optimal_binning_boundary(x: pd.Series, y: pd.Series, nan: float = -999.) -> list:
    '''

        利用决策树获得最优分箱的边界值列表

    '''

    boundary = []  # 待return的分箱边界值列表

    x = x.fillna(nan).values  # 填充缺失值

    y = y.values

    clf = DecisionTreeClassifier(criterion='gini',  # “基尼系数”最小化准则划分

                                 max_leaf_nodes=6,  # 最大叶子节点数

                                 min_samples_leaf=0.05)  # 叶子节点样本数量最小占比

    clf.fit(x.reshape(-1, 1), y)  # 训练决策树

    n_nodes = clf.tree_.node_count

    children_left = clf.tree_.children_left

    children_right = clf.tree_.children_right

    threshold = clf.tree_.threshold

    for i in range(n_nodes):

        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值

            boundary.append(threshold[i])

    boundary.sort()

    min_x = x.min()

    max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本

    boundary = [min_x] + boundary + [max_x]

    return boundary


def feature_woe_iv(x: pd.DataFrame, col_list: list, y: pd.Series, nan: float = -999.) -> pd.DataFrame:
    '''

        计算变量各个分箱的WOE、IV值，返回一个DataFrame

    '''

    if x.isnull:
        x.fillna(nan)

    all_iv_detail = pd.DataFrame([])

    for col in col_list:
        print(x[col])

        boundary = _optimal_binning_boundary(x[col], y, nan)  # 获得最优分箱边界值列表

        df = pd.concat([x[col], y], axis=1)

        df.columns = ['x', 'y']  # 特征变量、目标变量字段的重命名

        df['bins'] = pd.cut(x=x[col], bins=boundary, right=False)  # 获得每个x值所在的分箱区间

        grouped = df.groupby('bins')['y']  # 统计各分箱区间的好、坏、总客户数量

        result_df = grouped.agg([('good', lambda y: (y == 0).sum()),

                                 ('bad', lambda y: (y == 1).sum()),

                                 ('total', 'count')])

        print(result_df)

        result_df['rank'] = range(len(grouped))

        result_df['good_pct'] = result_df['good'] / result_df['good'].sum()  # 好客户占比

        result_df['bad_pct'] = result_df['bad'] / result_df['bad'].sum()  # 坏客户占比

        result_df['total_pct'] = result_df['total'] / result_df['total'].sum()  # 总客户占比

        result_df['badrate'] = result_df['bad'] / result_df['total']  # 坏比率

        result_df['woe'] = np.log(result_df['good_pct'] / result_df['bad_pct'])  # WOE

        result_df['badcumsum'] = result_df['bad'].cumsum()

        result_df['goodcumsum'] = result_df['good'].cumsum()

        result_df['ks'] = max(result_df['badcumsum'] / sum(result_df['badcumsum']) - result_df['goodcumsum'] / sum(
            result_df['goodcumsum']))

        result_df['iv'] = (result_df['good_pct'] - result_df['bad_pct']) * result_df['woe']  # IV

        result_df['IV'] = result_df['iv'].sum()

        result_df['var_name'] = x[col].name

        result_df.reset_index(inplace=True)

        print(f"该变量IV = {result_df['iv'].sum()}")

        all_iv_detail = pd.concat([all_iv_detail, result_df], axis=0)

        print(all_iv_detail)

    return all_iv_detail


data = pd.read_table('目标数据集.txt',sep="\t")
print(data)
print(data["Y"])
print(data.iloc[:,0:-1])
y = data["Y"]
x = data.iloc[:,0:-2]
x_list = x.columns.values.tolist()
woe = feature_woe_iv(x,x_list,y)
bon_list = woe["bins"]
woe_list = list(woe['woe'])

with open('目标数据集.txt','r',encoding='UTF-8') as f1:
    f1.readline()
    l = f1.readline().strip().split()
    temp = {'43':{},'44':{},'66':{},'71':{},'74':{},'76':{},'106':{},'112':{},'116':{},'119':{},'124':{},'127':{},'132':{},'140':{},'144':{},'154':{},'155':{},
            '157':{},'162':{},'166':{},'Y':{}}
    count = 0
    while l:
        for i,tag in enumerate(x_list):
            sasd = woe[woe.var_name == tag]
            for j in range(len(sasd)):
                ccc = sasd.loc[j, 'bins']
                if j == 0:
                    a = str(ccc).strip('[').strip(')').split(',')
                    ccc = pd.Interval(float(a[0])-0.1,float(a[1]),closed='left')
                if j == len(sasd):
                    a = str(ccc).strip('[').strip(')').split(',')
                    ccc = pd.Interval(float(a[0]),float(a[1])+0.1,closed='left')
                cc = float(l[i])
                if cc in ccc:
                    temp[tag][l[-2]] = j+1
        temp['Y'][l[-2]] = l[-1]
        l = f1.readline().strip().split()
data = pd.DataFrame(temp)
data.to_excel(r'2.xls',sheet_name='Sheet1')  #数据输出至Excel