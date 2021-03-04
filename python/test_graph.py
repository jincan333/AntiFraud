import math

from py2neo import Graph, Node, Relationship
import py2neo as neo
import pandas as pd
import time

graph = Graph('http://10.116.85.122:7474', auth=('neo4j','ShangQi01!'))
for i in range(1):
    tt = time.time()
    file = pd.read_csv('C:/Users/panlinxuan/Desktop/knowledge_graph/anti-graph_%d.csv' % i, encoding='UTF-8')
    file.columns = ['khbh','khxm','khxb','zjlx','zjhm', 'xl', 'sfyjsz','hyzk', 'shnx', 'csrq', 'zyhy', 'gzdw', 'gzzw',
                 'gzzc', 'dwsshy', 'dwsf','dwcs', 'dwxxdz','gzsj','dwdh', 'jzdfcxz','jzdsf','jzdcs', 'jzdxxdz',
                 'jzdrzrq','hjsf', 'hjcs','hjxxdz','sjhm','gddh']
    file.fillna('无',inplace=True)
    tt2 = time.time()
    result = pd.DataFrame()
    # result['Person'] = file.apply(lambda x: Node('人',客户编号=x['khbh'],客户姓名=x['khxm'],客户性别=x['khxb'],证件类型=x['zjlx'],
    #                                            证件号码=x['zjhm'],出生日期=x['csrq'],
    #
    #                                              学历=x['xl'],驾驶证情况=x['sfyjsz'],婚姻状况=x['hyzk'],
    #                                            税后年薪=x['shnx'],职业行业=x['zyhy']
    #                                            ,工作职务=x['gzzw'],工作职称=x['gzzc'],工作时间=x['gzsj'],
    #                                             居住房产性质=x['jzdfcxz'],入住时间=x['jzdrzrq'],
    #
    #                                              工作单位=x['gzdw'],手机号=x['sjhm'],固定电话=x['gddh'])
    #                               ,axis=1)
    result['gs'] = file.apply(lambda x: Node('公司', 公司名称=x['gzdw'],所属行业=x['dwsshy'],省份=x['dwsf'],
                                             城市=x['dwcs'],详细地址=x['dwxxdz'],单位电话=x['dwdh']), axis=1)
    # result['jzd'] = file.apply(lambda x: Node('地址', 地址编号=x['khbh'], 地址性质='居住地', 省份=x['jzdsf'],
    #                                             城市=x['jzdcs'], 详细地址=x['jzdxxdz']), axis=1)  # 标准化地址 todo
    # result['hjd'] = file.apply(lambda x: Node('地址', 地址编号=x['khbh'], 地址性质='户籍地', 省份=x['hjsf'],
    #                                             城市=x['hjcs'], 详细地址=x['hjxxdz']), axis=1)
    # result['gsdz'] = file.apply(lambda x: Node('地址', 地址编号=x['khbh'], 地址性质='公司地址', 省份=x['dwsf'],
    #                                           城市=x['dwcs'], 详细地址=x['dwxxdz']), axis=1)
    # result['phone1'] = file.apply(lambda x: Node('电话', 电话号码=x['sjhm'], 性质='手机'), axis=1)  # 标准 todo
    # result['phone2'] = file.apply(lambda x: Node('电话', 电话号码=x['gddh'], 性质='固话'), axis=1)  # 标准 todo 拆开
    # result['phone3'] = file.apply(lambda x: Node('电话', 电话号码=x['dwdh'], 性质='公司电话'), axis=1)
    result_list_1 = result['gs'].values.tolist()
    tt3 = time.time()
    for j in range(math.ceil(len(result_list_1)/20000)):
        if (j+1)*20000<len(result_list_1):
            nodes = result_list_1[j*20000:(j+1)*20000]
        else:
            nodes = result_list_1[j*20000:]
        temp_graph = neo.Subgraph(nodes=nodes)
        graph.create(temp_graph)

        if j % 1 == 0:
            print("完成%d条"%(200000*i+(j+1)*20000))
    # tt4 = time.time()
    # result['gzdw'] = result.apply(lambda x: Relationship(x['Person'],'工作单位',x['gs']),axis=1)
    # result_list_1 = result['gzdw'].values.tolist()
    # for j in range(math.ceil(len(result_list_1) / 20000)):
    #     if (j+1)*20000<len(result_list_1):
    #         nodes = result_list_1[j*20000:(j+1)*20000]
    #     else:
    #         nodes = result_list_1[j*20000:]
    #     temp_graph = neo.Subgraph(nodes=nodes)
    #     graph.create(temp_graph)
    #     if j % 1 == 0:
    #         print("完成%d条"%(200000*i+(j+1)*20000))

    print('总耗时',time.time()-tt,'秒\n读取耗时',tt2-tt,'秒\n计算耗时',tt3-tt2,'秒\n创建耗时',time.time()-tt3,'秒\n')
