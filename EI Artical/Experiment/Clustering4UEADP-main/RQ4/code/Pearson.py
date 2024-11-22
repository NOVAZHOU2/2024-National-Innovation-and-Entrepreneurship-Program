import copy
import math
import os
import pandas as pd
import numpy as np

# RQ4：无监督聚类技术的努力感知性能与分类性能有关吗？
"""
这个没有什么问题，就是计算相关系数，然后画图
需要改动一点，在main函数的注释都讲了
"""
def calcMean(x,y):
    # 就计算平均值
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean

def sckendall(a, b):
    L = len(a)
    count = 0
    for i in range(L - 1):
        for j in range(i + 1, L):
            count = count + np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    kendall_tau = count / (L * (L - 1) / 2)

    return kendall_tau

def calcPearson(x,y):
    # 计算Pearson相关系数的
    x_mean,y_mean = calcMean(x,y) # 计算两个的期望
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p

def read_data(dataset_path, metric, function):

    data_path = '{0}/{1}.xlsx'.format(dataset_path, metric)
    raw_datas = pd.read_excel(data_path)
    raw_datas = raw_datas[function].values

    return raw_datas

if __name__ == '__main__':
    metrics = ['Precision', 'Recall', 'F1', 'Precisionx', 'Recallx', 'F1x', 'PofB', 'PMI', 'IFA']
    # dataset_path = '../../Result'
    dataset_path = '../../Result_rq1'
    # 这里要把path改掉，或者在外面搞Result文件夹，把Result_rq1的文件复制一份进去
    # 卡这了，跑不了一点
    # 因为我跑出来，Kmedoids最优的时候不是按照LOC排序的，所以没有Kmedoids.LOC
    # functions = ['Kmedoids.LOC', 'KmeansPlus.LOC', 'Cure.LOC']
    functions = ['Kmedoids.AMC', 'KmeansPlus.Avg_CC', 'Cure.LOC', 'Ttsas.AMC']
    # 我对着表改成这样试试
    # 改完确实能跑一个像样的结果了
    # 说个大逆不道的话，我Kmedoids跑出来的结果挺好的，甚至跟学长论文里的图长得挺像的
    # 但是令两个方法的效果看起来都好差啊哈哈哈哈
    # 我自己瞎试试，发现Ttsas跑出来效果也不错哈哈哈
    for func in functions:
        result_list = []
        for metric in metrics:
            # 读了/Result/Precision……/Kmedoids.AMC……数据
            datas = read_data(dataset_path, metric, func)
            result_list.append(list(datas))

        pearson_res = []
        print(result_list)
        # result_list其实就是一个二维数组，每一行就是当前方法在不同性能的数据
        for i in result_list:
            tmp = []
            for j in result_list:
                tmp.append(calcPearson(i, j))
                # 就是拿不同性能指标的数据区计算Pearson系数
                # tmp.append(sckendall(x, y))
            pearson_res.append(tmp)

            # p = calcPearson(datas, datas)
            # print(p)
        print(pearson_res)

        new_metrics = ['Precision', 'Recall', 'F1', 'Precision@20%', 'Recall@20%', 'F1@20%', 'PofB@20%', 'PMI@20%',
                       'IFA']
        df = pd.DataFrame(data=pearson_res, columns=new_metrics)

        if not os.path.exists("../output/"):
            os.makedirs("../output")
        df.to_csv("../output/{0}.csv".format(func), index=False)


