# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
import xlrd
warnings.filterwarnings("ignore")


def read_data(dataset_path, metric):

    """
    :param dataset_path: 数据路径
    :param metric: 需要读取的指标
    :return: 把对应指标的文件中的数据读出并返回
    """

    # 读了这十个方法的的数据，要在RQ2里面创建一个Data，要把Result_rq2里的数据拷贝到Data里面好像
    # 但是Result_rq2里面的数据都只有8种方法，没有EALR和EATT，所以会报错


    # functions = ['Kmedoids', 'Kmeans++', 'Cure', 'ManualUp', 'CBS+', 'RF', 'GB', 'EALR', 'EATT', 'SERS']
    functions = ['HDBSCAN', 'ManualUp', 'CBS+', 'RF', 'GB', 'EALR', 'SERS'] # 我吧EATT删掉试一下
    datas = {}
    for function in functions:
        data_path = '{0}/{1}.xlsx'.format(dataset_path,metric)
        # print(data_path)
        raw_datas = pd.read_excel(data_path)
        # print(raw_datas)
        raw_datas = raw_datas[function].values

        datas[function] = raw_datas
    # print(datas)
    return datas

def process_data(datas,baseline):
    """
    :param datas: 所有方法某个指标的数据
    :param baseline: 指定的方法
    :return: 把指定的方法数据放在最前面，返回调整后的所有数据，以及对应的方法名排列
    """
    metric_datas = []
    functions = []

    baseline_data = datas[baseline]
    metric_datas.append(baseline_data)
    print(metric_datas)
    print("--------------------------------------------------")
    for key, value in datas.items():
        if key == baseline:
            continue
        metric_datas.append(value)
        functions.append(key)
    print(metric_datas)
    print("--------------------------------------------------")
    print(functions)
    return metric_datas, functions


def wilcoxon(l1, l2):
    w, p_value = stats.wilcoxon(l1, l2, correction=False)
    return p_value # 返回p值，如果p值小于0.05，会被认为是有显著差异的

def wdl(l1, l2):
    win = 0
    draw = 0
    loss = 0
    for i in range(len(l1)):
        if l1[i] < l2[i]:
            loss = loss+1
        if l1[i] == l2[i]:
            draw = draw+1
        if l1[i] > l2[i]:
            win = win+1

    return win, draw, loss

def average_improvement(l1, l2):
    # 计算平均值的差
    avgl1 = round(np.average(l1), 3)
    avgl2 = round(np.average(l2), 3)
    #imp = round((avgl1-avgl2)/avgl2, 4)
    imp = round((avgl1-avgl2), 3)
    return imp

def Wilcoxon_signed_rank_test(metric_datas, functions, metric, b):

    """
    这里有个问题是，这里的文件写完之后，和后面画图的路径是对不上的
    :param metric_datas: b方法的数据放在最前面后的数据
    :param functions: 对应的方法顺序
    :param metric: 当前指标
    :param b: 下标
    :return:
    """

    pvalues = []
    sortpvalues = []
    bhpvalues = []
    print('***********{0}***********'.format(metric))
    print(metric_datas)
    #improve_ave_dataset = []
    for i in range(1, len(metric_datas)):
        #这里需要改一下
        # 啊啊啊？学长这个改一下是什么意思
        pvalue = wilcoxon(metric_datas[0], metric_datas[i])
        # 就是把后面的每个方法和第一个方法计算p值
        pvalues.append(pvalue)
        sortpvalues.append(pvalue)
        #improve_ave_dataset.append(average_improvement(metric_datas[i], metric_datas[0]))
        if metric == 'Recall' or metric == 'PofB':
            print('-------------{0}-----------------'.format(functions[i - 1]))
            #print("compute p-value between %s and CBSplus: %s" % (functions[i-1], pvalue))
            #print("compute W/D/L between %s and CBSplus: %s" % (functions[i-1], wdl(metric_datas[i], metric_datas[0])))
            print("compute average improvement between {0} and CBSplus: {1}" .format(functions[i-1],
                                                                             average_improvement(metric_datas[i], metric_datas[0])))
    print("sortpvalues: {0}".format(sortpvalues))
    sortpvalues.sort()

    for i in range(len(pvalues)):
        bhpvalue = pvalues[i]*(len(pvalues))/(sortpvalues.index(pvalues[i])+1)
        # 这好像是一个多重比较校正，用p值 * 总数 / p值排名
        bhpvalues.append(bhpvalue)
        print("compute Benjamini—Hochberg p-value between %s and CBSplus: %s" % (functions[i-1], bhpvalue))

    Path('../output/p_{0}/'.format(metric)).mkdir(parents=True, exist_ok=True)
    output_path = '../output/p_{0}/{1}.csv'.format(metric,baseline[b])

    output = pd.DataFrame(data=[pvalues], columns=functions)
    #output = pd.DataFrame(data=[pvalues], columns=functions)
    output.to_csv(output_path, encoding='utf-8')


if __name__ == '__main__':
    #metrics = ['Precision', 'Recall' ,'F1', 'Precisionx', 'Recallx', 'F1x', 'PofB', 'PMI', 'Popt', 'IFA']
    metrics = ['IFA', 'PofB', 'PMI', 'Recallx']
    dataset_path = '../Data'
    for metric in metrics:
        # metric ： 指标
        print("Doing Wilcoxon signed rank test in %s ..." % ( metric))
        datas = read_data(dataset_path, metric) # 文件是，几个技术在每个测试集的指标，按照指标形成文件
        # datas就是，把几个方法，在几个测试集上，当前指标的数据
        baseline = ['HDBSCAN']
        for b in range(len(baseline)):
            metric_datas, functions = process_data(datas, baseline[b])
            Wilcoxon_signed_rank_test(metric_datas, functions, metric, b)