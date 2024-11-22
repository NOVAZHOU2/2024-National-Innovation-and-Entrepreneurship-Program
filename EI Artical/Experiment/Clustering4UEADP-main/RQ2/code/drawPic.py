import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

"""
可以看到画图路径哪里，output后面有子文件夹BH
但是我跑完Wilcoxon是没有这个文件夹的，然后我就傻了
而且数据标题也对不上的感觉
后面需要的话再研究一下吧
"""

def load_color(metric, functions):
    print(metric)
    # colors_path = '../output/BH/%s.csv' % (metric)
    # datas = pd.read_csv(colors_path)

    colors = []
    for function in functions:
        colors.append('red')
        # if datas[function][0] < 0.05:
        #     colors.append('red')
        # else:
        #     colors.append('black')

    return colors

def drawFigure(metric_datas, functions, metric):
    ymax = 0
    ymin = 100
    for data in metric_datas:
        if ymax < max(data):
            ymax = max(data)
        if ymin > min(data):
            ymin = min(data)

    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.tick_params(direction='in')

    xticks = np.arange(1, len(functions) * 2.5, 2.5)
    figure = ax.boxplot(metric_datas,
                        notch=False,  # notch shape
                        sym='r+',  # blue squares for outliers
                        vert=True,  # vertical box aligmnent
                        meanline=True,
                        showmeans=False,
                        patch_artist=False,
                        showfliers=False,
                        positions=xticks,
                        boxprops={'color': 'red'}
                        )

    colors = load_color(metric, functions)
    for i in range(len(colors)):
        k = figure['boxes'][i]
        k.set(color=colors[i])
        k = figure['medians'][i]
        k.set(color=colors[i], linewidth=2)
        k = figure['whiskers'][2 * i:2 * i + 2]
        for w in k:
            w.set(color=colors[i], linestyle='--')
        k = figure['caps'][2 * i:2 * i + 2]
        for w in k:
            w.set(color=colors[i])

    plt.xlim((0, 15))
    #functions_new = []
    #for func in functions:
    #    functions_new.append(func[:-4])
    plt.xticks(xticks, functions, rotation=25, weight='heavy', fontsize=12, ha='center')
    plt.yticks(fontsize=12,weight='heavy')
    if metric not in ['Precision', 'Recall' ,'F1', 'IFA','Popt']:
        if metric[-1] == 'x':
            plt.ylabel(metric[:-1] + '@20%', fontsize=12, weight='heavy')
        else:
            plt.ylabel(metric+'@20%', fontsize=12,weight='heavy')
    else:
        plt.ylabel(metric, fontsize=12,weight='heavy')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.axhline(y=0, color='blue', lw=1)

    Path('../output/figures/').mkdir(parents=True, exist_ok=True)
    output_path = '../output/figures/%s.jpg' % (metric)
    foo_fig = plt.gcf()
    foo_fig.savefig(output_path, format='jpg', dpi=1000, bbox_inches='tight')
    plt.clf()
    plt.close()


def processDatas(datas):
    """
    把ManualUp方法的结果作为了一个基准线好像
    然后看了三个方法跟这个基准线的差值
    :param datas: 某个性能的所有的数据
    :return: 差值, 方法名
    """
    baseline_data = 'ManualUp'
    metric_datas = []
    functions = []
    Clusterings = ['HDBSCAN', 'ManualUp', 'RF', 'GB', 'CBS+']
    for function in Clusterings:
        metric_datas.append(datas[function] - datas[baseline_data])
        functions.append(function)
    return metric_datas, functions

def read_data(dataset_path, metric):
    functions = ['HDBSCAN', 'ManualUp', 'RF', 'GB', 'CBS+']
    datas = {}
    for function in functions:
        data_path = '{0}/{1}.xlsx'.format(dataset_path,metric)
        raw_datas = pd.read_excel(data_path)

        raw_datas = raw_datas[function].values

        datas[function] = raw_datas

    return datas

if __name__ == '__main__':
    # metrics = ['Precision', 'Recall', 'F1', 'Precisionx','Recallx', 'PofB', 'PMI', 'IFA', 'F1x', 'Popt']
    metrics = ['Precision', 'Recall', 'F1', 'Precisionx','Recallx', 'PofB', 'PMI', 'IFA', 'F1x']
    # 最后这个Popt是什么的
    # dataset_path = '../../Result_rq1'
    dataset_path = '../../Result_rq2'
    # 应该是这里路径写错了啊，是rq2
    for metric in metrics:
        datas = read_data(dataset_path, metric)
        print("metric:", metric)
        print(datas)
        metric_datas, functions = processDatas(datas)
        print(metric_datas)
        print(functions)
        drawFigure(metric_datas, functions, metric)
