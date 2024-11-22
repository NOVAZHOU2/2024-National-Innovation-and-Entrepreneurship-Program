import os
import pandas as pd
import openpyxl # 要多导这个包，不然写文件会报错

# 就是获得每个指标，不同聚类方法（22种无监督聚类方法）的性能
# 等等，header里面是22种，但是function里面还是23种
# header中并没有ManualUp

"""
这里我改了一点，functions里的ManualUp被我删掉了
因为这个模型跑出来的结果，尤其是Recall，无论什么数据集，结果好像都是1
然后在scott-knott检验里面，会有涉及到求差，而且要做除法
然后就出现了除数为0之类的问题，然后就报错了
所以我删掉了，而且论文的图里面确实是没有ManualUp的数据的
跑出来的图，怎么说呢，能看，但是有些和论文里的还是有一点差距的
感觉应该是因为数据集不太一样的原因
"""

header = [
    'Kmeans',  'Xmeans', 'FCM', 'Gmeans', 'MBKmeans',

    'BIRCH',  'ROCK', "AHC",
    "DBSCAN", "HDBSCAN", 'OPTICS', 'MeanShift',
    'SOMAC', 'SYNCSOM', 'EMA',
    "AP",
    "BSAS", 'MBSAS', 'TTSAS',
    "BANG"
]

functions = [
             'Kmeans',  'Xmeans', 'Fcm', 'Gmeans', 'MiniBatchKmeans',
             'Birch', 'Rock', "Agglomerative",
             "Dbscan", 'Hdbscan', 'Optics', 'MeanShift',
             'Somsc', 'Syncsom', 'EMA',
             "AP",
             "Bsas", 'Mbsas', 'Ttsas',
             "Bang"
                      ]
# 好的，我觉得问题出在这里了，不能多第一个ManualUp，ManualUp会把所有模块都预测为有缺陷
# 然后Recall值就全是1
# 然后跑R语言代码，就是计算那个，Scott-Knott检验的时候，就会报错
features = ['WMC','DIT','NOC','CBO','RFC','LCOM','Ca','Ce','NPM','LCOM3',
            'LOC','DAM','MOA','MFA','CAM','IC','CBM','AMC','Max_CC','Avg_CC']

def writeToFile(file, name, argMax):
    # 单纯写文件
    new_header = ['Cross-version']
    for h in functions:
        new_header.append(h + '.' + features[argMax[h]])
        # 这里有个问题，他点后面是他按照什么特征排序最优，会导致后面的代码可能找不到对应的数据好像
        # 或者可以看写的文件里的表头，可以看到聚类名字后面会有一个.LOC之类的东西，就是按照这个特征排序是最优的
        # 但是后面的题目的代码都默认是.LOC，然后可能就有点问题了
    dataf = pd.DataFrame(columns=new_header, data=file)
    path = "../Result_rq1"
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = "{0}/{1}.xlsx".format(path, name)
    dataf.to_excel(file_path, index=False)

def getArgMax(path):
    folder_path = path
    init_list = [[0.]*20]*20 # 因为确实是22个方法，所以*22没错，这也验证了，可能要把ManualUp删掉的猜想，不然这个程序都会报错
    # 就是一个20 * 22的数组
    result_data = pd.DataFrame(columns=functions, data=init_list) # 列是方法，data，初始化数据为init_list
    for root, dirs, files, in os.walk(folder_path):
        # root：Output
        # dirs：['ans-1.4_ant-1.5'……]
        # files：[]
        if root == folder_path:
            thisroot = root
            # dirs：每个训练_测试文件夹
            for dir in dirs:
                # dir：具体一个训练_测试文件夹
                dir_path = os.path.join(thisroot, dir)
                # 拼接完整路径
                for root, dirs, files, in os.walk(dir_path):
                    #root：版本文件夹
                    #dirs：[]
                    #files：各个聚类方法的性能指标文件
                    for f in functions:
                        # 扫了全部聚类方法
                        #funcname = f[:-4]
                        file_path = os.path.join(dir_path, f+'.csv')
                        dataset = pd.read_csv(file_path) # 读取具体文件数据
                        pofb = dataset.loc[:, "Pofb"]
                        # 读取出Pofb这一列
                        result_data[f] += pofb
                        # 相当于扫了所有的输出数据集，把每个方法的pofb累加了
                        # 每一行对应按照不同的特征进行排序，相加的时候也是矩阵加法
                        # result_data的含义其实就是，按照不同基准排序后，Pofb的值（把不同软件，不同测试集的Pofb加起来了）
    print(result_data)
    argmax = result_data.idxmax() # 返回了每一列的表头和最大值索引，也就是每一个聚类方法最大值（就是根据不同基准排序的最大值）的索引
    # 我的理解就是，每种聚类方法不是有根据不同特征排序吗，挑一个最好的（因为Prob是越大越好）
    return argmax

def getMeasure(argMax):
    resultlist_pofb = []
    resultlist_ifa = []
    resultlist_pmi = []
    resultlist_precision = []
    resultlist_recall = []
    resultlist_f1 = []
    resultlist_precision20 = []
    resultlist_recall20 = []
    resultlist_f120 = []

    for root, dirs, files, in os.walk(folder_path):
        # root：Output
        # dirs：[ant-1.4_ant-1.5……]
        # files：[]
        if root == folder_path:
            thisroot = root
            for dir in dirs:
                # dir：数据版本文件夹名
                dir_path = os.path.join(thisroot, dir)
                # dir_path = 'Output/版本文件夹名/'
                for root, dirs, files, in os.walk(dir_path):
                    # root：版本文件夹名
                    # dirs：[]
                    # files：[每个聚类文件]
                    temp_ifa = []
                    temp_pofb = []
                    temp_pmi = []
                    temp_precision = []
                    temp_recall = []
                    temp_f1 = []
                    temp_precision20 = []
                    temp_recall20 = []
                    temp_f120 = []

                    temp_ifa.append(dir)
                    temp_pofb.append(dir)
                    temp_pmi.append(dir)
                    temp_precision.append(dir)
                    temp_recall.append(dir)
                    temp_f1.append(dir)
                    temp_precision20.append(dir)
                    temp_recall20.append(dir)
                    temp_f120.append(dir)


                    for f in functions:
                        #funcname = f[:-4]
                        feature_n = argMax[f] # 该方法的最大值下标
                        file_path = os.path.join(dir_path, f+'.csv')
                        # 把之前存的最大值索引对应行的数据全部读出来
                        dataset = pd.read_csv(file_path)
                        pofb = dataset["Pofb"].loc[feature_n]
                        ifa = dataset['IFA'].loc[feature_n]
                        pmi = dataset['PMI'].loc[feature_n]
                        precision = dataset['Precision'].loc[feature_n]
                        recall = dataset['Recall'].loc[feature_n]
                        f1 = dataset['F1'].loc[feature_n]
                        precision20 = dataset['Precisionx'].loc[feature_n]
                        recall20 = dataset['Recallx'].loc[feature_n]
                        f120 = dataset['F1x'].loc[feature_n]

                        # 每个聚类方法的指标值
                        temp_ifa.append(ifa)
                        temp_pofb.append(pofb)
                        temp_pmi.append(pmi)
                        temp_precision.append(precision)
                        temp_recall.append(recall)
                        temp_f1.append(f1)
                        temp_precision20.append(precision20)
                        temp_recall20.append(recall20)
                        temp_f120.append(f120)

                    # 每个数据版本是一行数据
                    resultlist_ifa.append(temp_ifa)
                    resultlist_pofb.append(temp_pofb)
                    resultlist_pmi.append(temp_pmi)
                    resultlist_precision.append(temp_precision)
                    resultlist_recall.append(temp_recall)
                    resultlist_f1.append(temp_f1)
                    resultlist_precision20.append(temp_precision20)
                    resultlist_recall20.append(temp_recall20)
                    resultlist_f120.append(temp_f120)


    # 就是把每个聚类方法最好性能的结果读出来，然后分别添加到每个性能文件里
    writeToFile(resultlist_ifa, "IFA", argMax)
    writeToFile(resultlist_pofb, "Pofb", argMax)
    writeToFile(resultlist_pmi, "PMI", argMax)
    writeToFile(resultlist_precision, "Precision", argMax)
    writeToFile(resultlist_recall, "Recall", argMax)
    writeToFile(resultlist_f1, "F1", argMax)
    writeToFile(resultlist_precision20, "Precisionx", argMax)
    writeToFile(resultlist_recall20, "Recallx", argMax)
    writeToFile(resultlist_f120, "F1x", argMax)



if __name__ == '__main__':
    dataset = pd.core.frame.DataFrame()
    folder_path = '../Output/'

    argmax = getArgMax(folder_path)
    print(argmax)
    getMeasure(argmax)

