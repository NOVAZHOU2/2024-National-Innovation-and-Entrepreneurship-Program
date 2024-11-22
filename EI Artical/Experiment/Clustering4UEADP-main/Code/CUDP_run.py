import numpy as np
import pandas as pd
import warnings
import os
from Measure import Performance, Popt, evaluate_classify
from functions import KmeansCluster, AgglomerativeCluster, BirchCluster, MeanShiftCluster, KmedoidsCluster, \
    MiniBatchKMeansCluster, AffinityPropagationCluster, BsasCluster, CureCluster, DbscanCluster, MbsasCluster, \
    OpticsCluster, RockCluster, Somsc, SyncsomCluster, BangCluster, Kmeans_plusplusCluster, ClaransCluster, \
    EmaCluster, FcmCluster, GmeansCluster, TtsasCluster, Xmeans, ManualUp, HdbScan, Spectral, gaussianMixture, isolationForest

warnings.filterwarnings('ignore')

header = ["Precision", "Recall", "F1", "Precisionx", "Recallx", "F1x", "IFA", "Pofb", "PMI"]

"""
因为跑不了HDBSCAN，我又换了一次py版本，现在是py3.9。
最开始用的是3.10，一堆奇奇怪怪的错，之后用3.6，但是不能跑HDBSCAN，现在3.9，好像没啥大问题

我在代码CUDP里加了三种方法
一个是HDBSCAN，一个是SpectralClustering，一个是GaussianMixture
三个方法在function最后面三个，然后CUDP做了对应的修改，去加载模型
本来还想写孤独森林那个的，但是应该是因为我不会用，导致在CUDP排序计算的时候
出现了除数为0的报错（哎哎，我太菜了导致的）
-----------------------------------------
然后略微修改了RQ1里的画图，图的结果就在RO1/pictures里面
IFA和PMI是越小越好，其他都是越大越好
我的感觉是，HDBSCAN在IFA和PMI表现的挺优秀的，在部分指标可能排名看起来没有特别好
但是没有特别好的指标，其实跟其他方法也差没有很多，比如对于F1x那个排名，其实除了前面四个方法，后面的差别都不大
（我当时改RQ1的画图的时候，属于是一股脑加上去了，没有仔细研究箱图的颜色，所以后面三个方法的箱体都是黑色的）
-----------------------------------------
然后RQ2也是因为加了三个方法，做出了修改
结果图在/RQ2/output/figures
Kmedoids是原本的论文中认为的最优的无监督聚类技术
RQ2好像就是把几个无监督聚类和几个有监督的聚类做了个检验，然后去判断无监督聚类能不能提高EADP的效果
然后那个蓝线应该是ManualUp
然后RQ2原本的代码的画图那里，选择颜色那个方法，提到了一个BH文件夹，但是源码里根本不会出现这个文件夹
所以我直接一股脑改成一个颜色，看看代码的运行结果
而且源码里面，跑Wilcoxon检验的代码里，原本有一个注释，是说“这里要改一下”，我也不知道要改什么（悲）

RQ3和RQ2是一份代码，论文里是把数据搞成表格，我后面再研究一下具体数据是不是符合的

然后RQ4和RQ5就与我们要研究的东西关系不大了，所以其实可以不看了

如果在自己的电脑上跑代码，每次要记得把Result_rq1和Result_rq2的数据拷贝到对应的/Data文件夹下
跑CUDP和supervisedMethod时间会相对长一点，需要小十分钟，毕竟模型不少，而且数据集还是有一定数量的

——2024.11.13 from cxx


几个需要自己安装的包及版本
（快被版本搞死了）
deep-forest               0.1.5
hdbscan                   0.8.39
joblib                    1.4.2
matplotlib                3.3.4
matplotlib-inline         0.1.7
numpy                     1.19.5
openpyxl                  3.1.5
openssl                   3.3.2
pandas                    1.1.5
pyclustering              0.10.1.2
scikit-learn              1.0.2
scikit-learn-extra        0.3.0
scipy                     1.8.0
torch                     2.5.1
xlrd                      1.2.0

然后我在一个文件里出现了错误，是xlrd读文件的错误
报错信息：for elem in self.tree.iter() if Element_has_iter else self.tree.getiterator():
AttributeError: 'ElementTree' object has no attribute 'getiterator'
我参考了这个链接的解法改了，就好了
https://blog.csdn.net/m0_59860603/article/details/121697922


缺少的数据（每个软件都缺初版本感觉）：
Ant1.3
Camel1
Ivy1.1
Jedit3.2
Jedit4.3
Log4j1.0
lucene2.0
poi1.5
Synapse1.0
Velocity1.4
xalan2.4
xerces1.1


"""

def ClassificationByFeature(feature):
    """
    把特征全部变成倒数后返回
    """
    result = []
    for f in feature:
        result.append(1/f)
    return np.array(result)


def transform_data(original_data):
    """
    :param original_data: 输入的原始数据
    :return: 数据中的特征矩阵和每个模块有没有bug
    """

    original_data = original_data.iloc[:, :]
    original_data = np.array(original_data)
    # 拷贝了一下数据
    k = len(original_data[0])
    # 获取列数

    original_data = np.array(original_data)
    # 为什么感觉这行有点多余
    original_data_X = original_data[:, 0:k - 1]
    # 第k - 1列是y值，也就是bugs数量，其他是特征，这里是获取所有特征
    original_data_y = original_data[:, k - 1]
    # 这里获得了y值

    y_list = []
    for i in original_data_y:
        if i >= 1:
            y_list.append(1)
        else:
            y_list.append(0)
    # y_list是每一个模块有没有bug
    return original_data_X, y_list # 返回了特征和每个模块有没有bug的01序列

def calculateASFM(X):
    features_sum = np.sum(X)
    asfm = features_sum / (len(X)*len(X[0]))
    return asfm

def devideCluster(y_predict,X):
    """
    :param y_predict:聚类后的每个类所属的聚类标签
    :param X: 特征
    :return: 聚类asfm大于平均值的特征（即预测为有缺陷），对应数量的1，小于平均值的特征（预测为无缺陷），对应数量的0
    """

    n_ = len(set(y_predict))
    # 聚类的类数应该
    res = [0] * n_
    count = [0] * n_
    index = [[]] * n_
    print("n_：", n_)
    for i in set(y_predict):
        temp = []
        for j in range(len(y_predict)):
            # len(y_predict)其实就是当前软件模块数量
            # y_predict其实就是将每个模块的特征值提取之后，用对应的聚类方法进行聚类
            # 然后会返回出每个模块属于的聚类标签
            # y_predict就是这个标签
            if i == y_predict[j]:
                res[i] += np.sum(X[j]) # 把对应模块的特征值加起来了，然后加到了对应聚类的res里？
                count[i] += 1 # 计数器
                temp.append(j) # 存了下标
        index[i] = temp # index[i]是聚类结果标签为i的软件模块的序号的序列
    asfm = [res[i] / count[i] for i in range(len(res))] # 计算特征平均值？
    # asfm是一个长度为n_，asfm[i]是res[i]/count[i]
    mean = np.mean(asfm) # 计算asfm的平均值
    # 就是我把每个聚类里所有特征都加起来，然后除以了点数，求出了平均值
    # 然后再把所有聚类的平均值再平均
    # 然后下面代码的意思应该是把期望超过mean值的，都视为是有缺陷的
    # 没超过就是没缺陷的
    defectX = []
    undefectX = []

    # len(asfm)：n_
    for i in range(len(asfm)):
        if asfm[i] >= mean: # 大于平均值
            defectX += np.array(X)[index[i]].tolist()
            # 把X中的index[i]的行提取出来，形成List
            # 也就是把所有软件特征，对应当前聚类的行，全部提取出来了
            #print(i)
            # print(np.array(X)[index[i]])
            # defectX：asfm值大于平均值的模块特征toList
        else:
            undefectX += np.array(X)[index[i]].tolist()
            # undefectX：asfm值小于平均值的模块特征toList
    defectY = [1]*len(defectX)
    # defectY：对应defectX对应为1（有bug）
    undefectY = [0]*len(undefectX)
    # defectX：就是把所有asfm大于平均值的聚类中的模块的特征添加进里面
    return defectX, defectY, undefectX, undefectY

if __name__ == '__main__':
    dataset_train = pd.core.frame.DataFrame()
    dataset_test = pd.core.frame.DataFrame()
    # print(dataset_test)
    # print(dataset_train)
    folder_path = '../CrossversionData/' # 给的代码里好像没有这个文件夹啊
    # 我创建了这个文件夹，然后对着论文里的结果图，往里面建了跨版本验证的数据文件夹
    # 每个文件夹两个文件，就是同一个软件两个不同的版本
    # 但是因为数据不全，所以文件其实也是不全的，缺了几个文件夹其实
    for root, dirs, files, in os.walk(folder_path):
        if root == folder_path:
            # 遍历/CrossversionData/下的文件夹
            thisroot = root
            # thisroot = '../CrossversionData/'

            for dir in dirs: # 遍历子目录
                dir_path = os.path.join(thisroot, dir) # 拼接出子目录的完整路径
                # dir_path = '/CrossversionData/子目录名字' —— 例如：/CrossversionData/ant-1.4-1.5

                for root, dirs, files, in os.walk(dir_path):
                    # files是/CrossversionData/子目录名字/下的所有文件
                    # files是所有文件名称的List
                    # 其实就是两个文件
                    # 例如：[ant-1.4.csv, ant-1.5.csv]
                    if (files[0][-7:-4] < files[1][-7:-4]):
                        # 就是比较了两个文件的版本号，旧版本是训练集，新版本的测试集
                        file_path_train = os.path.join(dir_path, files[0])
                        file_path_test = os.path.join(dir_path, files[1])

                        trainingfile = files[0]
                        testingfile = files[1]
                    else:
                        file_path_train = os.path.join(dir_path, files[1])
                        file_path_test = os.path.join(dir_path, files[0])
                        trainingfile = files[1]
                        testingfile = files[0]
                    # print('train', file_path_train)
                    # print('test', file_path_test)
                    # print('***********************************')

                    dataset_train = pd.read_csv(file_path_train)
                    dataset_test = pd.read_csv(file_path_test)
                    # 读了训练集和测试集的数据
                    training_data_x, training_data_y = transform_data(
                        dataset_train)
                    testing_data_x, testing_data_y = transform_data(
                        dataset_test)
                    # ……_data_x：特征值，……_data_y：有没有bug（01序列）

                    functions = [(ManualUp, testing_data_x),
                                 (KmeansCluster, 2, testing_data_x),
                                 (AgglomerativeCluster, 2, testing_data_x),
                                 (BirchCluster, 2, testing_data_x),

                                 (MiniBatchKMeansCluster, 2, testing_data_x),
                                 (MeanShiftCluster, testing_data_x),
                                 (AffinityPropagationCluster, testing_data_x),

                                 (BsasCluster, 2, 1.0, testing_data_x),

                                 (DbscanCluster, 3, 0.5, testing_data_x),
                                 (HdbScan, 2, testing_data_x),
                                 (MbsasCluster, 2, 1.0, testing_data_x),
                                 (OpticsCluster, 3, 0.5, testing_data_x),
                                 (RockCluster, 2, 1, testing_data_x),
                                 (Somsc, 2, testing_data_x),
                                 (SyncsomCluster, testing_data_x),

                                 (BangCluster, testing_data_x),

                                 (EmaCluster, testing_data_x),
                                 (FcmCluster, testing_data_x),
                                 (GmeansCluster, testing_data_x),
                                 (TtsasCluster, testing_data_x),
                                 (Xmeans, testing_data_x)


                                 ]
                    # function其实就是每个聚类方法以及其构造函数需要的数据的列表
                    # 总共23个方法

                    for func, *args in functions:
                        y_predict, func_name = func(*args)
                        # 调用对应聚类的方法进行聚类
                        performance_result = [[0] * 9] * 20
                        # 总共有20个特征
                        iterations = 1
                        print('Function：{0}'.format(func_name))

                        for p in range(iterations):
                            # Use ASFM to classify clustering results as defective or defect-free
                            # step2：Sort the defects according to their eigenvalues (20 dimensions) from small to large.
                            # 1.Extract features of defective modules
                            # 1. 提取缺陷模型的特征。
                            print(func_name)
                            defectX, defectY, undefectX, undefectY = devideCluster(y_predict, testing_data_x)



                            # 单纯跑结果好像可以不用下面这些
                            # 2.Sort each feature of the defective module, temporarily replacing it with the first column.
                            # 2. 对缺陷模块进行排序，暂时替换为第一列
                            #sort_axis = np.argsort(defectX[:,0]) # 这里排序会报错 —— list indices must be integers or slices, not tuple
                            # 因为不能直接利用[:, 0]去获得第一列的元素
                            # sorted_defectX = np.array(defectX)[sort_axis]

                            # 我理解之后，感觉改成这样
                            # 就是按照第一列的元素进行排序
                            first_column = [defectX[i][0] for i in range(0, len(defectX))]
                            sort_axis = np.argsort(first_column)
                            sorted_defectX = np.array(defectX)[sort_axis]

                            # After sorting, check the first 20% of the modules in LOC; if there are no defects, they need to be sorted and spliced.
                            # 排序后，检查 LOC 中前 20% 的模块;如果没有缺陷，则需要对其进行分类和拼接。
                            # sorted_undefectX = np.array(undefectX)[sort_axis]
                            # 3.Merge defective and non-defective
                            if len(sorted_defectX) != 0 & len(undefectX) != 0:
                                predtestX = np.vstack([sorted_defectX,undefectX])
                            # 把有缺陷和没有缺陷的软件特征进行连接，形成新的二维数组
                            # 一开始没有这个if，但是我跑第一个聚类方法的时候，出现了undefectX是空的，导致连接报错
                            index_real = [i for i in range(len(y_predict)) if testing_data_y[i] > 0]
                            # Extract the number of lines of code for workload interception



                            testingcodeN = testing_data_x[:, 10] # 提取所有模块的LOC数量
                            defectX = np.array(defectX)
                            undefectX = np.array(undefectX)
                            # print(undefectX)
                            for f in range(len(defectX[0])):
                                # 感觉是特征数量
                                if undefectX.any():
                                    # 在被预测为干净的软件模块中，如果有特征的值是不为0的
                                    feature_d = defectX[:, f] # 有缺陷的模块的第f个特征
                                    feayure_u = undefectX[:, f] # 好的，我感觉学长这里单词打的有点问题，这是没缺陷的
                                    density_d = ClassificationByFeature(feature_d) # 倒数
                                    density_u = ClassificationByFeature(feayure_u)
                                    sort_axis_d = np.argsort(-density_d) # 就是按照当前特征值倒数从大到小的索引数组（其实就是特征值从小到大？）
                                    # print("density_d: ", density_d)
                                    # print("sort_axis_d: ", sort_axis_d)
                                    sorted_defectX = defectX[sort_axis_d] # 根据这个索引排序后
                                    sort_axis_u = np.argsort(-density_u) # 对没缺陷的也排一下
                                    sorted_undefectX = undefectX[sort_axis_u]
                                    # Merge labels to sort y
                                    # predtestX = np.append(sorted_defectX, sorted_undefectX, axis=0)
                                    sort_y = np.append(sort_axis_d, sort_axis_u) # 索引数组
                                    # predtestX：把有缺陷的特征和没缺陷的特征链接起来，有缺陷的在前面
                                    # sort_y：把模块排序，有缺陷的在前面，两部分分别按照第f个缺陷排序的
                                else:
                                    feature = testing_data_x[:, f]
                                    density = ClassificationByFeature(feature)
                                    sort_y = np.argsort(-density)

                                testY = np.array(testing_data_y)[sort_y].tolist() # 真实值按照索引排序后（01序列）
                                predY = np.append(defectY, undefectY) # 预测值，和真实值都是01序列
                                sorted_code = testingcodeN[sort_y] # 排序后的所有模块的LOC数量
                                Precisionx, Recallx, F1x, IFA, PofB, PMI= Performance(testY, predY, sorted_code)
                                precision, recall, f1 = evaluate_classify(testY, predY)
                                header = ["Precision", "Recall", "F1", "Precisionx", "Recallx", "F1x", "IFA", "Pofb",
                                          "PMI"]
                                temp = [precision, recall, f1, Precisionx, Recallx, F1x, IFA, PofB, PMI] # 两类指标的值
                                performance_result[f] = list(np.add(performance_result[f], temp))

                        performance_result = np.array(performance_result) / iterations
                        # 这一段我的理解就是，我们前面利用聚类技术，把软件模块分为了有缺陷和没有缺陷两类，但是是没有具体排序的
                        # 然后不是有很多个特征吗（20个），我们可以选择不同的特征作为基准，然后去排序，不同的选择会有不同的性能结果
                        # 然后就把按照不同特征排序之后的效果记录下来了（就是那个performance_result数组）
                        # 把数据写入到Output文件夹中
                        datasetfile = trainingfile[:-4] + '_' + testingfile[:-4]
                        df = pd.DataFrame(data=performance_result, columns=header)
                        # performance_result：行数是特征值，也就是20，每一列就是不同的性能指标
                        # 表示的就是，按照每个特征进行排序，得到的结果的指标情况
                        if not os.path.exists("../Output/{0}".format(datasetfile)):
                            os.makedirs("../Output/{0}".format(datasetfile))
                        df.to_csv("../output/{0}/{1}.csv".format(datasetfile, func_name), index=False)
