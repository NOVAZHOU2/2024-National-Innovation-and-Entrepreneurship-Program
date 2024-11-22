import deepforest.cascade
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
import os
from deepforest.cascade import CascadeForestClassifier
from Measure import Performance, Popt, evaluate_classify

# 其实和前面的CUDP差不多，不过这个是搞监督聚类方法的

warnings.filterwarnings('ignore')

header = ["Precision", "Recall", "F1", "Precisionx", "Recallx", "F1x", "IFA", "Pofb", "PMI"]

from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

def feature_evaluation(X, y, classifier):
    scores = cross_val_score(classifier, X, y, cv=10)
    precision, recall, _, _ = precision_recall_fscore_support(y, classifier.predict(X), average='binary')
    accuracy = np.mean(scores)
    roc_auc = roc_auc_score(y, classifier.predict_proba(X)[:, 1])
    return precision, recall, accuracy, 2 * (precision * recall) / (precision + recall), roc_auc

def wrapper_method(X, y):
    F = list(range(X.shape[1]))
    selF = F.copy()
    all_feature_results = []

    min_features = 1  # set min features
    max_iterations = 10  # set max iteration

    for _ in range(max_iterations):
        results = []

        for feature in selF:
            classifier = GaussianNB()  # Using Naive Bayes for Wrapper method

            # Create a subset of features
            subset_X = X[:, [feature]]

            # Fit the classifier
            classifier.fit(subset_X, y)

            # Evaluate performance
            precision, recall, accuracy, f_measure, roc_auc = feature_evaluation(subset_X, y, classifier)
            results.append((f_measure, feature))

        # Record the results for all features in this iteration
        all_feature_results.extend(results)

        # Determine the best feature
        best_result = max(results, key=lambda x: x[0])
        if len(selF) < min_features:
            break  # Break the loop if there are too few features

        best_feature = best_result[1]

        # Recompute feature scores using Naive Bayes (if needed)

        # Identify removeF, the 50% of features of selF with the lowest feature evaluation
        remove_count = int(len(selF) * 0.5)
        removeF = [x[1] for x in sorted(results, key=lambda x: x[0])[:remove_count]]

        # Update selF
        selF = list(set(selF) - set(removeF))

    # Select the top half of features based on the overall results
    top_half_features = [x[1] for x in sorted(all_feature_results, key=lambda x: x[0], reverse=True)[:len(F)//2]]

    return top_half_features



def ClassificationBySERS(X, Y, testX, testingcodeN):
    selected_features = wrapper_method(X, Y)
    selected_features = sorted(selected_features)
    print("Selected Features:", selected_features)
    selected_X = np.array(X)[:, selected_features]
    selected_testX = np.array(testX)[:, selected_features]
    NB = GaussianNB()
    NB_pred = NB.fit(selected_X, Y).predict_proba(selected_testX)
    NB_pred = [p[1] for p in NB_pred]
    NB_pred3 = []
    for j in range(len(NB_pred)):
        if testingcodeN[j] != 0:
            if NB_pred[j] > 0.5:
                NB_pred3.append((NB_pred[j] / testingcodeN[j]) + 100000)
            else:
                NB_pred3.append((NB_pred[j] / testingcodeN[j]) - 100000)
        else:
            NB_pred3.append(-100000000)
    return NB_pred3


def optimizeParameter(classifier, x, y, params):

    """
    会返回训练好的模型
    :param classifier: 要求的模型类型
    :param x: 特征
    :param y: 结果
    :param params: 标准化？
    :return: 训练好的模型
    """

    model = GridSearchCV(estimator=classifier, param_grid=params, scoring='f1', cv=10)
    model.fit(x, y)
    best_model = model.best_estimator_
    print(model.best_params_)
    print('best f1:%f' % model.best_score_)
    return best_model

def ClassificationByLR(X, Y, testX, testingcodeN):
    """

    :param X: 训练集X
    :param Y: 训练集Y
    :param testX: 测试集X
    :param testingcodeN: 所有模块的LOC
    :return:
    """
    LinearR = LinearRegression()
    LR_tuned_parameters = {'normalize': [True, False]}

    model = optimizeParameter(LinearR, X, Y, LR_tuned_parameters)
    LR_pred = model.predict(testX)
    LR_pred3 = []
    for j in range(len(LR_pred)):
        if testingcodeN[j] != 0:
            if LR_pred[j] > 0.5:
                # 有缺陷的概率大于50%
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) + 100000)
                # 不过我不太理解后面这个100000是怎么来的
            else:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) - 100000)
        else:
            LR_pred3.append(-100000000)

    return LR_pred3

def ClassificationByEASC(X, Y, testX, testingcodeN):
    NB = GaussianNB()
    LR_pred = NB.fit(X, Y).predict_proba(testX)
    LR_pred = [p[1] for p in LR_pred]
    LR_pred3 = []
    for j in range(len(LR_pred)):
        if testingcodeN[j] != 0:
            if LR_pred[j] > 0.5:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) + 100000)
            else:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) - 100000)
        else:
            LR_pred3.append(-100000000)

    return LR_pred3

def ClassificationByCbs(X, Y, testX, testingcodeN):
    LR = LogisticRegression()
    LR_tuned_parameters = [{'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
                           'penalty':['l1','l2', 'elasticnet', 'none'],
                           'solver':['liblinear','lbfgs', 'sag', 'newton-cg', 'saga']}
                           ]

    model = optimizeParameter(LR, X, Y, LR_tuned_parameters)
    LR_pred = model.predict_proba(testX)


    LR_pred = [p[1] for p in LR_pred]
    LR_pred3 = []
    for j in range(len(LR_pred)):
        if testingcodeN[j] != 0:
            if LR_pred[j] > 0.5:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) + 100000)
            else:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) - 100000)
        else:
            LR_pred3.append(-100000000)
    return LR_pred3

def ClassificationByDeepForest(X, Y, testX, testingcodeN):
    model = CascadeForestClassifier()
    model.fit(X, Y)
    LR_pred = model.predict_proba(testX)

    LR_pred = [p[1] for p in LR_pred]
    LR_pred3 = []
    for j in range(len(LR_pred)):
        if testingcodeN[j] != 0:
            if LR_pred[j] > 0.5:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) + 100000)
            else:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) - 100000)
        else:
            LR_pred3.append(-100000000)

    return LR_pred3

def ClassificationByRandomForest(X, Y, testX, testingcodeN):
    model = RandomForestClassifier()
    RFC_tuned_parameters = {'n_estimators': [i for i in range(10,150,10)]}

    model = optimizeParameter(model, X, Y, RFC_tuned_parameters)
    LR_pred = model.predict_proba(testX)

    LR_pred = [p[1] for p in LR_pred]  
    LR_pred3 = []
    for j in range(len(LR_pred)):
        if testingcodeN[j] != 0:
            if LR_pred[j] > 0.5:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) + 100000)
            else:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) - 100000)
        else:
            LR_pred3.append(-100000000)

    return LR_pred3

def ClassificationByGradientBoosting(X, Y, testX, testingcodeN):
    model = GradientBoostingClassifier()
    GB_tuned_parameters={'n_estimators':[i for i in range(10,150,10)]
                         }
    model = optimizeParameter(model, X, Y, GB_tuned_parameters)

    model.fit(X, Y)
    LR_pred = model.predict_proba(testX)

    LR_pred = [p[1] for p in LR_pred]
    LR_pred3 = []
    for j in range(len(LR_pred)):
        if testingcodeN[j] != 0:
            if LR_pred[j] > 0.5:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) + 100000)
            else:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) - 100000)
        else:
            LR_pred3.append(-100000000)

    return LR_pred3

def transform_data(original_data):

    """
    :param original_data: 原始数据
    :return: 所有特征，有无bug的01序列
    """

    original_data = original_data.iloc[:, :]

    original_data = np.array(original_data)

    k = len(original_data[0])
    # 特征数量

    original_data = np.array(original_data)
    original_data_X = original_data[:, 0:k - 1]
    # 同样是获取所有特征

    original_data_y = original_data[:, k - 1]
    # 获取bug数量
    y_list = []
    for i in original_data_y:
        if i >= 1:
            y_list.append(1)
        else:
            y_list.append(0)
    # 也是生成有没有bug的01序列
    return original_data_X, y_list


def calculateASFM(X):
    features_sum = np.sum(X)
    asfm = features_sum / (len(X)*len(X[0]))
    return asfm

def devideCluster(y_predict,X):
    n_ = len(set(y_predict))
    res = [0] * n_
    count = [0] * n_
    index = [[]] * n_
    for i in set(y_predict):
        temp = []
        for j in range(len(y_predict)):
            if i == y_predict[j]:
                res[i] += np.sum(X[j])
                count[i] += 1
                temp.append(j)
        index[i] = temp
    asfm = [res[i] / count[i] for i in range(len(res))]
    mean = np.mean(asfm)
    defectX = []
    undefectX = []
    for i in range(len(asfm)):
        if asfm[i] >= mean:
            defectX += np.array(X)[index[i]].tolist()
        else:
            undefectX += np.array(X)[index[i]].tolist()
    defectY = [1]*len(defectX)
    undefectY = [0]*len(undefectX)
    return defectX, defectY, undefectX, undefectY

if __name__ == '__main__':
    dataset_train = pd.core.frame.DataFrame()
    dataset_test = pd.core.frame.DataFrame()

    folder_path = '../CrossversionData/'

    for root, dirs, files, in os.walk(folder_path):
        if root == folder_path:
            thisroot = root
            for dir in dirs:
                dir_path = os.path.join(thisroot, dir)

                for root, dirs, files, in os.walk(dir_path):
                    if (files[0][-7:-4] < files[1][-7:-4]):
                        # 同样是旧版本用作训练集，新版本是测试集
                        file_path_train = os.path.join(dir_path, files[0])
                        file_path_test = os.path.join(dir_path, files[1])
                        trainingfile = files[0]
                        testingfile = files[1]
                    else:
                        file_path_train = os.path.join(dir_path, files[1])
                        file_path_test = os.path.join(dir_path, files[0])
                        trainingfile = files[1]
                        testingfile = files[0]

                    #print('files[0][-7:-4]', files[0][-7:-4])
                    #print('files[1][-7:-4]', files[1][-7:-4])
                    #print(files[0][-7:-4], '>', files[1][-7:-4])
                    print('train', file_path_train)
                    print('test', file_path_test)
                    print('***********************************')

                    dataset_train = pd.read_csv(file_path_train) # 训练集数据
                    dataset_test = pd.read_csv(file_path_test) # 测试集数据
                    training_data_x, training_data_y = transform_data(dataset_train) # 获得特征数据和有无bug的01序列
                    testing_data_x, testing_data_y = transform_data(dataset_test)
                    testingcodeN = testing_data_x[:, 10] # 所有模块的LOC

                    # 这里有点点不懂，functions有7个，但是名字只有5个，而且没有EATT？
                    # 我对着论文看了一下，论文里是没有DF和EASC的，但是有EATT的，然后CBS+，GB，RF都属于CBS+
                    # 不过后面的实验数据，貌似只跟名字里的五个有关系，所以为什么论文里的表是有EATT的
                    functions = {'CBS+': ClassificationByCbs,
                                 'EALR':ClassificationByLR,
                                 'EASC':ClassificationByEASC,
                                 'RF':ClassificationByRandomForest,
                                 'GB':ClassificationByGradientBoosting,
                                 'DF':ClassificationByDeepForest,
                                 'SERS':ClassificationBySERS
                                 }
                    functions_name = ['EALR',  'CBS+', 'RF', 'GB', 'SERS']
                    # functions_name = ['SERS']
                    #functions_name = ['EALR']
                    for fname in functions_name:
                        print('=========================================')
                        print(fname)
                        print('=========================================')
                        resultlist = []
                        y_predict = functions[fname](training_data_x, training_data_y, testing_data_x, testingcodeN)
                        # 预测值，用训练集训练，去预测测试集
                        # 反正总体看过去，都是调库去跑结果，但是得到预测结果的那个式子不懂，结果都是负数，而且负很大（也有正数，但是绝对值都很大感觉）

                        sort_y = np.argsort(y_predict)[::-1]
                        # 倒序排序（从大到小）

                        testY = np.array(testing_data_y)[sort_y].tolist() # 真实值按照这个下标，重新排序
                        sorted_code = testingcodeN[sort_y] # LOC数组也按照这个下标重新排序
                        Precisionx, Recallx, F1x, IFA, PofB, PMI = Performance(testY, y_predict, sorted_code)
                        precision, recall, f1 = evaluate_classify(testY, y_predict)
                        header = ["Precision", "Recall", "F1", "Precisionx", "Recallx", "F1x", "IFA", "Pofb",
                                  "PMI"]
                        performance_result = [precision, recall, f1, Precisionx, Recallx, F1x, IFA, PofB, PMI]
                        # 同样去获得那几个指标
                        resultlist.append(performance_result)


                        # 在这插入一下我对有监督无监督这两个的理解
                        # 前面那个无监督聚类，因为并不动用训练集，只利用了测试集，是没有y值的，然后他只有一堆特征
                        # 他利用这些特征去把软件模块进行了聚类，然后利用了那个asfm东东就把软件分成了有缺陷和没有缺陷的预测了（虽然我还没懂原理）
                        # 但是他做到的只是判断有没有缺陷，但是还没有排序
                        # 而且并没有选择按照asfm的大小进行直接的排序
                        # 而是去扫了所有的特征，按照特征倒数都排一次
                        # 然后都算一次性能数据，全存了进去
                        #
                        # 而有监督的聚类
                        # 他有训练集去训练，训练完之后的模型跑测试集的时候会直接得到一个预测值，我可以把这个预测值类比为概率？
                        # 然后按照这个概率直接进行了判断，大于0.5就是预测为有缺陷的，小于就预测为没有缺陷的
                        # 但是他还是进行了一波神秘的计算，把这个作为了后续排序的依据
                        # 而不是扫描所有特征，利用特征值排序


                        datasetfile = trainingfile[:-4] + '_' + testingfile[:-4]
                        df = pd.DataFrame(data=resultlist, columns=header)
                        if not os.path.exists("../Output/{0}".format(datasetfile)):
                            os.makedirs("../Output/{0}".format(datasetfile))
                        df.to_csv("../Output/{0}/{1}.csv".format(datasetfile, fname), index=False)

