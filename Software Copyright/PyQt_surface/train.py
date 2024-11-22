import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from model import save_model
import numpy as np


def read_data(file_path):
    """
    读取CSV文件中的数据，将代码片段和对应的缺陷数量作为输出返回。

    参数:
    file_path (str): 文件路径

    返回:
    tuple: 包含代码片段的列表和缺陷数量的列表
    """
    data = pd.read_csv(file_path)
    code_snippets = data['code'].tolist()
    defect_counts = data['defects'].tolist()
    return code_snippets, defect_counts


def extract_features(code_snippets):
    """
    使用CodeBERT模型提取代码片段的特征。每个代码片段被转化为嵌入向量。

    参数:
    code_snippets (list): 包含代码片段的列表

    返回:
    list: 包含每个代码片段的特征向量
    """
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    features = []

    # 遍历每个代码片段，提取其特征向量
    for code in code_snippets:
        inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        # 获取最后一层隐藏状态的均值作为特征
        features.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    return features


def train_model(features, defect_counts):
    """
    训练线性回归模型，根据代码特征预测缺陷数量。使用对数变换处理目标变量，确保模型输出正值。

    参数:
    features (list): 包含代码特征的列表
    defect_counts (list): 包含代码缺陷数量的列表

    返回:
    tuple: 训练后的模型，测试集特征，测试集真实值，测试集预测值
    """
    # 对缺陷数量进行对数变换，避免对数计算中的零问题
    log_defects = np.log(np.array(defect_counts) + 1)

    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, log_defects, test_size=0.2, random_state=42)

    # 初始化线性回归模型并进行训练
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 进行预测，并对预测值进行反变换
    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log) - 1

    return model, X_test, y_test, y_pred


def evaluate_model(y_test, y_pred):
    """
    评估模型性能，计算测试集的均方误差 (MSE)。

    参数:
    y_test (list): 测试集的真实值
    y_pred (list): 测试集的预测值

    返回:
    float: 均方误差 (MSE)
    """
    from sklearn.metrics import mean_squared_error

    # 计算MSE，首先对测试集的真实值进行反变换
    mse = mean_squared_error(np.exp(y_test) - 1, y_pred)
    print(f"Mean Squared Error: {mse}")
    return mse


def main():
    """
    主函数，负责读取数据、提取特征、训练模型、评估模型并保存模型。
    """
    # 读取数据
    file_path = "../File/Code.csv"
    code_snippets, defect_counts = read_data(file_path)

    # 提取代码特征
    features = extract_features(code_snippets)

    # 训练模型并返回测试集和预测结果
    model, X_test, y_test, y_pred = train_model(features, defect_counts)

    # 评估模型性能
    evaluate_model(y_test, y_pred)

    # 保存训练好的模型
    model_file_path = "linear_regression_model.pkl"
    save_model(model, model_file_path)
    print(f"Model saved to {model_file_path}")


if __name__ == "__main__":
    main()
