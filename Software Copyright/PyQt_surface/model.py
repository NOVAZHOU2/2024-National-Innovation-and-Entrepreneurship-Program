# model.py
import joblib


def save_model(model, file_path):
    """
    将训练好的模型保存到文件。

    参数:
    - model: 训练好的模型
    - file_path: 模型保存的文件路径
    """
    joblib.dump(model, file_path)


def load_model(file_path):
    """
    从文件加载已保存的模型。

    参数:
    - file_path: 模型保存的文件路径

    返回:
    - model: 加载的模型
    """
    return joblib.load(file_path)
