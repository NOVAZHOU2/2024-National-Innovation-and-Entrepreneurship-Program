# predict.py
from model import load_model
from train import extract_features


def predict_defects(model, code_snippets):
    features = extract_features(code_snippets)
    predicted_defects = model.predict(features)
    predicted_defects = [max(1, round(defects)) for defects in predicted_defects]  # 确保缺陷个数为正数
    return predicted_defects


def main():
    # 加载模型
    model_file_path = "linear_regression_model.pkl"
    model = load_model(model_file_path)

    # 输入新的代码段列表
    new_code_snippets = [
        "A B C D",
    #"def divide(a, b): return a / b if b != 0 else 0",
     "def read_file(file_path): try: with open(file_path, 'r', encoding='utf-8') as file: data = file.read(); return data except FileNotFoundError: print(f'Error: The file {file_path} was not found.'); return None except IOError: print(f'Error: Could not read the file {file_path}.'); return None def main(): file_path = input('Enter the path to the file: '); file_contents = read_file(file_path); if f"
    ]

    # 预测缺陷个数
    predicted_defects = predict_defects(model, new_code_snippets)

    # 输出预测结果
    for code, defects in zip(new_code_snippets, predicted_defects):
        print(f"Code:\n{code}\nPredicted Defects: {int(defects)}\n")


if __name__ == "__main__":
    main()
