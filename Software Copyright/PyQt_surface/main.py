import os
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QFileDialog, QLabel
from PySide2.QtCore import QFile, QTextStream
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from predict import predict_defects  # 引入预测函数
from model import load_model  # 引入模型加载函数

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'C:\Users\13620\anaconda3\Lib\site-packages\PySide2\plugins\platforms'


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # 设置主布局
        main_layout = QHBoxLayout()

        # 左侧布局
        left_layout = QVBoxLayout()

        # 创建上传按钮
        self.upload_button = QPushButton('上传代码文件')
        self.upload_button.setFixedHeight(50)  # 调整按钮高度
        self.upload_button.clicked.connect(self.upload_files)
        left_layout.addWidget(self.upload_button)

        # 创建左侧文本输入框，用于展示文件内容
        self.input_box = QTextEdit()
        self.input_box.setReadOnly(True)  # 只读模式
        left_layout.addWidget(self.input_box)

        # 创建一个预测按钮
        self.predict_button = QPushButton('预测')
        self.predict_button.setFixedHeight(50)  # 增加按钮的高度
        self.predict_button.clicked.connect(self.on_predict)
        left_layout.addWidget(self.predict_button)

        # 右侧布局，包括文本输出和图表
        right_layout = QVBoxLayout()

        # 右侧输出框
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        right_layout.addWidget(self.output_box)

        # 添加柱状图区域
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        # 添加左侧和右侧布局到主布局
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # 设置左右布局比例
        main_layout.setStretchFactor(left_layout, 3)  # 左侧占60%
        main_layout.setStretchFactor(right_layout, 2)  # 右侧占40%

        # 设置窗口的主布局
        self.setLayout(main_layout)
        self.setWindowTitle('软件缺陷预测与排序系统')

        # 增大初始窗口的大小
        self.resize(1800, 1200)  # 增加初始窗口大小

        # 保存文件内容的列表
        self.file_contents = []

        # 加载模型
        self.model = load_model("linear_regression_model.pkl")

    def upload_files(self):
        # 选择多个文件
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择代码文件", "",
                                                     "代码文件 (*.cpp *.java *.py *.txt);;所有文件 (*)")

        if file_paths:
            self.file_contents.clear()  # 清空之前的文件内容
            combined_text = ""
            for file_path in file_paths:
                try:
                    # 使用 Python 内置文件读取，明确指定 UTF-8 编码
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()

                    # 以文件名作为标识符，展示每个文件内容
                    file_name = os.path.basename(file_path)
                    combined_text += f"文件: {file_name}\n" + content + "\n" + "-" * 40 + "\n\n"

                    # 保存每个文件的内容
                    self.file_contents.append((file_name, content))

                except Exception as e:
                    print(f"文件读取错误: {e}")  # 打印错误信息以便调试

            # 将所有文件的内容显示在输入框中
            self.input_box.setText(combined_text)

    def on_predict(self):
        if not self.file_contents:
            self.output_box.setText("No files uploaded or content is empty")
            return

        # 提取文件名和代码内容
        file_names = [name for name, _ in self.file_contents]
        code_snippets = [content for _, content in self.file_contents]

        # 使用模型进行预测
        predicted_defects = predict_defects(self.model, code_snippets)

        # 创建简短的文件编号（a, b, c, d...），并在括号内显示原文件名
        file_identifiers = [chr(97 + i) for i in range(len(file_names))]  # 使用 a, b, c, d 等作为文件ID
        labeled_file_names = [f"{file_names[i]}" for i in range(len(file_names))]  # 仅显示文件名，不加 "File" 前缀

        # 将文件编号、文件名、预测结果绑定在一起
        results = list(zip(file_identifiers, labeled_file_names, predicted_defects))

        # 按预测的缺陷数量排序
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)

        # 添加提示性说明
        output_text = (
            "Below is the number of defects predicted for each input file using the linear regression model, "
            "sorted in descending order of predicted defect counts:\n\n"
        )

        # 输出排序后的预测结果
        for identifier, file_name, defects in sorted_results:
            output_text += f"{identifier}: {file_name}, Predicted defects: {defects}\n"

        # 将最终的结果显示在输出框中
        self.output_box.setText(output_text)

        # 生成柱状图
        self.plot_defects_bar_chart([x[0] for x in sorted_results], [x[2] for x in sorted_results])

    def plot_defects_bar_chart(self, file_ids, defects):
        # 清除旧的图表
        self.figure.clear()

        # 创建柱状图
        ax = self.figure.add_subplot(111)

        # 使用简短的文件ID a, b, c, d...
        ax.bar(file_ids, defects)
        ax.set_title("Predicted Defects by File ID")
        ax.set_xlabel("File ID")
        ax.set_ylabel("Number of Defects")

        # 刷新画布
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
