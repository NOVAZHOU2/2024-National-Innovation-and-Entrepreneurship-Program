import os
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget, \
    QTextEdit, QFileDialog, QFrame, QLineEdit, QScrollArea
from PySide2.QtGui import QPalette, QColor, QFont, QLinearGradient, QBrush, QPixmap
from PySide2.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from predict import predict_defects
from model import load_model

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'C:\Users\13620\anaconda3\Lib\site-packages\PySide2\plugins\platforms'


class IntroWindow(QWidget):
    """
    主界面类，用于展示软件的主界面。
    包括背景设置、标题显示、以及三个功能按钮：操作指南、详情介绍和开始体验。
    """

    def __init__(self):
        """
        初始化IntroWindow类，设置窗口的基础属性和UI界面。
        """
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """
        设置主界面的用户界面，包括背景颜色、窗口属性、标题和功能按钮。
        """
        # 设置背景渐变颜色
        palette = QPalette()
        # 设置渐变颜色，从白色到浅蓝色
        gradient = QLinearGradient(0, 0, 1500, 1000)
        gradient.setColorAt(0.0, QColor(255, 255, 255))  # 渐变起点为白色
        gradient.setColorAt(1.0, QColor(173, 216, 230))  # 渐变终点为浅蓝色
        # 将渐变应用到背景
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)

        # 设置窗口标题和尺寸
        self.setWindowTitle('智能软件缺陷预测与排序系统')
        self.resize(1500, 1000)
        self.setFixedSize(1500, 1000)

        # 添加标题标签并居中
        title_label = QLabel('智能软件缺陷预测与排序系统', self)
        title_label.setAlignment(Qt.AlignCenter)
        # 设置标题的字体为楷体并加粗，字号为48
        font = QFont("KaiTi", 48)
        title_label.setFont(font)
        title_label.setStyleSheet("font-weight: bold;")
        # 设置标题标签的大小和位置
        title_label.setFixedSize(1500, 100)
        title_label.move(0, 200)

        # 创建操作指南按钮
        guide_button = QPushButton('操作指南', self)
        # 设置按钮字体为楷体，字号为20
        font = QFont("KaiTi", 20)
        guide_button.setFont(font)
        # 设置按钮的大小
        guide_button.setFixedSize(300, 100)
        # 连接按钮点击事件，点击后显示操作指南界面
        guide_button.clicked.connect(self.show_guide)
        # 设置按钮的位置
        guide_button.move(200, 700)

        # 创建详情介绍按钮
        detail_button = QPushButton('详情介绍', self)
        # 设置详情按钮字体及大小
        font = QFont("KaiTi", 20)
        detail_button.setFont(font)
        detail_button.setFixedSize(300, 100)
        # 连接按钮点击事件，点击后显示详情介绍界面
        detail_button.clicked.connect(self.show_details)
        # 设置按钮的位置
        detail_button.move(600, 700)

        # 创建开始体验按钮
        predict_button = QPushButton('开始体验', self)
        # 设置按钮字体为楷体，字号为20
        font = QFont("KaiTi", 20)
        predict_button.setFont(font)
        # 设置按钮大小
        predict_button.setFixedSize(300, 100)
        # 连接按钮点击事件，点击后显示登录体验界面
        predict_button.clicked.connect(self.show_login)
        # 设置按钮位置
        predict_button.move(1000, 700)

    def show_guide(self):
        """
        打开操作指南界面。
        创建一个GuideWindow类对象并显示
        """
        self.guide_window = GuideWindow()
        self.guide_window.show()

    def show_details(self):
        """
        打开详情介绍界面。
        创建一个DetailsWindow类对象并显示
        """
        self.details_window = DetailsWindow()
        self.details_window.show()

    def show_login(self):
        """
        打开登录体验界面。
        创建一个LoginWindow类对象并显示
        """
        self.login_window = LoginWindow()
        self.login_window.show()


class GuideWindow(QWidget):
    """
    使用手册界面类，用于展示用户操作指南和系统的使用说明。
    包括窗口背景设置、标题显示、以及详细的操作步骤和注意事项。
    """

    def __init__(self):
        """
        初始化GuideWindow类，设置窗口的基本属性和界面布局。
        """
        super().__init__()

        # 设置窗口标题
        self.setWindowTitle("智能软件缺陷预测与排序系统")
        # 设置窗口大小
        self.resize(1500, 1000)
        # 设置窗口背景为蓝白渐变
        palette = QPalette()

        # 创建渐变对象，从浅蓝色到白色，垂直渐变
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(173, 216, 230))  # 起始颜色：浅蓝色
        gradient.setColorAt(1.0, QColor(255, 255, 255))  # 终止颜色：白色
        # 将渐变应用到窗口背景
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)

        # 创建主布局，采用垂直布局
        main_layout = QVBoxLayout()

        # 在顶部和标题之间添加空白区域
        main_layout.addStretch(1)
        # 创建标题标签
        title_label = QLabel("智能软件缺陷预测与排序系统  使用手册")
        # 设置标题的字体为楷体，字号为24，加粗
        title_label.setFont(QFont("KaiTi", 24, QFont.Bold))
        # 设置标题居中对齐
        title_label.setAlignment(Qt.AlignCenter)
        # 将标题添加到主布局中
        main_layout.addWidget(title_label)

        # 在标题和文本内容之间添加空白区域
        main_layout.addStretch(1)
        # 创建内容标签，显示操作步骤
        content_label = QLabel(
            "1.登录界面操作"
            "\n    步骤1: 在登录界面输入您的手机号。"
            "\n    步骤2: 输入对应的密码。"
            "\n    步骤3: 点击“登录 / 注册”按钮以进入系统。"
            "\n2.上传代码文件"
            "\n    步骤1: 成功登录后，您将进入预测界面。"
            "\n    步骤2: 在预测界面上，点击“上传代码”按钮。"
            "\n    步骤3: 选择并上传您需要检测的代码文件。支持的文件格式包括C, C + +, Java, Python等。"
            "\n3.执行代码漏洞检测"
            "\n    步骤1: 在代码文件上传完成后，点击“预测”按钮。"
            "\n    步骤2: 等待几秒钟，系统将自动分析每个文件的代码缺陷。"
            "\n    步骤3: 分析完成后，右侧的输出栏中将显示每个文件预测的缺陷个数。"
            "\n    步骤4: 同时，系统会生成一个柱状图，直观展示各文件的缺陷数量排序。"
            "\n注意事项"
            "\n    确保在上传代码前已正确登录系统。"
            "\n    上传的代码文件应为有效的编程语言文件（如C, C + +, Java, Python等）。"
            "\n    请耐心等待预测过程完成，避免中途关闭或刷新页面。"
        )
        # 设置内容标签的字体为楷体，字号为16
        content_label.setFont(QFont("KaiTi", 16))
        # 允许内容自动换行
        content_label.setWordWrap(True)
        # 设置文本顶部对齐
        content_label.setAlignment(Qt.AlignTop)
        # 为内容添加边框和内陷效果
        content_label.setFrameShape(QFrame.Panel)
        content_label.setFrameShadow(QFrame.Sunken)
        content_label.setLineWidth(2)  # 设置边框线宽
        content_label.setMargin(10)  # 设置内容的边距
        # 创建内容布局并添加内容标签
        content_layout = QHBoxLayout()
        content_layout.addWidget(content_label)
        # 设置内容的外边距
        content_layout.setContentsMargins(20, 20, 20, 20)
        # 将内容布局添加到主布局
        main_layout.addLayout(content_layout)
        # 在内容和窗口底部添加空白区域
        main_layout.addStretch(1)

        # 将主布局设置为窗口的布局
        self.setLayout(main_layout)


class DetailsWindow(QWidget):
    """
    详情介绍界面类，用于展示详细的项目介绍和说明。
    包括窗口背景设置、标题、以及通过滚动区域显示的详细文本内容。
    """

    def __init__(self):
        """
        初始化 DetailsWindow 类，设置窗口的基本属性、背景、标题和详细内容显示区域。
        """
        super().__init__()

        # 设置窗口标题
        self.setWindowTitle("智能软件缺陷预测与排序系统")
        # 设置窗口大小
        self.resize(1500, 1000)
        # 设置窗口背景为蓝白渐变
        palette = QPalette()
        # 创建渐变对象，从浅蓝色到白色，垂直渐变
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(173, 216, 230))  # 起始颜色：浅蓝色
        gradient.setColorAt(1.0, QColor(255, 255, 255))  # 终止颜色：白色
        # 将渐变应用到窗口背景
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)
        # 设置窗口为固定大小
        self.setFixedSize(1500, 1000)

        # 添加标题标签，并通过绝对布局设置位置和大小
        self.title_label = QLabel('智能软件缺陷预测与排序系统：基于深度学习的软件缺陷分析师', self)
        # 设置标题的字体为楷体，字号为24，加粗
        self.title_label.setFont(QFont("KaiTi", 24, QFont.Bold))
        # 设置标题居中对齐
        self.title_label.setAlignment(Qt.AlignCenter)
        # 设置标题的位置和大小
        self.title_label.setGeometry(50, 100, 1400, 50)

        # 创建滚动区域，用于显示超出可视范围的内容
        self.scroll_area = QScrollArea(self)
        # 设置滚动区域的位置和大小
        self.scroll_area.setGeometry(100, 200, 1300, 700)
        # 允许滚动区域内的内容大小根据窗口自动调整
        self.scroll_area.setWidgetResizable(True)
        # 创建内容显示的 QWidget，用于容纳所有文本
        content_widget = QWidget()
        # 设置内容区域的大小
        content_widget.setFixedSize(1280, 1500)
        # 创建 QTextEdit，用于显示项目的详细内容
        content_text = QTextEdit(content_widget)
        # 设置文本框为只读模式，避免用户修改内容
        content_text.setReadOnly(True)
        # 设置文本框的大小和位置
        content_text.setGeometry(0, 0, 1280, 1500)
        # 设置文本框字体为楷体，字号为16
        content_text.setFont(QFont("KaiTi", 16))
        # 设置文本框的边框样式和内陷效果
        content_text.setFrameShape(QFrame.Panel)
        content_text.setFrameShadow(QFrame.Sunken)
        # 设置边框线宽
        content_text.setLineWidth(2)
        # 添加显示的内容文本
        content_text.setText(
            "    软件缺陷是指在软件开发过程中或软件已经投入使用后发现的，与预期功能或性"
            "能不符的问题或错误，并且导致软件不能按照预期的方式运行或不能满足用户需求。"
            "已有的软件缺陷预测算法无法充分提取代码的语义信息，并且排序过程中并未考虑到"
            "代价敏感问题。因此，本项目旨在采用预训练模型及基于代价敏感学习的缺陷排序算"
            "法来解决上述问题，以对软件模块进行合理的排序，从而提升软件质量保障的效率和效果。"
            "\n    针对已有方法在对软件模块进行缺陷预测时代码语义理解不充分和未考虑对存在"
            "缺陷的软件模块测试优先级的困境，尝试以代码预训练模型，对代码的多语义特征进行"
            "提取。并基于混合注意力机制的特征学习方法，同时利用单头注意力编码器和多头注意"
            "力编码器对代码 token 进行编码，提高软件缺陷预测模型的性能。且基于代价敏感学习"
            "的学习排序方法，修改损失函数，让软件模块按照缺陷个数，缺陷严重程度更大或者缺"
            "陷密度更高进行正确排序，有效解决当前现有软件缺陷预测方法技术准确率低以及未考"
            "虑对软件模块的测试优先级的问题。"
            "\n    本项目是在现有研究成果的基础上，考虑到软件缺陷会对用户造成巨额经济损失的这一痛点问题，立志在理论、技术和方法上创新，提出对软件缺陷排序方法的研究，其"
            "特色和创新之处在于：本项目提出了基于 CodeBERT 的软件代码多语义提取方法、基"
            "于混合注意力机制的特征学习方法、基于代价敏感学习的学习排序方法。提出的基于"
            "CodeBERT 的软件代码多语义提取方法，首先将软件代码表示为抽象语法树，然后利用"
            "CodeBERT 从抽象语法树中提取代码词汇、语法和结构信息，实现了对软件代码多语义"
            "信息的提取，为后续缺陷预测模型的建立提供了缺陷代码信息基础。"
            "\n    本项目提出了一种"
            "基于混合注意力机制的特征学习方法，为了对代码 token 进行编码，团队结合了基于"
            "Bi-GRU 的单头注意力编码器和基于 Transformer 的多头注意力编码器。后续通过捕获"
            "代码 token 之间的依赖关系，提高了软件缺陷预测模型的性能，使得并能通过混合注意"
            "力机制对代码 token 赋予的不同权重解释预测结果。"
            "\n    通过代价敏感学习算法，可以计算"
            "错误排序的代价并相应地修改学习排序算法的损失函数。这样能够更好地排序缺陷严重"
            "程度更大、缺陷个数更多或缺陷密度更高的软件模块，从而减少相关问题的代价，有效"
            "地提升了软件测试的效率和质量。"
            "\n    这些 CodeBERT、注意力机制、代价敏感学习、学习"
            "排序技术的深度运用，既体现了大规模软件代码环境下缺陷预测研究的独特技术特征，"
            "也为今后这些技术在软件理论技术研究中的常态化运用作出了示范。"
        )

        # 将内容 widget 设置为滚动区域的 widget
        self.scroll_area.setWidget(content_widget)


class PredictWindow(QWidget):
    """
    预测窗口类，用于用户上传代码文件、执行缺陷预测并显示结果。
    包括文件上传、预测按钮、显示输入代码和预测结果的文本框，以及缺陷数量的柱状图。
    """

    def __init__(self):
        """
        初始化 PredictWindow 类，设置主界面的布局和组件。
        """
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """
        初始化主界面的布局和用户界面，包括文件上传按钮、预测按钮、文本框和柱状图显示区域。
        """
        # 设置主布局为水平布局，左侧为文件操作，右侧为结果显示
        main_layout = QHBoxLayout()
        # 左侧布局，包含文件上传和输入框
        left_layout = QVBoxLayout()

        # 创建上传按钮，用于上传代码文件
        self.upload_button = QPushButton('上传代码文件')
        self.upload_button.setFixedHeight(50)  # 设置按钮高度
        self.upload_button.clicked.connect(self.upload_files)  # 连接文件上传事件
        left_layout.addWidget(self.upload_button)

        # 创建文本框，用于显示上传的文件内容
        self.input_box = QTextEdit()
        self.input_box.setReadOnly(True)  # 设置为只读模式
        left_layout.addWidget(self.input_box)

        # 创建预测按钮，用于执行代码缺陷预测
        self.predict_button = QPushButton('预测')
        self.predict_button.setFixedHeight(50)  # 设置按钮高度
        self.predict_button.clicked.connect(self.on_predict)  # 连接预测事件
        left_layout.addWidget(self.predict_button)

        # 右侧布局，包含结果文本框和柱状图显示区域
        right_layout = QVBoxLayout()

        # 创建文本框，用于显示预测结果
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)  # 设置为只读模式
        right_layout.addWidget(self.output_box)

        # 创建柱状图区域，用于显示缺陷预测结果
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        # 将左侧和右侧布局添加到主布局中
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # 设置左右布局的比例
        main_layout.setStretchFactor(left_layout, 3)  # 左侧占60%
        main_layout.setStretchFactor(right_layout, 2)  # 右侧占40%

        # 设置窗口主布局
        self.setLayout(main_layout)
        self.setWindowTitle('智能软件缺陷预测与排序系统')
        # 设置窗口大小
        self.resize(1800, 1200)

        # 初始化文件内容的列表，用于存储上传的文件内容
        self.file_contents = []

        # 加载缺陷预测模型
        self.model = load_model("linear_regression_model.pkl")

    def upload_files(self):
        """
        文件上传功能，允许用户选择多个代码文件并展示其内容。
        """
        # 打开文件对话框，选择代码文件
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择代码文件", "",
                                                     "代码文件 (*.cpp *.java *.py *.txt);;所有文件 (*)")

        if file_paths:
            # 清空之前的文件内容
            self.file_contents.clear()
            combined_text = ""

            # 逐个读取文件内容并展示
            for file_path in file_paths:
                try:
                    # 打开文件并读取内容，使用 UTF-8 编码
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()

                    # 获取文件名并展示文件内容
                    file_name = os.path.basename(file_path)
                    combined_text += f"文件: {file_name}\n" + content + "\n" + "-" * 40 + "\n\n"

                    # 保存文件名和内容到文件内容列表中
                    self.file_contents.append((file_name, content))

                except Exception as e:
                    # 如果文件读取失败，输出错误信息
                    print(f"文件读取错误: {e}")

            # 将读取到的文件内容显示在输入框中
            self.input_box.setText(combined_text)

    def on_predict(self):
        """
        预测功能，使用加载的模型对上传的代码文件进行缺陷预测，并显示结果。
        """
        # 如果没有上传文件，则显示提示
        if not self.file_contents:
            self.output_box.setText("No files uploaded or content is empty")
            return

        # 提取文件名和文件内容
        file_names = [name for name, _ in self.file_contents]
        code_snippets = [content for _, content in self.file_contents]

        # 使用模型进行缺陷预测
        predicted_defects = predict_defects(self.model, code_snippets)

        # 生成文件编号（a, b, c, d...）
        file_identifiers = [chr(97 + i) for i in range(len(file_names))]
        # 生成文件名和预测结果的组合
        labeled_file_names = [f"{file_names[i]}" for i in range(len(file_names))]
        results = list(zip(file_identifiers, labeled_file_names, predicted_defects))
        # 按照预测的缺陷数量降序排列
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
        # 构建结果文本
        output_text = (
            "Below is the number of defects predicted for each input file using the linear regression model, "
            "sorted in descending order of predicted defect counts:\n\n"
        )

        # 输出排序后的预测结果
        for identifier, file_name, defects in sorted_results:
            output_text += f"{identifier}: {file_name}, Predicted defects: {defects}\n"
        # 将结果显示在输出框中
        self.output_box.setText(output_text)
        # 绘制柱状图显示预测结果
        self.plot_defects_bar_chart([x[0] for x in sorted_results], [x[2] for x in sorted_results])

    def plot_defects_bar_chart(self, file_ids, defects):
        """
        绘制柱状图，展示预测的缺陷数量。
        """
        # 清除旧的图表内容
        self.figure.clear()

        # 创建新的柱状图
        ax = self.figure.add_subplot(111)
        ax.bar(file_ids, defects)
        # 设置图表标题和坐标轴标签
        ax.set_title("Predicted Defects by File ID")
        ax.set_xlabel("File ID")
        ax.set_ylabel("Number of Defects")
        # 刷新画布，显示新图表
        self.canvas.draw()


class LoginWindow(QWidget):
    """
    登录窗口类，提供用户登录功能。
    包括手机号输入框、密码输入框以及登录/注册按钮。
    """

    def __init__(self):
        """
        初始化 LoginWindow 类，设置窗口基本属性和用户界面布局。
        """
        super().__init__()

        # 设置窗口标题
        self.setWindowTitle("智能软件缺陷预测与排序系统")

        # 设置窗口的固定大小为1200x800
        self.setFixedSize(1200, 800)

        # 设置背景为渐变颜色，起点为白色，终点为浅蓝色
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 1200, 800)
        # 起点颜色：白色
        gradient.setColorAt(0.0, QColor(255, 255, 255))
        # 终点颜色：浅蓝色
        gradient.setColorAt(1.0, QColor(173, 216, 230))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)

        # 使用绝对布局，通过 move() 方法设置部件的位置

        # 添加左上角的Logo
        logo_label = QLabel(self)
        # 加载并缩放Logo图片
        pixmap = QPixmap("logo.png")
        # 按比例缩放图片
        scaled_pixmap = pixmap.scaled(450, 450, Qt.KeepAspectRatio)
        logo_label.setPixmap(scaled_pixmap)
        # 设置Logo的位置
        logo_label.move(100, 100)

        # 创建手机号标签，并设置字体
        phone_label = QLabel("手机号", self)
        # 楷体字体，字号为20
        phone_label.setFont(QFont("KaiTi", 20, QFont.Bold))
        # 设置手机号标签的位置
        phone_label.move(600, 300)

        # 创建手机号输入框，并设置字体和大小
        self.phone_input = QLineEdit(self)
        # 楷体字体，字号为17
        font = QFont("KaiTi", 17)
        self.phone_input.setFont(font)
        # 设置输入框高度
        self.phone_input.setFixedHeight(50)
        # 设置输入框宽度
        self.phone_input.setFixedWidth(350)
        # 设置手机号输入框的位置
        self.phone_input.move(750, 300)

        # 创建密码标签，并设置字体
        password_label = QLabel("密码", self)
        # 楷体字体，字号为20
        password_label.setFont(QFont("KaiTi", 20, QFont.Bold))
        # 设置密码标签的位置
        password_label.move(600, 400)

        # 创建密码输入框，并设置字体和大小
        self.password_input = QLineEdit(self)
        # 楷体字体，字号为16
        font = QFont("KaiTi", 16)
        self.password_input.setFont(font)
        # 设置输入框高度
        self.password_input.setFixedHeight(50)
        # 设置输入框宽度
        self.password_input.setFixedWidth(350)
        # 设置密码输入框为密码模式，隐藏输入字符
        self.password_input.setEchoMode(QLineEdit.Password)
        # 设置密码输入框的位置
        self.password_input.move(750, 400)

        # 创建登录/注册按钮，并设置样式
        login_button = QPushButton("登录/注册", self)
        # 楷体字体，字号为20
        login_button.setFont(QFont("KaiTi", 20, QFont.Bold))
        # 设置按钮的固定大小
        login_button.setFixedSize(250, 50)
        # 设置按钮的背景颜色、文本颜色和圆角效果
        login_button.setStyleSheet("background-color: #1E90FF; color: white; border-radius: 10px;")
        # 设置按钮的位置
        login_button.move(600, 500)
        # 连接按钮的点击事件，点击按钮后显示预测界面
        login_button.clicked.connect(self.show_predict)

    def show_predict(self):
        """
        打开预测界面。
        """
        self.predict_window = PredictWindow()
        self.predict_window.show()


if __name__ == '__main__':
    app = QApplication([])
    intro_window = IntroWindow()
    intro_window.show()
    app.exec_()
