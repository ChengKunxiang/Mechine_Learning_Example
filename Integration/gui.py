import sys
import os
import pandas as pd
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QLabel, QFileDialog, QWidget,
                             QGroupBox, QProgressBar, QMessageBox, QTabWidget,
                             QComboBox, QCheckBox, QLineEdit, QFormLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QMutex
from PyQt5.QtGui import QFont, QTextCursor

# 导入您的训练和预测模块
try:
    from train import MultiTargetModelComparisonSystem
    from manuafacture import MultiTargetModelPredictor
except ImportError as e:
    print(f"导入模块错误: {e}")


# 创建一个线程安全的输出重定向器
class ThreadSafeStream:
    def __init__(self, callback):
        self.callback = callback
        self.mutex = QMutex()

    def write(self, text):
        if text.strip():  # 只发送非空文本
            self.mutex.lock()
            try:
                self.callback(text)
            finally:
                self.mutex.unlock()

    def flush(self):
        pass


class TrainingThread(QThread):
    """训练线程，避免界面卡死"""
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool)

    def __init__(self, data_path, target_columns, output_dir=None):
        super().__init__()
        self.data_path = data_path
        self.target_columns = target_columns
        self.output_dir = output_dir
        self.system = None
        self._is_running = True

    def run(self):
        try:
            self.update_signal.emit("开始初始化训练系统...")

            # 创建训练系统实例
            self.system = MultiTargetModelComparisonSystem(self.output_dir)

            # 重定向标准输出到我们的信号
            import sys
            original_stdout = sys.stdout
            sys.stdout = ThreadSafeStream(self.update_signal.emit)

            try:
                # 运行全面分析
                self.update_signal.emit("开始加载和预处理数据...")
                results = self.system.run_comprehensive_analysis(self.data_path, self.target_columns)

                self.update_signal.emit("训练完成！")
                self.finished_signal.emit(True)
            finally:
                # 恢复标准输出
                sys.stdout = original_stdout

        except Exception as e:
            error_msg = f"训练过程中出现错误: {str(e)}\n{traceback.format_exc()}"
            self.update_signal.emit(error_msg)
            self.finished_signal.emit(False)

    def stop(self):
        """停止训练"""
        self._is_running = False
        self.terminate()  # 强制终止线程


class PredictionThread(QThread):
    """预测线程"""
    update_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)

    def __init__(self, model_dir, data_path, include_original=True, include_probabilities=True, output_format='csv'):
        super().__init__()
        self.model_dir = model_dir
        self.data_path = data_path
        self.include_original = include_original
        self.include_probabilities = include_probabilities
        self.output_format = output_format
        self.predictor = None
        self._is_running = True

    def run(self):
        try:
            self.update_signal.emit("开始加载预测器...")

            # 重定向标准输出到我们的信号
            import sys
            original_stdout = sys.stdout
            sys.stdout = ThreadSafeStream(self.update_signal.emit)

            try:
                # 创建预测器实例
                self.predictor = MultiTargetModelPredictor(self.model_dir)
                self.update_signal.emit("预测器加载成功！")

                # 进行预测
                predictions = self.predictor.predict(
                    self.data_path,
                    include_original_data=self.include_original,
                    include_probabilities=self.include_probabilities,
                    output_format=self.output_format
                )

                result_text = f"预测完成！\n"
                result_text += f"输出目录: {self.predictor.output_dir}\n"
                result_text += f"预测结果形状: {predictions.shape}\n"
                result_text += f"前5行预览:\n{predictions.head().to_string()}"

                self.result_signal.emit(result_text)
                self.finished_signal.emit(True)
            finally:
                # 恢复标准输出
                sys.stdout = original_stdout

        except Exception as e:
            error_msg = f"预测过程中出现错误: {str(e)}\n{traceback.format_exc()}"
            self.update_signal.emit(error_msg)
            self.finished_signal.emit(False)

    def stop(self):
        """停止预测"""
        self._is_running = False
        self.terminate()  # 强制终止线程


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.prediction_thread = None
        self.current_model_dir = "output/models"  # 默认模型目录
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('多目标变量机器学习系统')
        self.setGeometry(100, 100, 1200, 800)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建标签页
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        # 训练标签页
        train_tab = QWidget()
        train_layout = QVBoxLayout(train_tab)
        tab_widget.addTab(train_tab, "模型训练")

        # 预测标签页
        predict_tab = QWidget()
        predict_layout = QVBoxLayout(predict_tab)
        tab_widget.addTab(predict_tab, "模型预测")

        # 初始化两个标签页的UI
        self.init_train_tab(train_layout)
        self.init_predict_tab(predict_layout)

    def init_train_tab(self, layout):
        """初始化训练标签页"""
        # 训练设置组
        train_group = QGroupBox("训练设置")
        train_group_layout = QVBoxLayout(train_group)

        # 训练数据选择
        train_data_layout = QHBoxLayout()
        self.train_path_label = QLabel("未选择训练数据")
        self.train_path_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        self.train_path_label.setMinimumHeight(30)
        btn_train_browse = QPushButton("选择训练数据")
        btn_train_browse.clicked.connect(self.select_train_data)
        train_data_layout.addWidget(QLabel("训练数据路径:"))
        train_data_layout.addWidget(self.train_path_label, 1)
        train_data_layout.addWidget(btn_train_browse)

        # 目标变量设置
        target_layout = QHBoxLayout()
        self.target_input = QLineEdit()
        self.target_input.setPlaceholderText("请输入目标变量列名，用逗号分隔，如: 升力系数,阻力系数,俯仰力矩系数")
        self.target_input.textChanged.connect(self.check_train_ready)
        target_layout.addWidget(QLabel("目标变量:"))
        target_layout.addWidget(self.target_input, 1)

        # 输出目录设置
        output_layout = QHBoxLayout()
        self.output_dir_label = QLabel("./output")
        self.output_dir_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        btn_output_browse = QPushButton("选择输出目录")
        btn_output_browse.clicked.connect(self.select_output_dir)
        output_layout.addWidget(QLabel("输出目录:"))
        output_layout.addWidget(self.output_dir_label, 1)
        output_layout.addWidget(btn_output_browse)

        # 训练按钮和进度条
        train_control_layout = QHBoxLayout()
        self.btn_start_train = QPushButton("开始训练")
        self.btn_start_train.clicked.connect(self.start_training)
        self.btn_start_train.setEnabled(False)
        self.btn_stop_train = QPushButton("停止训练")
        self.btn_stop_train.clicked.connect(self.stop_training)
        self.btn_stop_train.setEnabled(False)  # 初始时禁用停止按钮
        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        train_control_layout.addWidget(self.btn_start_train)
        train_control_layout.addWidget(self.btn_stop_train)
        train_control_layout.addWidget(self.train_progress, 1)

        train_group_layout.addLayout(train_data_layout)
        train_group_layout.addLayout(target_layout)
        train_group_layout.addLayout(output_layout)
        train_group_layout.addLayout(train_control_layout)

        # 输出文本框
        output_group = QGroupBox("训练输出")
        output_layout = QVBoxLayout(output_group)
        self.train_output = QTextEdit()
        self.train_output.setReadOnly(True)
        self.train_output.setFont(QFont("Consolas", 9))
        output_layout.addWidget(self.train_output)

        # 添加到主布局
        layout.addWidget(train_group)
        layout.addWidget(output_group, 1)

        # 初始检查按钮状态
        self.check_train_ready()

    def init_predict_tab(self, layout):
        """初始化预测标签页"""
        # 预测设置组
        predict_group = QGroupBox("预测设置")
        predict_form_layout = QFormLayout(predict_group)

        # 模型目录选择
        model_layout = QHBoxLayout()
        self.model_dir_label = QLabel("output/models")
        self.model_dir_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        btn_model_browse = QPushButton("选择模型目录")
        btn_model_browse.clicked.connect(self.select_model_dir)
        model_layout.addWidget(self.model_dir_label, 1)
        model_layout.addWidget(btn_model_browse)
        predict_form_layout.addRow("模型目录:", model_layout)

        # 预测数据选择
        predict_data_layout = QHBoxLayout()
        self.predict_path_label = QLabel("未选择预测数据")
        self.predict_path_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        btn_predict_browse = QPushButton("选择预测数据")
        btn_predict_browse.clicked.connect(self.select_predict_data)
        predict_data_layout.addWidget(self.predict_path_label, 1)
        predict_data_layout.addWidget(btn_predict_browse)
        predict_form_layout.addRow("预测数据路径:", predict_data_layout)

        # 预测选项
        self.include_original_check = QCheckBox("包含原始数据")
        self.include_original_check.setChecked(True)
        self.include_prob_check = QCheckBox("包含预测概率")
        self.include_prob_check.setChecked(True)

        options_layout = QHBoxLayout()
        options_layout.addWidget(self.include_original_check)
        options_layout.addWidget(self.include_prob_check)
        predict_form_layout.addRow("预测选项:", options_layout)

        # 输出格式选择
        self.format_combo = QComboBox()
        self.format_combo.addItems(["CSV", "Excel"])
        predict_form_layout.addRow("输出格式:", self.format_combo)

        # 预测按钮
        predict_btn_layout = QHBoxLayout()
        self.btn_start_predict = QPushButton("开始预测")
        self.btn_start_predict.clicked.connect(self.start_prediction)
        self.btn_start_predict.setEnabled(False)
        self.btn_stop_predict = QPushButton("停止预测")
        self.btn_stop_predict.clicked.connect(self.stop_prediction)
        self.btn_stop_predict.setEnabled(False)  # 初始时禁用停止按钮
        predict_btn_layout.addWidget(self.btn_start_predict)
        predict_btn_layout.addWidget(self.btn_stop_predict)
        predict_form_layout.addRow("", predict_btn_layout)

        # 输出文本框
        output_group = QGroupBox("预测输出")
        output_layout = QVBoxLayout(output_group)
        self.predict_output = QTextEdit()
        self.predict_output.setReadOnly(True)
        self.predict_output.setFont(QFont("Consolas", 9))
        output_layout.addWidget(self.predict_output)

        # 结果展示
        result_group = QGroupBox("预测结果")
        result_layout = QVBoxLayout(result_group)
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        self.result_output.setFont(QFont("Consolas", 9))
        result_layout.addWidget(self.result_output)

        # 垂直布局
        layout.addWidget(predict_group)
        layout.addWidget(output_group, 1)
        layout.addWidget(result_group, 1)

        # 初始检查按钮状态
        self.check_predict_ready()

    def select_train_data(self):
        """选择训练数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择训练数据文件", "", "CSV文件 (*.csv);;所有文件 (*)"
        )
        if file_path:
            self.train_path_label.setText(file_path)
            self.check_train_ready()

            # 尝试自动识别目标变量
            try:
                df = pd.read_csv(file_path, nrows=5)  # 只读取前5行来查看列名
                columns = df.columns.tolist()
                self.train_output.append(f"数据列名: {', '.join(columns)}")

                # 如果目标变量输入框为空，自动填充示例
                if not self.target_input.text().strip():
                    # 尝试猜测目标变量（取后几列作为示例）
                    example_targets = columns[-3:] if len(columns) >= 3 else columns
                    self.target_input.setText(", ".join(example_targets))
                    self.train_output.append(f"已自动填充目标变量示例: {', '.join(example_targets)}")
                    self.train_output.append("请根据您的需求修改目标变量列名")

            except Exception as e:
                self.train_output.append(f"读取数据列名时出错: {e}")

    def select_output_dir(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir_label.setText(dir_path)

    def select_model_dir(self):
        """选择模型目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if dir_path:
            self.model_dir_label.setText(dir_path)
            self.current_model_dir = dir_path
            self.check_predict_ready()

    def select_predict_data(self):
        """选择预测数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择预测数据文件", "", "CSV文件 (*.csv);;所有文件 (*)"
        )
        if file_path:
            self.predict_path_label.setText(file_path)
            self.check_predict_ready()

    def check_train_ready(self):
        """检查训练是否就绪"""
        has_data = self.train_path_label.text() != "未选择训练数据"
        has_targets = bool(self.target_input.text().strip())

        # 检查数据文件是否存在
        if has_data:
            file_path = self.train_path_label.text()
            if not os.path.exists(file_path):
                self.train_output.append(f"警告: 文件不存在: {file_path}")
                has_data = False
            else:
                # 检查文件是否是CSV格式
                try:
                    pd.read_csv(file_path, nrows=1)
                except Exception as e:
                    self.train_output.append(f"警告: 文件不是有效的CSV格式: {e}")
                    has_data = False

        # 检查是否有线程在运行 - 修复NoneType错误
        is_running = False
        if self.training_thread is not None:
            is_running = self.training_thread.isRunning()

        self.btn_start_train.setEnabled(has_data and has_targets and not is_running)
        self.btn_stop_train.setEnabled(is_running)

        # 提供状态反馈
        if is_running:
            self.train_output.append("状态: 训练正在进行中...")
        elif not has_data:
            self.train_output.append("状态: 请选择训练数据文件")
        elif not has_targets:
            self.train_output.append("状态: 请输入目标变量列名")
        else:
            self.train_output.append("状态: 可以开始训练")

    def check_predict_ready(self):
        """检查预测是否就绪"""
        model_dir = self.model_dir_label.text()
        has_model = os.path.exists(model_dir) and os.path.isdir(model_dir)
        has_data = self.predict_path_label.text() != "未选择预测数据"

        # 检查预测数据文件是否存在
        if has_data:
            file_path = self.predict_path_label.text()
            if not os.path.exists(file_path):
                self.predict_output.append(f"警告: 文件不存在: {file_path}")
                has_data = False

        # 检查是否有线程在运行 - 修复NoneType错误
        is_running = False
        if self.prediction_thread is not None:
            is_running = self.prediction_thread.isRunning()

        self.btn_start_predict.setEnabled(has_model and has_data and not is_running)
        self.btn_stop_predict.setEnabled(is_running)

    def start_training(self):
        """开始训练"""
        if self.training_thread is not None and self.training_thread.isRunning():
            QMessageBox.warning(self, "警告", "训练正在进行中，请等待完成！")
            return

        # 获取目标变量
        target_text = self.target_input.text().strip()
        target_columns = [col.strip() for col in target_text.split(',') if col.strip()]

        if not target_columns:
            QMessageBox.warning(self, "警告", "请输入有效的目标变量！")
            return

        # 检查数据文件是否存在
        data_path = self.train_path_label.text()
        if not os.path.exists(data_path):
            QMessageBox.critical(self, "错误", f"数据文件不存在: {data_path}")
            return

        # 清空输出
        self.train_output.clear()

        # 禁用开始按钮，启用停止按钮，显示进度条
        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.train_progress.setVisible(True)
        self.train_progress.setRange(0, 0)  # 无限进度条

        # 创建并启动训练线程
        output_dir = self.output_dir_label.text() if self.output_dir_label.text() != "./output" else None
        self.training_thread = TrainingThread(
            data_path,
            target_columns,
            output_dir
        )
        self.training_thread.update_signal.connect(self.update_train_output)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()

        self.train_output.append("开始训练...")

    def stop_training(self):
        """停止训练"""
        if self.training_thread is not None and self.training_thread.isRunning():
            reply = QMessageBox.question(self, "确认停止",
                                         "确定要停止训练吗？",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.training_thread.stop()
                self.training_thread.wait()  # 等待线程完全停止
                self.train_output.append("训练已停止")
                self.training_finished(False)

    def start_prediction(self):
        """开始预测"""
        if self.prediction_thread is not None and self.prediction_thread.isRunning():
            QMessageBox.warning(self, "警告", "预测正在进行中，请等待完成！")
            return

        # 检查模型目录是否存在
        model_dir = self.model_dir_label.text()
        if not os.path.exists(model_dir):
            QMessageBox.critical(self, "错误", f"模型目录不存在: {model_dir}")
            return

        # 检查预测数据文件是否存在
        data_path = self.predict_path_label.text()
        if not os.path.exists(data_path):
            QMessageBox.critical(self, "错误", f"预测数据文件不存在: {data_path}")
            return

        # 清空输出
        self.predict_output.clear()
        self.result_output.clear()

        # 禁用开始按钮，启用停止按钮
        self.btn_start_predict.setEnabled(False)
        self.btn_stop_predict.setEnabled(True)

        # 创建并启动预测线程
        self.prediction_thread = PredictionThread(
            model_dir,
            data_path,
            self.include_original_check.isChecked(),
            self.include_prob_check.isChecked(),
            self.format_combo.currentText().lower()
        )
        self.prediction_thread.update_signal.connect(self.update_predict_output)
        self.prediction_thread.result_signal.connect(self.update_prediction_result)
        self.prediction_thread.finished_signal.connect(self.prediction_finished)
        self.prediction_thread.start()

        self.predict_output.append("开始预测...")

    def stop_prediction(self):
        """停止预测"""
        if self.prediction_thread is not None and self.prediction_thread.isRunning():
            reply = QMessageBox.question(self, "确认停止",
                                         "确定要停止预测吗？",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.prediction_thread.stop()
                self.prediction_thread.wait()  # 等待线程完全停止
                self.predict_output.append("预测已停止")
                self.prediction_finished(False)

    def update_train_output(self, text):
        """更新训练输出"""
        # 使用QApplication.processEvents()确保GUI保持响应
        QApplication.processEvents()

        self.train_output.moveCursor(QTextCursor.End)
        self.train_output.insertPlainText(text)
        self.train_output.ensureCursorVisible()

    def update_predict_output(self, text):
        """更新预测输出"""
        # 使用QApplication.processEvents()确保GUI保持响应
        QApplication.processEvents()

        self.predict_output.moveCursor(QTextCursor.End)
        self.predict_output.insertPlainText(text)
        self.predict_output.ensureCursorVisible()

    def update_prediction_result(self, text):
        """更新预测结果"""
        self.result_output.setPlainText(text)

    def training_finished(self, success):
        """训练完成"""
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.train_progress.setVisible(False)

        if success:
            self.train_output.append("\n训练完成！")
            # 更新模型目录为最新训练的模型
            model_dir = os.path.join(self.output_dir_label.text(), "models")
            if os.path.exists(model_dir):
                self.model_dir_label.setText(model_dir)
                self.current_model_dir = model_dir
                self.check_predict_ready()
        else:
            self.train_output.append("\n训练失败或已停止！")

        # 更新按钮状态
        self.check_train_ready()

    def prediction_finished(self, success):
        """预测完成"""
        self.btn_start_predict.setEnabled(True)
        self.btn_stop_predict.setEnabled(False)

        if success:
            self.predict_output.append("\n预测完成！")
        else:
            self.predict_output.append("\n预测失败或已停止！")

        # 更新按钮状态
        self.check_predict_ready()

    def closeEvent(self, event):
        """关闭事件，确保线程安全退出"""
        # 检查是否有线程在运行
        threads_running = False

        if self.training_thread is not None and self.training_thread.isRunning():
            threads_running = True
            reply = QMessageBox.question(self, "确认退出",
                                         "训练正在进行中，确定要退出吗？",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.training_thread.stop()
                self.training_thread.wait(5000)  # 等待5秒
            else:
                event.ignore()
                return

        if self.prediction_thread is not None and self.prediction_thread.isRunning():
            threads_running = True
            reply = QMessageBox.question(self, "确认退出",
                                         "预测正在进行中，确定要退出吗？",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.prediction_thread.stop()
                self.prediction_thread.wait(5000)  # 等待5秒
            else:
                event.ignore()
                return

        # 如果没有线程运行或用户确认退出，则接受关闭事件
        if not threads_running:
            event.accept()
        else:
            # 再次确认所有线程都已停止
            training_done = self.training_thread is None or not self.training_thread.isRunning()
            prediction_done = self.prediction_thread is None or not self.prediction_thread.isRunning()

            if training_done and prediction_done:
                event.accept()
            else:
                event.ignore()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("多目标变量机器学习系统")

    # 设置高DPI支持，避免在高分辨率屏幕上显示模糊
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()