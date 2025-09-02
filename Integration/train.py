import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, mean_absolute_error,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score, explained_variance_score
)
from xgboost import XGBRegressor, XGBClassifier
import joblib
import os
import datetime
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

class MultiTargetModelComparisonSystem:
    def __init__(self, output_dir=None):
        """
        初始化多目标变量系统

        参数:
        output_dir (str): 输出目录路径，如果为None则使用当前时间创建目录
        """
        self.models = {}
        self.results = {}
        self.best_models = {}  # 存储每个目标变量的最佳模型
        self.feature_importance_results = {}
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.X = None
        self.Y = None
        self.feature_names = None
        self.target_names = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.problem_types = {}  # 存储每个目标变量的问题类型
        self.overall_problem_type = None  # 整体问题类型

        # 创建输出目录结构
        if output_dir is None:
            # 可以设置输出文件夹名字
            # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # self.output_dir = f"multi_target_model_comparison_{timestamp}"
            self.output_dir = './output'
        else:
            self.output_dir = output_dir

        # 创建子目录
        self.csv_dir = os.path.join(self.output_dir, "csv")
        self.images_dir = os.path.join(self.output_dir, "images")
        self.models_dir = os.path.join(self.output_dir, "models")

        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        print(f"输出目录: {self.output_dir}")

    def load_and_preprocess_data(self, file_path, target_columns, test_size=0.2, random_state=42):
        """
        加载和预处理数据

        参数:
        file_path (str): CSV文件路径
        target_columns (list): 目标变量列名列表
        test_size (float): 测试集比例
        random_state (int): 随机种子
        """
        # 加载数据
        data = pd.read_csv(file_path)
        print(f"数据集形状: {data.shape}")
        print(f"目标变量: {target_columns}")

        # 处理缺失值
        data = data.dropna()

        # 分离特征和目标
        self.X = data.drop(columns=target_columns)
        self.Y = data[target_columns]
        self.feature_names = self.X.columns.tolist()
        self.target_names = target_columns

        # 确定每个目标变量的问题类型
        for target in target_columns:
            if data[target].dtype == 'object' or data[target].nunique() < 10:
                self.problem_types[target] = 'classification'
                print(f"目标变量 '{target}': 分类 (共有 {data[target].nunique()} 个类别)")
            else:
                self.problem_types[target] = 'regression'
                print(f"目标变量 '{target}': 回归")

        # 确定整体问题类型（如果所有目标变量都是同一类型）
        problem_types_set = set(self.problem_types.values())
        if len(problem_types_set) == 1:
            self.overall_problem_type = list(problem_types_set)[0]
            print(f"\n整体问题类型: {self.overall_problem_type}")
        else:
            print(f"\n混合问题类型: 包含回归和分类目标变量")

        # 编码分类特征
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col].astype(str))
            self.label_encoders[col] = le

        # 编码目标变量（如果是分类问题）
        self.target_encoders = {}
        for target in target_columns:
            if self.problem_types[target] == 'classification':
                le = LabelEncoder()
                self.Y[target] = le.fit_transform(self.Y[target])
                self.target_encoders[target] = le

        # 划分训练集和测试集
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state
        )

        # 标准化特征
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"\n训练集形状: X={self.X_train.shape}, Y={self.Y_train.shape}")
        print(f"测试集形状: X={self.X_test.shape}, Y={self.Y_test.shape}")

        return self.X, self.Y

    def initialize_models(self):
        """初始化所有模型"""
        # 为每个目标变量创建模型
        self.models = {}

        for target in self.target_names:
            if self.problem_types[target] == 'regression':
                self.models[target] = {
                    'Linear Regression': LinearRegression(),
                    'Stochastic Gradient Descent': SGDRegressor(max_iter=1000, tol=1e-3),
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
                }
            else:  # classification
                self.models[target] = {
                    'Logistic Regression': LogisticRegression(max_iter=1000),
                    'SGD Classifier': SGDClassifier(max_iter=1000, tol=1e-3),
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
                }

    def train_and_evaluate_models(self):
        """训练和评估所有模型"""
        self.results = {}
        self.best_models = {}

        for target in self.target_names:
            print(f"\n{'=' * 50}")
            print(f"处理目标变量: {target}")
            print(f"{'=' * 50}")

            # 提取当前目标变量的数据
            y_train = self.Y_train[target].values
            y_test = self.Y_test[target].values

            self.results[target] = {}

            for name, model in self.models[target].items():
                print(f"\n训练 {name}...")

                # 训练模型
                model.fit(self.X_train, y_train)

                # 预测
                y_pred = model.predict(self.X_test)

                # 评估模型
                if self.problem_types[target] == 'regression':
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    evs = explained_variance_score(y_test, y_pred)

                    self.results[target][name] = {
                        'MSE': mse,
                        'MAE': mae,
                        'R2': r2,
                        'Explained Variance': evs
                    }
                    print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, EVS: {evs:.4f}")
                else:
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    self.results[target][name] = {
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1
                    }
                    print(
                        f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

                    # 对于分类问题，还可以计算其他指标
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(self.X_test)
                        # 对于多分类问题，我们需要处理每个类别的概率
                        if y_proba.shape[1] > 2:
                            # 使用OneVsRest方法计算AUC
                            from sklearn.preprocessing import label_binarize
                            from sklearn.metrics import roc_auc_score
                            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                            auc_score = roc_auc_score(y_test_bin, y_proba, multi_class='ovr')
                        else:
                            # 二分类问题
                            y_proba = y_proba[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_proba)
                            auc_score = auc(fpr, tpr)

                        self.results[target][name]['AUC'] = auc_score
                        print(f"{name} - AUC: {auc_score:.4f}")

            # 确定当前目标变量的最佳模型
            if self.problem_types[target] == 'regression':
                # 对于回归问题，选择R$^2$最高的模型
                best_model_name = max(self.results[target].items(), key=lambda x: x[1]['R2'])[0]
            else:
                # 对于分类问题，选择F1分数最高的模型
                best_model_name = max(self.results[target].items(), key=lambda x: x[1]['F1 Score'])[0]

            self.best_models[target] = self.models[target][best_model_name]
            print(f"\n目标变量 '{target}' 的最佳模型: {best_model_name}")

        # 保存所有模型
        for target in self.target_names:
            for name, model in self.models[target].items():
                model_path = os.path.join(self.models_dir, f"{target}_{name.replace(' ', '_')}.pkl")
                joblib.dump(model, model_path)
                print(f"模型 {target}_{name} 已保存到: {model_path}")

            # 保存最佳模型
            best_model_path = os.path.join(self.models_dir, f"best_model_{target}.pkl")
            joblib.dump(self.best_models[target], best_model_path)
            print(f"最佳模型 {target} 已保存到: {best_model_path}")

        # 保存预处理对象
        preprocessing_path = os.path.join(self.models_dir, "preprocessing.pkl")
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encoders': self.target_encoders,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'problem_types': self.problem_types
        }, preprocessing_path)
        print(f"预处理对象已保存到: {preprocessing_path}")

        return self.results

    def plot_comprehensive_results(self):
        """绘制全面的模型比较结果"""
        # 为每个目标变量创建图表
        for target in self.target_names:
            if self.problem_types[target] == 'regression':
                # 创建子图
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                axes = axes.flatten()

                metrics = ['MSE', 'MAE', 'R2', 'Explained Variance']
                colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']

                for i, metric in enumerate(metrics):
                    values = [self.results[target][name][metric] for name in self.results[target]]

                    # 对于MSE和MAE，值越小越好，所以使用反向排序
                    if metric in ['MSE', 'MAE']:
                        # 找到最佳值（最小值）并突出显示
                        best_idx = np.argmin(values)
                        colors_list = [colors[i] if j != best_idx else 'red' for j in range(len(values))]
                    else:
                        # 对于R$^2$和EVS，值越大越好
                        best_idx = np.argmax(values)
                        colors_list = [colors[i] if j != best_idx else 'green' for j in range(len(values))]

                    axes[i].bar(self.results[target].keys(), values, color=colors_list)
                    axes[i].set_title(f'{target} - 模型比较 - {metric}')
                    axes[i].set_ylabel(metric)
                    axes[i].tick_params(axis='x', rotation=45)

                    # 添加数值标签
                    for j, v in enumerate(values):
                        axes[i].text(j, v + 0.01 * (max(values) - min(values)),
                                     f'{v:.4f}', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(os.path.join(self.images_dir, f'{target}_regression_comparison.png'),
                            dpi=300, bbox_inches='tight')
                plt.show()

            else:  # classification
                # 创建子图
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                axes = axes.flatten()

                metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                if 'AUC' in list(self.results[target].values())[0]:
                    metrics.append('AUC')

                colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightblue']

                for i, metric in enumerate(metrics):
                    if i >= len(axes):
                        break

                    values = [self.results[target][name][metric] for name in self.results[target]]

                    # 所有分类指标都是值越大越好
                    best_idx = np.argmax(values)
                    colors_list = [colors[i] if j != best_idx else 'green' for j in range(len(values))]

                    axes[i].bar(self.results[target].keys(), values, color=colors_list)
                    axes[i].set_title(f'{target} - 模型比较 - {metric}')
                    axes[i].set_ylabel(metric)
                    axes[i].tick_params(axis='x', rotation=45)

                    # 添加数值标签
                    for j, v in enumerate(values):
                        axes[i].text(j, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

                # 隐藏多余的子图
                for i in range(len(metrics), len(axes)):
                    axes[i].set_visible(False)

                plt.tight_layout()
                plt.savefig(os.path.join(self.images_dir, f'{target}_classification_comparison.png'),
                            dpi=300, bbox_inches='tight')
                plt.show()

    def plot_prediction_vs_actual(self):
        """绘制预测值与实际值的对比图（仅回归问题）"""
        for target in self.target_names:
            if self.problem_types[target] != 'regression':
                continue

            # 使用最佳模型进行预测
            y_pred = self.best_models[target].predict(self.X_test)
            y_test = self.Y_test[target].values

            # 创建散点图
            plt.figure(figsize=(10, 8))
            plt.scatter(y_test, y_pred, alpha=0.5)

            # 添加理想线
            max_val = max(np.max(y_test), np.max(y_pred))
            min_val = min(np.min(y_test), np.min(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.title(f'{target} - 预测值 vs 实际值')

            # 添加R$^2$分数
            r2 = r2_score(y_test, y_pred)
            plt.text(0.05, 0.95, f'R$^2$ = {r2:.4f}', transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))

            plt.savefig(os.path.join(self.images_dir, f'{target}_prediction_vs_actual.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

            # 绘制残差图
            residuals = y_test - y_pred
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('预测值')
            plt.ylabel('残差')
            plt.title(f'{target} - 残差图')

            plt.savefig(os.path.join(self.images_dir, f'{target}_residual_plot.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

    def plot_error_analysis(self):
        """绘制误差分析图（仅回归问题）"""
        for target in self.target_names:
            if self.problem_types[target] != 'regression':
                continue

            # 使用最佳模型进行预测
            y_pred = self.best_models[target].predict(self.X_test)
            y_test = self.Y_test[target].values
            errors = np.abs(y_test - y_pred)

            # 绘制误差分布
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.hist(errors, bins=30, alpha=0.7, color='skyblue')
            plt.xlabel('绝对误差')
            plt.ylabel('频率')
            plt.title(f'{target} - 绝对误差分布')

            # 绘制误差箱线图
            plt.subplot(1, 2, 2)
            plt.boxplot(errors)
            plt.ylabel('绝对误差')
            plt.title(f'{target} - 绝对误差箱线图')

            plt.tight_layout()
            plt.savefig(os.path.join(self.images_dir, f'{target}_error_analysis.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

            # 计算并打印误差统计
            print(f"\n{target} 误差分析:")
            print(f"平均绝对误差: {np.mean(errors):.4f}")
            print(f"误差标准差: {np.std(errors):.4f}")
            print(f"最大误差: {np.max(errors):.4f}")
            print(f"误差中位数: {np.median(errors):.4f}")
            print(f"95%分位数误差: {np.percentile(errors, 95):.4f}")

    def plot_learning_curves(self):
        """绘制学习曲线"""
        for target in self.target_names:
            if self.best_models[target] is None:
                continue

            y_train = self.Y_train[target].values

            train_sizes, train_scores, test_scores = learning_curve(
                self.best_models[target], self.X_train, y_train, cv=5,
                train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1,
                scoring='r2' if self.problem_types[target] == 'regression' else 'accuracy'
            )

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, 'o-', color='r', label='训练得分')
            plt.plot(train_sizes, test_mean, 'o-', color='g', label='交叉验证得分')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

            plt.title(f'{target} - 学习曲线')
            plt.xlabel('训练样本数')
            plt.ylabel('R$^2$分数' if self.problem_types[target] == 'regression' else '准确率')
            plt.legend(loc='best')
            plt.grid(True)
            plt.savefig(os.path.join(self.images_dir, f'{target}_learning_curve.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

    def plot_confusion_matrix(self):
        """绘制混淆矩阵（仅分类问题）"""
        for target in self.target_names:
            if self.problem_types[target] != 'classification':
                continue

            y_pred = self.best_models[target].predict(self.X_test)
            y_test = self.Y_test[target].values
            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{target} - 混淆矩阵')
            plt.ylabel('真实标签')
            plt.xlabel('预测标签')
            plt.savefig(os.path.join(self.images_dir, f'{target}_confusion_matrix.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

    def plot_roc_curve(self):
        """绘制ROC曲线（仅分类问题且支持概率预测）"""
        for target in self.target_names:
            if (self.problem_types[target] != 'classification' or
                    not hasattr(self.best_models[target], "predict_proba")):
                continue

            y_proba = self.best_models[target].predict_proba(self.X_test)
            y_test = self.Y_test[target].values

            # 处理多分类问题
            if y_proba.shape[1] > 2:
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_curve, auc

                # 二值化输出
                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                n_classes = y_test_bin.shape[1]

                # 计算每个类的ROC曲线和AUC
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                # 绘制所有ROC曲线
                plt.figure(figsize=(8, 6))
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
                for i, color in zip(range(n_classes), colors):
                    if i < len(colors):
                        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                                 label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

            else:
                # 二分类问题
                y_proba = y_proba[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2,
                         label='ROC curve (area = %0.2f)' % roc_auc)

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正率')
            plt.ylabel('真正率')
            plt.title(f'{target} - ROC曲线')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.images_dir, f'{target}_roc_curve.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

    def plot_feature_importance(self):
        """绘制特征重要性（对于树模型）"""
        for target in self.target_names:
            if not hasattr(self.best_models[target], 'feature_importances_'):
                print(f"{target} 的模型不支持特征重要性分析")
                continue

            importances = self.best_models[target].feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.title(f"{target} - 特征重要性")
            plt.bar(range(len(importances)), importances[indices], align="center")
            plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45)
            plt.xlim([-1, len(importances)])
            plt.savefig(os.path.join(self.images_dir, f'{target}_feature_importance.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

    def save_results(self):
        """保存结果到CSV文件"""
        # 为每个目标变量保存结果
        for target in self.target_names:
            if self.problem_types[target] == 'regression':
                df = pd.DataFrame.from_dict(self.results[target], orient='index')
            else:
                df = pd.DataFrame.from_dict(self.results[target], orient='index')

            csv_path = os.path.join(self.csv_dir, f'{target}_model_comparison_results.csv')
            df.to_csv(csv_path)
            print(f"{target} 结果已保存到 {csv_path}")

        # 创建汇总表格
        summary_data = []
        for target in self.target_names:
            best_model_name = max(self.results[target].items(),
                                  key=lambda x: x[1][
                                      'R2' if self.problem_types[target] == 'regression' else 'F1 Score'])[0]
            best_score = self.results[target][best_model_name][
                'R2' if self.problem_types[target] == 'regression' else 'F1 Score']

            summary_data.append({
                'Target': target,
                'Type': self.problem_types[target],
                'Best Model': best_model_name,
                'Best Score': best_score
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.csv_dir, 'summary_results.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"汇总结果已保存到 {summary_path}")

    def run_comprehensive_analysis(self, file_path, target_columns):
        """运行全面的分析流程"""
        print("=" * 50)
        print("开始多目标变量机器学习模型比较分析")
        print("=" * 50)

        # 1. 加载和预处理数据
        X, Y = self.load_and_preprocess_data(file_path, target_columns)

        # 2. 初始化模型
        self.initialize_models()

        # 3. 训练和评估模型
        results = self.train_and_evaluate_models()

        # 4. 绘制全面的结果比较
        self.plot_comprehensive_results()

        # 5. 对于回归问题，绘制预测值与实际值对比图
        self.plot_prediction_vs_actual()

        # 6. 对于回归问题，绘制误差分析图
        self.plot_error_analysis()

        # 7. 对于分类问题，绘制混淆矩阵
        self.plot_confusion_matrix()

        # 8. 对于分类问题，绘制ROC曲线
        self.plot_roc_curve()

        # 9. 绘制学习曲线
        self.plot_learning_curves()

        # 10. 绘制特征重要性
        self.plot_feature_importance()

        # 11. 保存结果
        self.save_results()

        print("=" * 50)
        print("分析完成!")
        print("=" * 50)

        return results


# 使用示例
if __name__ == "__main__":
    # 创建系统实例
    system = MultiTargetModelComparisonSystem()

    # 运行全面分析
    # 请替换为您的CSV文件路径和目标列名列表
    file_path = "数据.csv"  # 替换为您的CSV文件路径
    target_columns = ["升力系数", "阻力系数", "俯仰力矩系数", "压心", "内流升力系数", "内流阻力系数",
                      "内流俯仰力矩系数"]

    # 运行分析
    results = system.run_comprehensive_analysis(file_path, target_columns)

    # 打印结果摘要
    print("\n模型性能摘要:")
    for target, target_results in results.items():
        print(f"\n目标变量: {target}")
        for model_name, metrics in target_results.items():
            print(f"  {model_name}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value:.4f}")