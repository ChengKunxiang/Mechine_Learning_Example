import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

class EnhancedModelComparisonSystem:
    def __init__(self, output_dir=None):
        """
        初始化系统

        参数:
        output_dir (str): 输出目录路径，如果为None则使用当前时间创建目录
        """
        self.models = {}
        self.results = {}
        self.best_models = {}  # 存储每个指标的最佳模型
        self.feature_importance_results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.problem_type = None  # 'regression' or 'classification'
        self.best_model = None
        self.best_model_name = None

        # 创建输出目录结构
        if output_dir is None:
            # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # self.output_dir = f"model_comparison_{timestamp}"
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

    def load_and_preprocess_data(self, file_path, target_column, test_size=0.2, random_state=42):
        """
        加载和预处理数据
        """
        # 加载数据
        data = pd.read_csv(file_path)
        print(f"数据集形状: {data.shape}")

        # 处理缺失值
        data = data.dropna()

        # 分离特征和目标
        self.X = data.drop(columns=[target_column])
        self.y = data[target_column]
        self.feature_names = self.X.columns.tolist()

        # 确定问题类型
        if self.y.dtype == 'object' or self.y.nunique() < 10:
            self.problem_type = 'classification'
            print(f"\n问题类型: 分类 (共有 {self.y.nunique()} 个类别)")
        else:
            self.problem_type = 'regression'
            print(f"\n问题类型: 回归")

        # 编码分类特征
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col].astype(str))
            self.label_encoders[col] = le

        # 编码目标变量（如果是分类问题）
        if self.problem_type == 'classification':
            self.target_encoder = LabelEncoder()
            self.y = self.target_encoder.fit_transform(self.y)

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # 标准化特征
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"\n训练集形状: {self.X_train.shape}")
        print(f"测试集形状: {self.X_test.shape}")

        return self.X, self.y

    def initialize_models(self):
        """初始化所有模型"""
        if self.problem_type == 'regression':
            self.models = {
                'Linear Regression': LinearRegression(),
                'Stochastic Gradient Descent': SGDRegressor(max_iter=1000, tol=1e-3),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
            }
        else:  # classification
            self.models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'SGD Classifier': SGDClassifier(max_iter=1000, tol=1e-3),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
            }

    def train_and_evaluate_models(self):
        """训练和评估所有模型"""
        self.results = {}

        for name, model in self.models.items():
            print(f"\n训练 {name}...")

            # 训练模型
            model.fit(self.X_train, self.y_train)

            # 预测
            y_pred = model.predict(self.X_test)

            # 评估模型
            if self.problem_type == 'regression':
                mse = mean_squared_error(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                evs = explained_variance_score(self.y_test, y_pred)

                self.results[name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'R2': r2,
                    'Explained Variance': evs
                }
                print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, EVS: {evs:.4f}")
            else:
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)

                self.results[name] = {
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
                        y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
                        auc_score = roc_auc_score(y_test_bin, y_proba, multi_class='ovr')
                    else:
                        # 二分类问题
                        y_proba = y_proba[:, 1]
                        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                        auc_score = auc(fpr, tpr)

                    self.results[name]['AUC'] = auc_score
                    print(f"{name} - AUC: {auc_score:.4f}")

        # 确定每个指标的最佳模型
        self.best_models = {}
        if self.problem_type == 'regression':
            metrics = ['MSE', 'MAE', 'R2', 'Explained Variance']
            for metric in metrics:
                if metric in ['MSE', 'MAE']:
                    # 对于MSE和MAE，值越小越好
                    best_model_name = min(self.results.items(), key=lambda x: x[1][metric])[0]
                else:
                    # 对于R$^2$和EVS，值越大越好
                    best_model_name = max(self.results.items(), key=lambda x: x[1][metric])[0]
                self.best_models[metric] = best_model_name
                print(f"{metric} 最佳模型: {best_model_name}")

            # 默认选择R$^2$最高的模型作为整体最佳模型
            self.best_model_name = self.best_models['R2']
        else:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
            for metric in metrics:
                if metric in list(self.results.values())[0]:
                    best_model_name = max(self.results.items(), key=lambda x: x[1][metric])[0]
                    self.best_models[metric] = best_model_name
                    print(f"{metric} 最佳模型: {best_model_name}")

            # 默认选择F1分数最高的模型作为整体最佳模型
            self.best_model_name = self.best_models['F1 Score']

        self.best_model = self.models[self.best_model_name]
        print(
            f"\n整体最佳模型 (基于{'R2' if self.problem_type == 'regression' else 'F1 Score'}): {self.best_model_name}")

        # 保存所有模型
        for name, model in self.models.items():
            model_path = os.path.join(self.models_dir, f"{name.replace(' ', '_')}.pkl")
            joblib.dump(model, model_path)
            print(f"模型 {name} 已保存到: {model_path}")

        # 保存预处理对象
        preprocessing_path = os.path.join(self.models_dir, "preprocessing.pkl")
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encoder': getattr(self, 'target_encoder', None),
            'feature_names': self.feature_names,
            'problem_type': self.problem_type,
            'target_name': 'target'
        }, preprocessing_path)
        print(f"预处理对象已保存到: {preprocessing_path}")

        return self.results

    def analyze_feature_performance(self):
        """分析各模型在各变量上的性能（仅回归问题）"""
        if self.problem_type != 'regression':
            print("特征性能分析仅适用于回归问题")
            return None

        self.feature_importance_results = {}

        for model_name, model in self.models.items():
            print(f"\n分析 {model_name} 在各变量上的性能...")

            # 为每个特征单独训练模型
            feature_r2 = {}
            feature_mae = {}

            for i, feature_name in enumerate(self.feature_names):
                # 只使用当前特征进行训练和预测
                X_train_single = self.X_train[:, i].reshape(-1, 1)
                X_test_single = self.X_test[:, i].reshape(-1, 1)

                # 克隆模型
                from sklearn.base import clone
                single_feature_model = clone(model)

                # 训练和预测
                single_feature_model.fit(X_train_single, self.y_train)
                y_pred = single_feature_model.predict(X_test_single)

                # 计算指标
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)

                feature_r2[feature_name] = r2
                feature_mae[feature_name] = mae

            self.feature_importance_results[model_name] = {
                'R2': feature_r2,
                'MAE': feature_mae
            }

            # 打印每个模型的特征性能
            print(f"{model_name} 各变量R$^2$:")
            for feature, r2_val in sorted(feature_r2.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {r2_val:.4f}")

        return self.feature_importance_results

    def plot_feature_performance_comparison(self):
        """绘制各模型在各变量上的性能比较图（仅回归问题）"""
        if not self.feature_importance_results:
            print("请先运行 analyze_feature_performance() 方法")
            return

        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        # 准备数据
        models = list(self.feature_importance_results.keys())
        features = list(self.feature_importance_results[models[0]]['R2'].keys())

        # R$^2$比较
        r2_data = []
        for model in models:
            r2_values = [self.feature_importance_results[model]['R2'][feature] for feature in features]
            r2_data.append(r2_values)

        # MAE比较
        mae_data = []
        for model in models:
            mae_values = [self.feature_importance_results[model]['MAE'][feature] for feature in features]
            mae_data.append(mae_values)

        # 绘制R$^2$热力图
        r2_df = pd.DataFrame(r2_data, index=models, columns=features)
        sns.heatmap(r2_df, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax1)
        ax1.set_title('各模型在各变量上的R$^2$分数')
        ax1.tick_params(axis='x', rotation=45)

        # 绘制MAE热力图
        mae_df = pd.DataFrame(mae_data, index=models, columns=features)
        sns.heatmap(mae_df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax2)
        ax2.set_title('各模型在各变量上的MAE')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, 'feature_performance_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 绘制每个特征的模型比较
        self.plot_performance_by_feature()

    def plot_performance_by_feature(self):
        """按特征绘制各模型的性能比较（仅回归问题）"""
        if not self.feature_importance_results:
            print("请先运行 analyze_feature_performance() 方法")
            return

        models = list(self.feature_importance_results.keys())
        features = list(self.feature_importance_results[models[0]]['R2'].keys())

        # 为每个特征创建一个图表
        for feature in features:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # R$^2$比较
            r2_values = [self.feature_importance_results[model]['R2'][feature] for model in models]
            bars1 = ax1.bar(models, r2_values, color='skyblue')
            ax1.set_title(f'{feature} - 各模型R$^2$比较')
            ax1.set_ylabel('R$^2$分数')
            ax1.tick_params(axis='x', rotation=45)

            # 在柱状图上添加数值标签
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom')

            # MAE比较
            mae_values = [self.feature_importance_results[model]['MAE'][feature] for model in models]
            bars2 = ax2.bar(models, mae_values, color='lightcoral')
            ax2.set_title(f'{feature} - 各模型MAE比较')
            ax2.set_ylabel('MAE')
            ax2.tick_params(axis='x', rotation=45)

            # 在柱状图上添加数值标签
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(self.images_dir, f'performance_comparison_{feature}.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

    def plot_comprehensive_results(self):
        """绘制全面的模型比较结果"""
        if self.problem_type == 'regression':
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()

            metrics = ['MSE', 'MAE', 'R2', 'Explained Variance']
            colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']

            for i, metric in enumerate(metrics):
                values = [self.results[name][metric] for name in self.results]

                # 对于MSE和MAE，值越小越好，所以使用反向排序
                if metric in ['MSE', 'MAE']:
                    # 找到最佳值（最小值）并突出显示
                    best_idx = np.argmin(values)
                    colors_list = [colors[i] if j != best_idx else 'red' for j in range(len(values))]
                else:
                    # 对于R$^2$和EVS，值越大越好
                    best_idx = np.argmax(values)
                    colors_list = [colors[i] if j != best_idx else 'green' for j in range(len(values))]

                axes[i].bar(self.results.keys(), values, color=colors_list)
                axes[i].set_title(f'模型比较 - {metric} (最佳: {self.best_models[metric]})')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)

                # 添加数值标签
                for j, v in enumerate(values):
                    axes[i].text(j, v + 0.01 * (max(values) - min(values)),
                                 f'{v:.4f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(self.images_dir, 'comprehensive_regression_comparison.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

        else:  # classification
            # 创建子图
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()

            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            if 'AUC' in list(self.results.values())[0]:
                metrics.append('AUC')

            colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightblue']

            for i, metric in enumerate(metrics):
                if i >= len(axes):
                    break

                values = [self.results[name][metric] for name in self.results]

                # 所有分类指标都是值越大越好
                best_idx = np.argmax(values)
                colors_list = [colors[i] if j != best_idx else 'green' for j in range(len(values))]

                axes[i].bar(self.results.keys(), values, color=colors_list)
                axes[i].set_title(f'模型比较 - {metric} (最佳: {self.best_models[metric]})')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)

                # 添加数值标签
                for j, v in enumerate(values):
                    axes[i].text(j, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

            # 隐藏多余的子图
            for i in range(len(metrics), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(self.images_dir, 'comprehensive_classification_comparison.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

    def plot_prediction_vs_actual(self):
        """绘制预测值与实际值的对比图（仅回归问题）"""
        if self.problem_type != 'regression':
            return

        # 使用最佳模型进行预测
        y_pred = self.best_model.predict(self.X_test)

        # 创建散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_test, y_pred, alpha=0.5)

        # 添加理想线
        max_val = max(np.max(self.y_test), np.max(y_pred))
        min_val = min(np.min(self.y_test), np.min(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'{self.best_model_name} - 预测值 vs 实际值')

        # 添加R$^2$分数
        r2 = r2_score(self.y_test, y_pred)
        plt.text(0.05, 0.95, f'R$^2$ = {r2:.4f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

        plt.savefig(os.path.join(self.images_dir, 'prediction_vs_actual.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 绘制残差图
        residuals = self.y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title(f'{self.best_model_name} - 残差图')

        plt.savefig(os.path.join(self.images_dir, 'residual_plot.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_error_analysis(self):
        """绘制误差分析图（仅回归问题）"""
        if self.problem_type != 'regression':
            return

        # 使用最佳模型进行预测
        y_pred = self.best_model.predict(self.X_test)
        errors = np.abs(self.y_test - y_pred)

        # 绘制误差分布
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=30, alpha=0.7, color='skyblue')
        plt.xlabel('绝对误差')
        plt.ylabel('频率')
        plt.title('绝对误差分布')

        # 绘制误差箱线图
        plt.subplot(1, 2, 2)
        plt.boxplot(errors)
        plt.ylabel('绝对误差')
        plt.title('绝对误差箱线图')

        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, 'error_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 计算并打印误差统计
        print("\n误差分析:")
        print(f"平均绝对误差: {np.mean(errors):.4f}")
        print(f"误差标准差: {np.std(errors):.4f}")
        print(f"最大误差: {np.max(errors):.4f}")
        print(f"误差中位数: {np.median(errors):.4f}")
        print(f"95%分位数误差: {np.percentile(errors, 95):.4f}")

    def plot_learning_curves(self):
        """绘制学习曲线"""
        if self.best_model is None:
            return

        train_sizes, train_scores, test_scores = learning_curve(
            self.best_model, self.X_train, self.y_train, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1,
            scoring='r2' if self.problem_type == 'regression' else 'accuracy'
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

        plt.title(f'{self.best_model_name} 的学习曲线')
        plt.xlabel('训练样本数')
        plt.ylabel('R$^2$分数' if self.problem_type == 'regression' else '准确率')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(self.images_dir, 'learning_curve.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self):
        """绘制混淆矩阵（仅分类问题）"""
        if self.problem_type != 'classification':
            return

        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{self.best_model_name} 的混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(os.path.join(self.images_dir, 'confusion_matrix.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curve(self):
        """绘制ROC曲线（仅分类问题且支持概率预测）"""
        if (self.problem_type != 'classification' or
                not hasattr(self.best_model, "predict_proba")):
            return

        y_proba = self.best_model.predict_proba(self.X_test)

        # 处理多分类问题
        if y_proba.shape[1] > 2:
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc

            # 二值化输出
            y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
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
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label='ROC curve (area = %0.2f)' % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title(f'{self.best_model_name} 的ROC曲线')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.images_dir, 'roc_curve.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self):
        """绘制特征重要性（对于树模型）"""
        if not hasattr(self.best_model, 'feature_importances_'):
            print("该模型不支持特征重要性分析")
            return

        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("特征重要性")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45)
        plt.xlim([-1, len(importances)])
        plt.savefig(os.path.join(self.images_dir, 'feature_importance.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self):
        """保存结果到CSV文件"""
        if self.problem_type == 'regression':
            df = pd.DataFrame.from_dict(self.results, orient='index')
            # 添加每个指标的最佳模型标记
            for metric in ['MSE', 'MAE', 'R2', 'Explained Variance']:
                best_model = self.best_models[metric]
                df[f'最佳{metric}'] = df.index == best_model
        else:
            df = pd.DataFrame.from_dict(self.results, orient='index')
            # 添加每个指标的最佳模型标记
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
                if metric in self.best_models:
                    best_model = self.best_models[metric]
                    df[f'最佳{metric}'] = df.index == best_model

        csv_path = os.path.join(self.csv_dir, 'model_comparison_results.csv')
        df.to_csv(csv_path)
        print(f"\n结果已保存到 {csv_path}")

        # 如果进行了特征性能分析，也保存这些结果
        if self.feature_importance_results:
            # 保存R$^2$结果
            r2_df = pd.DataFrame()
            for model, metrics in self.feature_importance_results.items():
                r2_df[model] = pd.Series(metrics['R2'])
            r2_path = os.path.join(self.csv_dir, 'feature_r2_results.csv')
            r2_df.to_csv(r2_path)
            print(f"特征R$^2$结果已保存到 {r2_path}")

            # 保存MAE结果
            mae_df = pd.DataFrame()
            for model, metrics in self.feature_importance_results.items():
                mae_df[model] = pd.Series(metrics['MAE'])
            mae_path = os.path.join(self.csv_dir, 'feature_mae_results.csv')
            mae_df.to_csv(mae_path)
            print(f"特征MAE结果已保存到 {mae_path}")

    def run_comprehensive_analysis(self, file_path, target_column):
        """运行全面的分析流程"""
        print("=" * 50)
        print("开始全面的机器学习模型比较分析")
        print("=" * 50)

        # 1. 加载和预处理数据
        X, y = self.load_and_preprocess_data(file_path, target_column)

        # 2. 初始化模型
        self.initialize_models()

        # 3. 训练和评估模型
        results = self.train_and_evaluate_models()

        # 4. 绘制全面的结果比较
        self.plot_comprehensive_results()

        # 5. 对于回归问题，分析各变量上的性能
        if self.problem_type == 'regression':
            feature_results = self.analyze_feature_performance()
            self.plot_feature_performance_comparison()

        # 6. 对于回归问题，绘制预测值与实际值对比图
        if self.problem_type == 'regression':
            self.plot_prediction_vs_actual()
            self.plot_error_analysis()

        # 7. 对于分类问题，绘制混淆矩阵和ROC曲线
        else:
            self.plot_confusion_matrix()
            self.plot_roc_curve()

        # 8. 绘制学习曲线
        self.plot_learning_curves()

        # 9. 绘制特征重要性
        self.plot_feature_importance()

        # 10. 保存结果
        self.save_results()

        print("=" * 50)
        print("分析完成!")
        print("=" * 50)

        return results


# 使用示例
if __name__ == "__main__":
    # 创建系统实例
    system = EnhancedModelComparisonSystem()

    # 运行全面分析
    # 请替换为您的CSV文件路径和目标列名
    file_path = "数据.csv"  # 替换为您的CSV文件路径
    target_column = "升力系数"  # 替换为您的目标列名

    # 运行分析
    results = system.run_comprehensive_analysis(file_path, target_column)

    # 打印结果摘要
    print("\n模型性能摘要:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print()

    # 打印每个指标的最佳模型
    print("\n每个指标的最佳模型:")
    for metric, best_model in system.best_models.items():
        print(f"{metric}: {best_model}")