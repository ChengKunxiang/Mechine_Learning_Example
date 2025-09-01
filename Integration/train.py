import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, mean_absolute_error,
    classification_report, confusion_matrix, roc_curve, auc
)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import learning_curve, validation_curve
import joblib
import os
import datetime
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

class EnhancedModelComparisonSystem:
    def __init__(self, output_dir=None):
        """
        初始化系统

        参数:
        output_dir (str): 输出目录路径，如果为None则使用当前时间创建目录
        """
        self.models = {}
        self.results = {}
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
            # 可以选择导出的文件夹
            # self.output_dir = f"model_comparison_{timestamp}"
            self.output_dir = "output"
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

        参数:
        file_path (str): CSV文件路径
        target_column (str): 目标变量列名
        test_size (float): 测试集比例
        random_state (int): 随机种子
        """
        # 加载数据
        data = pd.read_csv(file_path)
        print(f"数据集形状: {data.shape}")
        print("\n前5行数据:")
        print(data.head())

        # 检查缺失值
        print("\n缺失值统计:")
        print(data.isnull().sum())

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

            # 保存模型
            model_path = os.path.join(self.models_dir, f"{name.replace(' ', '_')}.pkl")
            joblib.dump(model, model_path)
            print(f"模型已保存到: {model_path}")

            # 预测
            y_pred = model.predict(self.X_test)

            # 评估模型
            if self.problem_type == 'regression':
                mse = mean_squared_error(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                self.results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
                print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            else:
                accuracy = accuracy_score(self.y_test, y_pred)
                self.results[name] = {'Accuracy': accuracy}
                print(f"{name} - Accuracy: {accuracy:.4f}")

                # 对于分类问题，还可以计算其他指标
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(self.X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    self.results[name]['AUC'] = roc_auc
                    print(f"{name} - AUC: {roc_auc:.4f}")

        # 确定最佳模型
        if self.problem_type == 'regression':
            # 对于回归问题，选择R$^2$最高的模型
            self.best_model_name = max(self.results.items(), key=lambda x: x[1]['R2'])[0]
        else:
            # 对于分类问题，选择准确率最高的模型
            self.best_model_name = max(self.results.items(), key=lambda x: x[1]['Accuracy'])[0]

        self.best_model = self.models[self.best_model_name]
        print(f"\n最佳模型: {self.best_model_name}")

        # 保存最佳模型
        best_model_path = os.path.join(self.models_dir, "best_model.pkl")
        joblib.dump(self.best_model, best_model_path)
        print(f"最佳模型已保存到: {best_model_path}")

        # 保存预处理对象
        preprocessing_path = os.path.join(self.models_dir, "preprocessing.pkl")
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encoder': getattr(self, 'target_encoder', None),
            'feature_names': self.feature_names,
            'problem_type': self.problem_type
        }, preprocessing_path)
        print(f"预处理对象已保存到: {preprocessing_path}")

        return self.results

    def analyze_feature_performance(self):
        """分析各模型在各变量上的性能"""
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
        """绘制各模型在各变量上的性能比较图"""
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
        """按特征绘制各模型的性能比较"""
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
            ax1.bar(models, r2_values, color='skyblue')
            ax1.set_title(f'{feature} - 各模型R$^2$比较')
            ax1.set_ylabel('R$^2$分数')
            ax1.tick_params(axis='x', rotation=45)

            # MAE比较
            mae_values = [self.feature_importance_results[model]['MAE'][feature] for model in models]
            ax2.bar(models, mae_values, color='lightcoral')
            ax2.set_title(f'{feature} - 各模型MAE比较')
            ax2.set_ylabel('MAE')
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(self.images_dir, f'performance_comparison_{feature}.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

    def plot_results(self):
        """绘制模型比较结果"""
        if self.problem_type == 'regression':
            # 创建子图
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

            # MSE比较
            mse_values = [self.results[name]['MSE'] for name in self.results]
            ax1.bar(self.results.keys(), mse_values, color='skyblue')
            ax1.set_title('模型比较 - 均方误差 (MSE)')
            ax1.set_ylabel('MSE')
            ax1.tick_params(axis='x', rotation=45)

            # MAE比较
            mae_values = [self.results[name]['MAE'] for name in self.results]
            ax2.bar(self.results.keys(), mae_values, color='lightgreen')
            ax2.set_title('模型比较 - 平均绝对误差 (MAE)')
            ax2.set_ylabel('MAE')
            ax2.tick_params(axis='x', rotation=45)

            # R$^2$比较
            r2_values = [self.results[name]['R2'] for name in self.results]
            ax3.bar(self.results.keys(), r2_values, color='lightcoral')
            ax3.set_title('模型比较 - R$^2$分数')
            ax3.set_ylabel('R$^2$')
            ax3.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(self.images_dir, 'regression_model_comparison.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

            # 学习曲线
            self.plot_learning_curves()

        else:  # classification
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # 准确率比较
            acc_values = [self.results[name]['Accuracy'] for name in self.results]
            ax1.bar(self.results.keys(), acc_values, color='skyblue')
            ax1.set_title('模型比较 - 准确率')
            ax1.set_ylabel('准确率')
            ax1.tick_params(axis='x', rotation=45)

            # AUC比较（如果可用）
            if 'AUC' in list(self.results.values())[0]:
                auc_values = [self.results[name].get('AUC', 0) for name in self.results]
                ax2.bar(self.results.keys(), auc_values, color='lightgreen')
                ax2.set_title('模型比较 - AUC')
                ax2.set_ylabel('AUC')
                ax2.tick_params(axis='x', rotation=45)
            else:
                ax2.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(self.images_dir, 'classification_model_comparison.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

            # 绘制最佳模型的混淆矩阵
            self.plot_confusion_matrix(self.best_model, self.models)

            # 绘制ROC曲线（如果可用）
            if hasattr(self.best_model, "predict_proba"):
                self.plot_roc_curve(self.best_model)

            # 学习曲线
            self.plot_learning_curves()

    def plot_learning_curves(self):
        """绘制学习曲线"""
        if self.best_model is None:
            return

        train_sizes, train_scores, test_scores = learning_curve(
            self.best_model, self.X_train, self.y_train, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
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

        plt.title(f'{type(self.best_model).__name__} 的学习曲线')
        plt.xlabel('训练样本数')
        plt.ylabel('得分')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(self.images_dir, 'learning_curve.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, best_model, models):
        """绘制混淆矩阵"""
        # 找到最佳模型的名称
        best_model_name = None
        for name, model in models.items():
            if model == best_model:
                best_model_name = name
                break

        if best_model_name is None:
            return

        y_pred = best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{best_model_name} 的混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(os.path.join(self.images_dir, 'confusion_matrix.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curve(self, model):
        """绘制ROC曲线"""
        if not hasattr(model, "predict_proba"):
            return

        y_proba = model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title('接收者操作特征(ROC)曲线')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.images_dir, 'roc_curve.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self):
        """绘制特征重要性（对于树模型）"""
        if hasattr(self.best_model, 'feature_importances_'):
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

    def cross_validate_models(self, cv=5):
        """使用交叉验证评估模型"""
        cv_results = {}

        for name, model in self.models.items():
            print(f"\n对 {name} 进行交叉验证...")
            if self.problem_type == 'regression':
                scoring = 'r2'
            else:
                scoring = 'accuracy'

            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring, n_jobs=-1)
            cv_results[name] = {
                '平均得分': np.mean(scores),
                '标准差': np.std(scores),
                '所有得分': scores
            }
            print(f"{name} - 平均得分: {np.mean(scores):.4f} (±{np.std(scores):.4f})")

        return cv_results

    def save_results(self):
        """保存结果到CSV文件"""
        if self.problem_type == 'regression':
            df = pd.DataFrame.from_dict(self.results, orient='index')
            df['最佳模型'] = df['R2'] == df['R2'].max()
        else:
            df = pd.DataFrame.from_dict(self.results, orient='index')
            df['最佳模型'] = df['Accuracy'] == df['Accuracy'].max()

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

    def run_complete_analysis(self, file_path, target_column):
        """运行完整的分析流程"""
        print("=" * 50)
        print("开始机器学习模型比较分析")
        print("=" * 50)

        # 1. 加载和预处理数据
        X, y = self.load_and_preprocess_data(file_path, target_column)

        # 2. 初始化模型
        self.initialize_models()

        # 3. 训练和评估模型
        results = self.train_and_evaluate_models()

        # 4. 分析各模型在各变量上的性能（仅回归问题）
        if self.problem_type == 'regression':
            feature_results = self.analyze_feature_performance()
            self.plot_feature_performance_comparison()

        # 5. 交叉验证
        cv_results = self.cross_validate_models()

        # 6. 绘制结果
        self.plot_results()

        # 7. 绘制特征重要性
        self.plot_feature_importance()

        # 8. 保存结果
        self.save_results()

        print("=" * 50)
        print("分析完成!")
        print(f"所有结果已保存到: {self.output_dir}")
        print("=" * 50)

        return results, cv_results


# 预测函数
def predict_with_saved_model(new_data_path, model_dir):
    """
    使用保存的模型进行预测

    参数:
    new_data_path (str): 新数据的CSV文件路径
    model_dir (str): 模型目录路径

    返回:
    预测结果
    """
    # 加载预处理对象
    preprocessing_path = os.path.join(model_dir, "preprocessing.pkl")
    if not os.path.exists(preprocessing_path):
        raise FileNotFoundError(f"预处理文件未找到: {preprocessing_path}")

    preprocessing = joblib.load(preprocessing_path)
    scaler = preprocessing['scaler']
    label_encoders = preprocessing['label_encoders']
    target_encoder = preprocessing['target_encoder']
    feature_names = preprocessing['feature_names']
    problem_type = preprocessing['problem_type']

    # 加载最佳模型
    best_model_path = os.path.join(model_dir, "best_model.pkl")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"最佳模型文件未找到: {best_model_path}")

    best_model = joblib.load(best_model_path)

    # 加载新数据
    new_data = pd.read_csv(new_data_path)
    print(f"新数据形状: {new_data.shape}")

    # 确保特征顺序与训练时一致
    if set(new_data.columns) != set(feature_names):
        print("警告: 新数据的特征与训练数据不完全一致")
        # 只保留训练时使用的特征
        new_data = new_data[[col for col in new_data.columns if col in feature_names]]

    # 预处理新数据
    X_new = new_data.copy()

    # 编码分类特征
    categorical_cols = X_new.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col in label_encoders:
            # 处理新数据中可能出现的未知类别
            known_categories = set(label_encoders[col].classes_)
            X_new[col] = X_new[col].apply(lambda x: x if x in known_categories else 'Unknown')
            X_new[col] = label_encoders[col].transform(X_new[col])

    # 标准化特征
    X_new = scaler.transform(X_new)

    # 预测
    predictions = best_model.predict(X_new)

    # 如果是分类问题且目标变量被编码过，则解码预测结果
    if problem_type == 'classification' and target_encoder is not None:
        predictions = target_encoder.inverse_transform(predictions)

    # 创建包含预测结果的数据框
    result_df = pd.DataFrame({
        '预测结果': predictions
    })

    # 保存预测结果
    output_path = os.path.join(model_dir, "predictions.csv")
    result_df.to_csv(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")

    return result_df


# 使用示例
if __name__ == "__main__":
    # 创建系统实例
    system = EnhancedModelComparisonSystem()

    # 运行完整分析
    # 请替换为您的CSV文件路径和目标列名
    file_path = "数据.csv"  # 替换为您的CSV文件路径
    target_column = "升力系数"  # 替换为您的目标列名

    # 运行分析
    results, cv_results = system.run_complete_analysis(file_path, target_column)

    # 打印结果摘要
    print("\n模型性能摘要:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print()

    # 示例：如何使用保存的模型进行预测
    print("\n如何使用保存的模型进行预测:")
    print(f"predictions = predict_with_saved_model('new_data.csv', '{system.output_dir}')")