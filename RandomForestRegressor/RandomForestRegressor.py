import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class RandomForestModel:
    def __init__(self, problem_type='classification', n_estimators=100, random_state=42):
        """
        初始化随机森林模型

        参数:
        problem_type (str): 问题类型，'classification' 或 'regression'
        n_estimators (int): 森林中树的数量
        random_state (int): 随机种子，用于重现结果
        """
        self.problem_type = problem_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False

    def preprocess_data(self, X, y=None, training=True):
        """
        数据预处理：处理缺失值、标准化数值特征、编码分类特征

        参数:
        X: 特征数据
        y: 目标数据（可选）
        training: 是否为训练阶段

        返回:
        处理后的特征和目标数据
        """
        X_processed = X.copy()

        # 处理缺失值
        if isinstance(X_processed, pd.DataFrame):
            # 对于数值列，用中位数填充缺失值
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if training:
                    fill_value = X_processed[col].median()
                else:
                    fill_value = getattr(self, f'median_{col}', 0)
                X_processed[col].fillna(fill_value, inplace=True)

            # 对于分类列，用众数填充缺失值并编码
            categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if training:
                    fill_value = X_processed[col].mode()[0] if not X_processed[col].mode().empty else 'Unknown'
                    self.label_encoders[col] = LabelEncoder()
                    X_processed[col].fillna(fill_value, inplace=True)
                    X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col])
                else:
                    fill_value = getattr(self, f'mode_{col}', 'Unknown')
                    X_processed[col].fillna(fill_value, inplace=True)
                    X_processed[col] = self.label_encoders[col].transform(X_processed[col])

        # 标准化数值特征
        if training:
            self.scaler.fit(X_processed)
        X_processed = self.scaler.transform(X_processed)

        # 处理目标变量（如果是分类问题且需要编码）
        y_processed = y
        if y is not None and self.problem_type == 'classification' and not np.issubdtype(
                type(y.iloc[0]) if hasattr(y, 'iloc') else type(y[0]), np.number):
            if training:
                self.target_encoder = LabelEncoder()
                y_processed = self.target_encoder.fit_transform(y)
            else:
                y_processed = self.target_encoder.transform(y)

        return (X_processed, y_processed) if y is not None else X_processed

    def train(self, X, y):
        """
        训练随机森林模型

        参数:
        X: 特征数据
        y: 目标数据
        """
        # 数据预处理
        X_processed, y_processed = self.preprocess_data(X, y, training=True)

        # 创建并训练模型
        if self.problem_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1  # 使用所有可用的CPU核心
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )

        self.model.fit(X_processed, y_processed)
        self.is_trained = True

        # 计算训练集上的性能
        train_pred = self.model.predict(X_processed)
        if self.problem_type == 'classification':
            train_accuracy = accuracy_score(y_processed, train_pred)
            print(f"训练集准确率: {train_accuracy:.4f}")
        else:
            train_mse = mean_squared_error(y_processed, train_pred)
            print(f"训练集均方误差: {train_mse:.4f}")

    def predict(self, X):
        """
        使用训练好的模型进行预测

        参数:
        X: 特征数据

        返回:
        预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")

        X_processed = self.preprocess_data(X, training=False)
        predictions = self.model.predict(X_processed)

        # 如果是分类问题且目标变量被编码过，则解码预测结果
        if self.problem_type == 'classification' and hasattr(self, 'target_encoder'):
            predictions = self.target_encoder.inverse_transform(predictions)

        return predictions

    def evaluate(self, X_test, y_test):
        """
        评估模型性能

        参数:
        X_test: 测试集特征
        y_test: 测试集真实值

        返回:
        评估指标
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")

        # 数据预处理
        X_processed, y_processed = self.preprocess_data(X_test, y_test, training=False)

        # 预测
        predictions = self.model.predict(X_processed)

        # 计算性能指标
        if self.problem_type == 'classification':
            accuracy = accuracy_score(y_processed, predictions)
            print(f"测试集准确率: {accuracy:.4f}")
            print("\n分类报告:")
            print(classification_report(y_processed, predictions))

            # 绘制混淆矩阵
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_processed, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('混淆矩阵')
            plt.ylabel('真实标签')
            plt.xlabel('预测标签')
            plt.show()

            return accuracy
        else:
            mse = mean_squared_error(y_processed, predictions)
            rmse = np.sqrt(mse)
            print(f"测试集均方误差(MSE): {mse:.4f}")
            print(f"测试集均方根误差(RMSE): {rmse:.4f}")

            # 绘制预测值与真实值的散点图
            plt.figure(figsize=(8, 6))
            plt.scatter(y_processed, predictions, alpha=0.5)
            plt.plot([y_processed.min(), y_processed.max()],
                     [y_processed.min(), y_processed.max()], 'r--', lw=2)
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.title('预测值 vs 真实值')
            plt.show()

            return mse

    def feature_importance(self, feature_names=None):
        """
        显示特征重要性

        参数:
        feature_names: 特征名称列表
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")

        if feature_names is None and hasattr(self, 'feature_names'):
            feature_names = self.feature_names

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("特征重要性排名:")
        for i, idx in enumerate(indices):
            if feature_names is not None:
                print(f"{i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")
            else:
                print(f"{i + 1}. 特征 {idx}: {importances[idx]:.4f}")

        # 绘制特征重要性图表
        plt.figure(figsize=(10, 6))
        if feature_names is not None:
            sorted_names = [feature_names[i] for i in indices]
            plt.barh(range(len(importances)), importances[indices], align='center')
            plt.yticks(range(len(importances)), sorted_names)
        else:
            plt.barh(range(len(importances)), importances[indices], align='center')
        plt.xlabel('特征重要性')
        plt.title('随机森林特征重要性')
        plt.gca().invert_yaxis()  # 最重要的特征显示在顶部
        plt.show()


# 示例使用方式
if __name__ == "__main__":
    # 示例1：分类问题
    print("=== 分类问题示例 ===")

    # 加载数据（这里使用鸢尾花数据集作为示例）
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建并训练模型
    rf_model = RandomForestModel(problem_type='classification', n_estimators=100, random_state=42)
    rf_model.train(X_train, y_train)

    # 预测
    predictions = rf_model.predict(X_test)
    print("前10个预测结果:", predictions[:10])

    # 评估模型
    rf_model.evaluate(X_test, y_test)

    # 显示特征重要性
    rf_model.feature_importance(feature_names=iris.feature_names)

    # 示例2：回归问题
    print("\n=== 回归问题示例 ===")

    # 加载数据（这里使用糖尿病数据集作为示例）
    from sklearn.datasets import load_diabetes

    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建并训练模型
    rf_model = RandomForestModel(problem_type='regression', n_estimators=100, random_state=42)
    rf_model.train(X_train, y_train)

    # 预测
    predictions = rf_model.predict(X_test)
    print("前10个预测结果:", predictions[:10])

    # 评估模型
    rf_model.evaluate(X_test, y_test)

    # 显示特征重要性
    rf_model.feature_importance(feature_names=diabetes.feature_names)