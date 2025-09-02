import pandas as pd
import numpy as np
import joblib
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class MultiTargetModelPredictor:
    def __init__(self, model_dir):
        """
        初始化多目标预测器

        参数:
        model_dir (str): 包含保存的模型和预处理对象的目录路径
        """
        self.model_dir = model_dir
        self.preprocessing = None
        self.best_models = {}  # 存储每个目标变量的最佳模型
        self.feature_names = None
        self.target_names = None
        self.problem_types = {}  # 存储每个目标变量的问题类型

        # 创建输出目录（在程序相同目录下）
        self.output_dir = os.path.join(os.getcwd(), "multi_target_prediction_output")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"输出目录已创建: {self.output_dir}")

        # 加载预处理对象和模型
        self.load_models()

    def load_models(self):
        """加载预处理对象和所有目标变量的最佳模型"""
        # 加载预处理对象
        preprocessing_path = os.path.join(self.model_dir, "preprocessing.pkl")
        if not os.path.exists(preprocessing_path):
            raise FileNotFoundError(f"预处理文件未找到: {preprocessing_path}")

        self.preprocessing = joblib.load(preprocessing_path)
        print("预处理对象加载成功")

        # 获取特征名称和目标变量名称
        self.feature_names = self.preprocessing['feature_names']
        self.target_names = self.preprocessing['target_names']
        self.problem_types = self.preprocessing['problem_types']

        print(f"目标变量: {self.target_names}")
        print(f"问题类型: {self.problem_types}")

        # 加载每个目标变量的最佳模型
        for target in self.target_names:
            best_model_path = os.path.join(self.model_dir, f"best_model_{target}.pkl")
            if not os.path.exists(best_model_path):
                raise FileNotFoundError(f"最佳模型文件未找到: {best_model_path}")

            self.best_models[target] = joblib.load(best_model_path)
            print(f"目标变量 '{target}' 的最佳模型加载成功")

    def preprocess_new_data(self, new_data):
        """
        预处理新数据，使其与训练数据格式一致

        参数:
        new_data (DataFrame): 新数据

        返回:
        预处理后的数据
        """
        # 创建数据副本
        X_new = new_data.copy()

        # 确保特征顺序与训练时一致
        missing_features = set(self.feature_names) - set(X_new.columns)
        extra_features = set(X_new.columns) - set(self.feature_names)

        if missing_features:
            print(f"警告: 新数据缺少以下特征: {missing_features}")
            # 为缺失的特征添加默认值（0）
            for feature in missing_features:
                X_new[feature] = 0

        if extra_features:
            print(f"警告: 新数据包含以下额外特征: {extra_features}")
            # 移除训练时未使用的特征
            X_new = X_new[[col for col in X_new.columns if col in self.feature_names]]

        # 确保特征顺序与训练时一致
        X_new = X_new[self.feature_names]

        # 处理分类特征
        label_encoders = self.preprocessing['label_encoders']
        categorical_cols = X_new.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if col in label_encoders:
                # 处理新数据中可能出现的未知类别
                known_categories = set(label_encoders[col].classes_)
                X_new[col] = X_new[col].apply(
                    lambda x: x if str(x) in known_categories else 'Unknown'
                )
                try:
                    X_new[col] = label_encoders[col].transform(X_new[col].astype(str))
                except ValueError as e:
                    print(f"编码特征 '{col}' 时出错: {e}")
                    # 对于无法编码的值，使用最常见的类别
                    most_common = label_encoders[col].transform(
                        [label_encoders[col].classes_[0]]
                    )[0]
                    X_new[col] = most_common

        # 标准化特征
        X_new = self.preprocessing['scaler'].transform(X_new)

        return X_new

    def predict(self, new_data_path, include_original_data=True,
                include_probabilities=False, output_format='csv'):
        """
        使用保存的模型进行预测，并导出结果

        参数:
        new_data_path (str): 新数据的CSV文件路径
        include_original_data (bool): 是否在输出中包含原始数据
        include_probabilities (bool): 是否包含预测概率（仅分类问题）
        output_format (str): 输出格式，支持'csv'和'excel'

        返回:
        预测结果DataFrame
        """
        # 加载新数据
        new_data = pd.read_csv(new_data_path)
        print(f"新数据形状: {new_data.shape}")
        print("\n新数据前5行:")
        print(new_data.head())

        # 预处理新数据
        X_processed = self.preprocess_new_data(new_data)

        # 为每个目标变量进行预测
        predictions = {}
        probabilities = {}

        for target in self.target_names:
            print(f"\n为目标变量 '{target}' 进行预测...")

            # 预测
            y_pred = self.best_models[target].predict(X_processed)

            # 如果是分类问题且目标变量被编码过，则解码预测结果
            if (self.problem_types[target] == 'classification' and
                    target in self.preprocessing['target_encoders']):
                y_pred = self.preprocessing['target_encoders'][target].inverse_transform(y_pred)

            predictions[target] = y_pred

            # 对于分类问题，获取预测概率（如果支持）
            if (self.problem_types[target] == 'classification' and
                    include_probabilities and
                    hasattr(self.best_models[target], "predict_proba")):

                y_proba = self.best_models[target].predict_proba(X_processed)

                # 获取类别名称
                if target in self.preprocessing['target_encoders']:
                    class_names = self.preprocessing['target_encoders'][target].classes_
                else:
                    class_names = [f"类别_{i}" for i in range(y_proba.shape[1])]

                # 创建概率数据框
                proba_dict = {}
                for i, name in enumerate(class_names):
                    proba_dict[f"{target}_概率_{name}"] = y_proba[:, i]

                probabilities[target] = pd.DataFrame(proba_dict)

        # 创建结果数据框
        result_dfs = []

        # 添加原始数据
        if include_original_data:
            result_dfs.append(new_data.reset_index(drop=True))

        # 添加预测结果
        predictions_df = pd.DataFrame(predictions)
        predictions_df.columns = [f'预测_{col}' for col in predictions_df.columns]
        result_dfs.append(predictions_df)

        # 添加预测概率
        if include_probabilities and probabilities:
            for target, proba_df in probabilities.items():
                result_dfs.append(proba_df)

        # 合并所有数据框
        final_result = pd.concat(result_dfs, axis=1)

        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_filename = os.path.splitext(os.path.basename(new_data_path))[0]

        if output_format == 'csv':
            output_filename = f"multi_target_predictions_{data_filename}_{timestamp}.csv"
            output_path = os.path.join(self.output_dir, output_filename)
            final_result.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"预测结果已保存到: {output_path}")

            # 同时保存一个不带时间戳的版本（方便后续更新）
            latest_output_path = os.path.join(self.output_dir, f"latest_multi_target_predictions_{data_filename}.csv")
            final_result.to_csv(latest_output_path, index=False, encoding='utf-8-sig')
            print(f"最新预测结果已保存到: {latest_output_path}")

        elif output_format == 'excel':
            output_filename = f"multi_target_predictions_{data_filename}_{timestamp}.xlsx"
            output_path = os.path.join(self.output_dir, output_filename)
            final_result.to_excel(output_path, index=False)
            print(f"预测结果已保存到: {output_path}")

            # 同时保存一个不带时间戳的版本
            latest_output_path = os.path.join(self.output_dir, f"latest_multi_target_predictions_{data_filename}.xlsx")
            final_result.to_excel(latest_output_path, index=False)
            print(f"最新预测结果已保存到: {latest_output_path}")

        else:
            raise ValueError(f"不支持的输出格式: {output_format}")

        # 打印预测结果摘要
        self.generate_prediction_summary(predictions, data_filename)

        return final_result

    def generate_prediction_summary(self, predictions, data_filename):
        """
        生成预测结果摘要

        参数:
        predictions (dict): 包含预测结果的字典
        data_filename (str): 数据文件名（用于生成报告文件名）
        """
        summary_content = []

        # 添加基本信息
        summary_content.append("多目标变量预测结果摘要")
        summary_content.append("=" * 50)
        summary_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_content.append(f"数据文件: {data_filename}")
        summary_content.append(f"目标变量数量: {len(self.target_names)}")
        summary_content.append("")

        # 添加每个目标变量的预测结果统计
        for target in self.target_names:
            pred_values = predictions[target]

            summary_content.append(f"目标变量: {target}")
            summary_content.append(f"问题类型: {self.problem_types[target]}")

            if self.problem_types[target] == 'classification':
                # 分类问题的统计
                unique, counts = np.unique(pred_values, return_counts=True)
                summary_content.append("预测结果分布:")
                for value, count in zip(unique, counts):
                    percentage = count / len(pred_values) * 100
                    summary_content.append(f"  {value}: {count} ({percentage:.2f}%)")
            else:
                # 回归问题的统计
                summary_content.append("预测结果统计:")
                summary_content.append(f"  平均值: {np.mean(pred_values):.4f}")
                summary_content.append(f"  最小值: {np.min(pred_values):.4f}")
                summary_content.append(f"  最大值: {np.max(pred_values):.4f}")
                summary_content.append(f"  标准差: {np.std(pred_values):.4f}")
                summary_content.append(f"  中位数: {np.median(pred_values):.4f}")

            summary_content.append("")

        # 保存摘要
        summary_path = os.path.join(self.output_dir, f"prediction_summary_{data_filename}.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_content))

        print(f"预测摘要已保存到: {summary_path}")

    def batch_predict(self, data_dir, file_pattern="*.csv", **kwargs):
        """
        批量预测多个文件

        参数:
        data_dir (str): 包含数据文件的目录路径
        file_pattern (str): 文件匹配模式
        **kwargs: 传递给predict方法的其他参数

        返回:
        所有预测结果的字典
        """
        import glob

        # 查找所有匹配的文件
        file_paths = glob.glob(os.path.join(data_dir, file_pattern))

        if not file_paths:
            print(f"在 {data_dir} 中未找到匹配 {file_pattern} 的文件")
            return {}

        print(f"找到 {len(file_paths)} 个文件进行批量预测")

        # 预测每个文件
        all_results = {}
        for file_path in file_paths:
            print(f"\n处理文件: {os.path.basename(file_path)}")
            try:
                # 进行预测
                result = self.predict(file_path, **kwargs)
                all_results[os.path.basename(file_path)] = result

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                import traceback
                traceback.print_exc()

        return all_results


# 使用示例
if __name__ == "__main__":
    # 设置模型目录和新数据路径
    model_directory = "output/models"  # 替换为您的模型目录路径
    new_data_path = "测试.csv"  # 替换为您的新数据路径

    # 创建预测器实例
    try:
        predictor = MultiTargetModelPredictor(model_directory)

        # 进行预测并导出结果
        predictions = predictor.predict(
            new_data_path,
            include_original_data=True,
            include_probabilities=True,
            output_format='csv'
        )

        print("\n预测完成！")
        print(f"所有输出文件已保存到: {predictor.output_dir}")

        # 批量预测示例（如果需要）
        # data_directory = "batch_data"  # 包含多个CSV文件的目录
        # batch_results = predictor.batch_predict(data_directory)

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()