import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 设置中文字体（避免警告）
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
try:
    chinese_fonts = [
        f.name for f in font_manager.fontManager.ttflist
        if any(x in f.name for x in ['宋体', '黑体', '微软雅黑', 'SimHei', 'Microsoft YaHei'])
    ]
    if chinese_fonts:
        plt.rcParams['font.sans-serif'] = [chinese_fonts[0], 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        print("未找到中文字体，将使用英文标签")
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('seaborn-v0_8')


def load_and_predict(model_path, new_data_path, has_target=False):
    """
    加载已保存的梯度提升模型并对新数据进行预测

    参数:
    model_path: 模型文件路径
    new_data_path: 新数据文件路径
    has_target: 新数据是否包含目标变量（用于评估）
    """
    # 1. 加载已保存的梯度提升模型
    try:
        model = joblib.load(model_path)
        print("模型加载成功!")
        print(f"模型类型: {type(model).__name__}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

    # 2. 加载新数据
    try:
        # 假设新数据是CSV格式，您可以根据实际情况调整
        new_data = pd.read_csv(new_data_path)
        print(f"新数据加载成功! 数据形状: {new_data.shape}")
        print(f"数据列名: {list(new_data.columns)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

    # 3. 准备预测数据
    if has_target:
        # 如果数据包含目标变量，分离特征和目标
        if 'target' in new_data.columns:
            X_new = new_data.drop('target', axis=1)
            y_true = new_data['target']
        else:
            # 尝试找到目标列（可能是其他名称）
            # 这里假设最后一列是目标变量
            X_new = new_data.iloc[:, :-1]
            y_true = new_data.iloc[:, -1]
            print(f"假设目标变量是: {new_data.columns[-1]}")
    else:
        X_new = new_data
        y_true = None

    # 4. 确保特征顺序与训练时一致
    # 注意：在实际应用中，您可能需要确保特征顺序和预处理与训练时一致
    try:
        # 如果模型有feature_names_in_属性，确保特征顺序一致
        if hasattr(model, 'feature_names_in_'):
            X_new = X_new[model.feature_names_in_]
            print("已按训练时的特征顺序重新排列数据")
    except Exception as e:
        print(f"特征顺序调整时出错: {e}")

    # 5. 使用模型进行预测
    predictions = model.predict(X_new)

    # 6. 如果有真实值，评估预测性能
    if y_true is not None:
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        print(f"\n模型评估指标:")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"决定系数 (R²): {r2:.4f}")

        # 可视化预测结果 vs 实际值
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, predictions, alpha=0.6, s=50)

        # 绘制理想拟合线 (y=x)
        max_val = max(np.max(y_true), np.max(predictions))
        min_val = min(np.min(y_true), np.min(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')

        # 添加回归线
        z = np.polyfit(y_true, predictions, 1)
        p = np.poly1d(z)
        plt.plot(y_true, p(y_true), "g-", lw=2, label=f'Regression Line (y={z[0]:.2f}x+{z[1]:.2f})')

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values (New Data)')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

        # 在图上添加评估指标文本
        textstr = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig('new_data_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 7. 输出预测结果
    print(f"\n预测结果 (前10个样本):")
    for i, pred in enumerate(predictions[:10]):
        print(f"样本 {i + 1}: {pred:.4f}")

    # 8. 将预测结果保存到文件
    result_df = pd.DataFrame({
        '预测值': predictions
    })

    # 如果新数据有索引，保留索引
    if hasattr(new_data, 'index'):
        result_df.index = new_data.index

    result_df.to_csv('predictions.csv', index=True)
    print(f"\n预测结果已保存到 'predictions.csv'")

    # 9. 可视化预测值分布
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Predicted Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Values')
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    return predictions


# 使用示例
if __name__ == "__main__":
    # 指定模型文件路径和新数据文件路径
    model_path = "gradient_boosting_model.pkl"  # 替换为您的模型文件路径
    new_data_path = "测试.csv"  # 替换为您的新数据文件路径

    # 进行预测
    # 如果新数据包含目标变量，设置 has_target=True
    predictions = load_and_predict(model_path, new_data_path, has_target=False)

    # 如果需要，可以进一步处理预测结果
    if predictions is not None:
        print(f"\n预测完成! 共预测了 {len(predictions)} 个样本")
        print(f"预测值范围: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
        print(f"预测值平均值: {np.mean(predictions):.4f}")
        print(f"预测值标准差: {np.std(predictions):.4f}")