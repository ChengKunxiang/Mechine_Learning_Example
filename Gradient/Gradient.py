import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import make_regression

# 设置随机种子以确保结果可重现
np.random.seed(42)

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

# 1. 创建示例回归数据集
print("正在生成示例数据...")
data = pd.read_csv('数据.csv')
X = data.drop('升力系数', axis=1)
y = data['升力系数']
feature_names = X.columns.tolist()

# 转换为DataFrame（可选，有助于特征命名）
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"数据集形状: {X.shape}")
print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}]")

# 2. 分割数据集 (90% 训练, 10% 测试)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 3. 创建并训练梯度提升回归模型
print("\n正在训练梯度提升模型...")
model = GradientBoostingRegressor(
    n_estimators=100,  # 树的数量
    learning_rate=0.1,  # 学习率
    max_depth=3,  # 树的最大深度
    min_samples_split=2,  # 分裂内部节点所需的最小样本数
    min_samples_leaf=1,  # 叶节点所需的最小样本数
    subsample=1.0,  # 用于拟合个体基础学习器的样本比例
    max_features=None,  # 每个树考虑的特征数量
    random_state=42  # 随机种子
)

# 训练模型
model.fit(X_train, y_train)

# 4. 模型预测
y_pred = model.predict(X_test)

# 5. 模型评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n模型评估指标:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 6. 特征重要性
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

print("\n特征重要性:")
for i in sorted_idx:
    print(f"{feature_names[i]}: {feature_importance[i]:.4f}")

# 7. 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance[sorted_idx])
plt.xticks(range(len(feature_importance)), np.array(feature_names)[sorted_idx], rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig('gradient_boosting_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 可视化预测值 vs 实际值
plt.figure(figsize=(10, 8))

# 绘制散点图
plt.scatter(y_test, y_pred, alpha=0.6, s=50)

# 绘制理想拟合线 (y=x)
max_val = max(np.max(y_test), np.max(y_pred))
min_val = min(np.min(y_test), np.min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')

# 添加回归线
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "g-", lw=2, label=f'Regression Line (y={z[0]:.2f}x+{z[1]:.2f})')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# 在图上添加评估指标文本
textstr = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('gradient_boosting_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 可视化残差
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.grid(True, alpha=0.3)
plt.savefig('gradient_boosting_residuals.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. 可视化训练过程中的损失下降
train_score = np.zeros((model.n_estimators,), dtype=np.float64)
test_score = np.zeros((model.n_estimators,), dtype=np.float64)

for i, y_pred in enumerate(model.staged_predict(X_train)):
    train_score[i] = mean_squared_error(y_train, y_pred)

for i, y_pred in enumerate(model.staged_predict(X_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(model.n_estimators) + 1, train_score, 'b-', label='Training Set MSE')
plt.plot(np.arange(model.n_estimators) + 1, test_score, 'r-', label='Test Set MSE')
plt.xlabel('Boosting Iterations')
plt.ylabel('MSE')
plt.title('Training and Test Error During Boosting')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig('gradient_boosting_training_error.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. 保存模型（可选）
import joblib

joblib.dump(model, 'gradient_boosting_model.pkl')
print("\n模型已保存为 'gradient_boosting_model.pkl'")

# 12. 显示前10个预测结果和实际值的对比
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Residual': residuals
})
print("\n前10个样本的预测结果:")
print(results_df.head(10).round(3))

# 13. 使用模型进行新预测的示例
print("\n使用模型进行新预测的示例:")
# 创建一些新数据
new_samples = np.random.rand(5, X.shape[1]) * 2 - 1  # 生成5个新样本
new_predictions = model.predict(new_samples)
for i, pred in enumerate(new_predictions):
    print(f"样本 {i + 1} 预测值: {pred:.4f}")