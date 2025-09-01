import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

plt.rcParams['font.family'] = 'SimHei'

# 设置随机种子以确保结果可重现
np.random.seed(42)
plt.style.use('seaborn-v0_8')

# 1. 创建示例数据或加载您的数据
# 如果您有自己的数据，可以使用 pd.read_csv() 加载
# 这里我们创建一些示例数据
n_samples = 1000
n_features = 5

X = np.random.rand(n_samples, n_features) * 10
# 创建目标变量 y，这里使用线性组合加上一些噪声
true_coefficients = np.array([2.5, -1.3, 0.7, 3.2, -0.5])
y = np.dot(X, true_coefficients) + np.random.randn(n_samples) * 2

# 转换为DataFrame（可选，有助于特征命名）
feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 2. 分割数据集 (90% 训练, 10% 测试)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 3. 创建并训练XGBoost回归模型
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    early_stopping_rounds=10
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

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
plt.title("特征重要性")
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 可视化预测值 vs 实际值
plt.figure(figsize=(10, 8))

# 绘制散点图
plt.scatter(y_test, y_pred, alpha=0.6, s=50)

# 绘制理想拟合线 (y=x)
max_val = max(np.max(y_test), np.max(y_pred))
min_val = min(np.min(y_test), np.min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想拟合线')

# 添加回归线
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "g-", lw=2, label=f'回归线 (y={z[0]:.2f}x+{z[1]:.2f})')

plt.xlabel('实际值', fontsize=12)
plt.ylabel('预测值', fontsize=12)
plt.title('预测值 vs 实际值', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# 在图上添加评估指标文本
textstr = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 可视化残差
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差分析')
plt.grid(True, alpha=0.3)
plt.savefig('residuals.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. 保存模型
model.save_model('xgboost_regression_model.json')
print("\n模型已保存为 'xgboost_regression_model.json'")

# 11. 可选：显示前10个预测结果和实际值的对比
results_df = pd.DataFrame({
    '实际值': y_test,
    '预测值': y_pred,
    '残差': residuals
})
print("\n前10个样本的预测结果:")
print(results_df.head(10).round(3))