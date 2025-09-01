import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns

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
X, y = make_regression(
    n_samples=1000,  # 样本数量
    n_features=5,    # 特征数量
    n_informative=5, # 有信息的特征数量
    noise=10.0,      # 噪声水平
    random_state=42  # 随机种子
)

# 转换为DataFrame（有助于特征命名和解释）
feature_names = [f'X{i+1}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['y'] = y

print(f"数据集形状: {X.shape}")
print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}]")

# 2. 数据探索和可视化
print("\n数据探索:")
print(df.describe())

# 可视化特征与目标变量的关系
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(feature_names):
    axes[i].scatter(df[feature], df['y'], alpha=0.6)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('y')
    axes[i].set_title(f'{feature} vs y')

# 移除多余的子图
for i in range(len(feature_names), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('feature_target_relationships.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 分割数据集 (90% 训练, 10% 测试)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 4. 可选：特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 创建并训练线性回归模型
print("\n正在训练线性回归模型...")
model = LinearRegression()

# 使用标准化后的数据或原始数据
# model.fit(X_train_scaled, y_train)
model.fit(X_train, y_train)

# 6. 获取模型系数和截距
coefficients = model.coef_
intercept = model.intercept_

print("\n回归方程:")
equation = f"y = {intercept:.4f}"
for i, coef in enumerate(coefficients):
    equation += f" + {coef:.4f}*{feature_names[i]}"
print(equation)

# 7. 模型预测
y_pred = model.predict(X_test)
# 如果使用标准化数据: y_pred = model.predict(X_test_scaled)

# 8. 模型评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n模型评估指标:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 9. 使用statsmodels进行更详细的回归分析
print("\n详细回归分析 (使用statsmodels):")
X_with_const = sm.add_constant(X_train)  # 添加常数项
sm_model = sm.OLS(y_train, X_with_const).fit()
print(sm_model.summary())

# 10. 可视化预测值 vs 实际值
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
plt.savefig('linear_regression_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. 可视化残差
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.grid(True, alpha=0.3)
plt.savefig('linear_regression_residuals.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. 残差的正态性检验
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True, alpha=0.3)
plt.savefig('residuals_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 13. 系数可视化
plt.figure(figsize=(10, 6))
coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
coefficients_df = coefficients_df.sort_values('Coefficient', key=abs, ascending=False)
plt.bar(coefficients_df['Feature'], coefficients_df['Coefficient'])
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Linear Regression Coefficients')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('linear_regression_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()

# 14. 保存模型
import joblib
joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # 保存标准化器
print("\n模型已保存为 'linear_regression_model.pkl'")
print("标准化器已保存为 'scaler.pkl'")

# 15. 显示前10个预测结果和实际值的对比
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Residual': residuals
})
print("\n前10个样本的预测结果:")
print(results_df.head(10).round(3))

# 16. 使用模型进行新预测的示例
print("\n使用模型进行新预测的示例:")
# 创建一些新数据
new_samples = np.random.rand(5, X.shape[1]) * 2 - 1  # 生成5个新样本
new_predictions = model.predict(new_samples)
for i, (sample, pred) in enumerate(zip(new_samples, new_predictions)):
    sample_str = " + ".join([f"{coef:.2f}*{val:.2f}" for coef, val in zip(coefficients, sample)])
    print(f"样本 {i+1}: {intercept:.2f} + {sample_str} = {pred:.4f}")

# 17. 可选：尝试正则化线性模型
print("\n尝试正则化线性模型:")
# Ridge回归 (L2正则化)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_pred)
print(f"Ridge回归 R²: {ridge_r2:.4f}")

# Lasso回归 (L1正则化)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_r2 = r2_score(y_test, lasso_pred)
print(f"Lasso回归 R²: {lasso_r2:.4f}")

# 比较系数
coefficients_comparison = pd.DataFrame({
    'Feature': feature_names,
    'OLS': model.coef_,
    'Ridge': ridge_model.coef_,
    'Lasso': lasso_model.coef_
})
print("\n系数比较:")
print(coefficients_comparison.round(4))