# 完整示例：从数据加载到预测
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# 1. 加载模型
model = xgb.XGBRegressor()
model.load_model("xgboost_regression_model.json")

# 2. 加载新数据
new_data = pd.read_csv("new_data.csv")

# 3. 数据预处理（假设与训练时相同）
# 例如，如果训练时进行了特征缩放：
scaler = StandardScaler()
# 注意：应该使用训练时的scaler对象，这里仅为示例
# 在实际应用中，您应该保存并加载训练时使用的scaler
new_data_scaled = scaler.fit_transform(new_data)

# 4. 进行预测
predictions = model.predict(new_data_scaled)

# 5. 处理预测结果
results = pd.DataFrame({
    '原始数据索引': new_data.index,
    '预测值': predictions
})

# 6. 保存结果
results.to_csv("predictions.csv", index=False)
print("预测完成并已保存结果!")