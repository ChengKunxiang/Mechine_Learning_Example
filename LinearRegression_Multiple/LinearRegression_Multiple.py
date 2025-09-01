import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tecplot.tecutil.sv import PRINTDEBUG

data = pd.read_csv('数据.csv')
print(data.head())

fig = plt.figure(figsize = (20,10))
fig1 = plt.subplot(2, 4, 1)
plt.scatter(data.loc[:, "马赫数"],data.loc[:, "升力系数"])
fig2 = plt.subplot(2, 4, 2)
plt.scatter(data.loc[:, "攻角"],data.loc[:, "升力系数"])
fig3 = plt.subplot(2, 4, 3)
plt.scatter(data.loc[:, "模型雷诺数"],data.loc[:, "升力系数"])
fig4 = plt.subplot(2, 4, 4)
plt.scatter(data.loc[:, "缩比"],data.loc[:, "升力系数"])
fig5 = plt.subplot(2, 4, 5)
plt.scatter(data.loc[:, "壁温比"],data.loc[:, "升力系数"])
fig6 = plt.subplot(2, 4, 6)
plt.scatter(data.loc[:, "静温"],data.loc[:, "升力系数"])
fig7 = plt.subplot(2, 4, 7)
plt.scatter(data.loc[:, "壁温"],data.loc[:, "升力系数"])
plt.show()
y = data.loc[:, "升力系数"]
y = np.array(y)
y = y.reshape(-1, 1)
print("y = \n", y)

# 多因子计算
x_multi = data.drop(['升力系数'], axis=1)
# print(x_multi.head())
LR_multi = LinearRegression()
LR_multi.fit(x_multi, y)
MSE = mean_squared_error(y, LR_multi.predict(x_multi))
R2 = r2_score(y, LR_multi.predict(x_multi))

print("线性回归的斜率：",LR_multi.coef_)
print("线性回归的截率：",LR_multi.intercept_)
print("y的预测值：\n", LR_multi.predict(x_multi))
print("MSE:",MSE)
print("R2:",R2)
