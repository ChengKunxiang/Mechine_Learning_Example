import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tecplot.tecutil.sv import PRINTDEBUG

data = pd.read_csv("数据.csv")
print(data)

x = data.loc[:,'x']
y = data.loc[:,'y']
plt.figure()
plt.scatter(x, y)
plt.show()

lr = LinearRegression()
lr.fit(x,y)

print("斜率为：",lr.coef_)
print("截率为：",lr.intercept_)
print("y的预测值：\n", lr.predict(y))

MSE = mean_squared_error(y, lr.predict(x))
R2 = r2_score(y, lr.predict(x))
plt.figure()
plt.plot(y, lr.predict(x))
plt.xlabel("real-value")
plt.ylabel("predicted-value")
plt.show()

print("MSE:",MSE)
print("R2:",R2)
