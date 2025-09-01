import numpy as np
import  pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from tecplot.tecutil.sv import PRINTDEBUG

matplotlib.rcParams['font.family'] = 'SimHei'
data = pd.read_csv("数据.csv")
print(data)

x = data.loc[:,'升力系数']
y = data.loc[:,'阻力系数']
fig1 = plt.figure()
plt.xlabel('升力系数')
plt.ylabel('阻力系数')
plt.title('升力系数 vs 阻力系数')
plt.scatter(x, y)
plt.show()

mask = data.loc[:,'有无尾支撑'] == 'yes'
print(mask)

fig2 = plt.figure()
point_yes = plt.scatter(data.loc[:, '升力系数'][mask], data.loc[:, '阻力系数'][mask])
point_no = plt.scatter(data.loc[:, '升力系数'][~mask], data.loc[:, '阻力系数'][~mask])
plt.legend((point_yes, point_no), ('yes', 'no'))
plt.xlabel('升力系数')
plt.ylabel('阻力系数')
plt.title('升力系数 vs 阻力系数')
plt.show()

X = data.drop(['有无尾支撑'],axis=1)
y = data.loc[:,'有无尾支撑']
X1 = data.loc[:,'升力系数']
X2 = data.loc[:,'阻力系数']

# 一阶
LR = LogisticRegression()
LR.fit(X, y)
y_pred = LR.predict(X)
accuracy = accuracy_score(y, y_pred)

print('accuracy:', accuracy)

# 二阶
X1_2 = X1*X1
X2_2 = X2*X2
X1_X2 = X1*X2
X_new = {'X1':X1,'X2':X2,'X1_2':X1_2,'X2_2':X2_2,'X1_X2':X1_X2}
X_new = pd.DataFrame(X_new)
print(X_new)

LR2 = LogisticRegression()
LR2.fit(X_new, y)
y_pred_2 = LR2.predict(X_new)
accuracy_2 = accuracy_score(y, y_pred_2)
print('accuracy:', accuracy_2)

theta0 = LR2.intercept_
theta1,  theta2, theta3,  theta4, theta5 = LR2.coef_[0][0],LR2.coef_[0][1],LR2.coef_[0][2],LR2.coef_[0][3],LR2.coef_[0][4]
print('theta0:', theta0)
print('theta1:', theta1)
print('theta2:', theta2)
print('theta3:', theta3)
print('theta4:', theta4)
print('theta5:', theta5)