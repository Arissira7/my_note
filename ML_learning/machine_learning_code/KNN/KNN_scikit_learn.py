import numpy as np  # 导入numpy库，用于数值计算
from sklearn.neighbors import KNeighborsClassifier # 从sklearn库中导入K最近邻分类器
from matplotlib.colors import ListedColormap  # 导入自定义颜色映射
import matplotlib.pyplot as plt  # 导入matplotlib库中的pyplot模块，用于绘图

'''读入gauss数据集'''
data = np.loadtxt('gauss.txt', delimiter=',') # np.loadtxt()默认按行读取
x_train = data[:, :2]
y_train = data[:, 2]

'''可视化读取的gauss数据集'''
plt.figure() # 创建一个图形框
# 两个类别的散点
plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], c="blue", marker="o")
plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], c="red", marker="x")
# 设置坐标轴名称
plt.xlabel('X1')
plt.ylabel('X2')    
# 显示图形
plt.show()

# 设置步长
step = 0.02
# 设置网格边界
x_min, x_max = np.min(x_train[:, 0]) - 1, np.max(x_train[:, 0]) + 1
y_min, y_max = np.min(x_train[:, 1]) - 1, np.max(x_train[:, 1]) + 1
# 构造网格
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
grid_data = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)