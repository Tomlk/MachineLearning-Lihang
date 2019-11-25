import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

X, Y = make_moons(200, noise=0.20,random_state=0)
#前面加一列
X=np.insert(X, 0, 1, axis=1)
#将Y变成列向量
Y=Y.reshape(200,1)

def plot_decision_boundary(pred_func):
    # 设置图像边界的最大最小值，并加了0.5的边距
    x_min, x_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    y_min, y_max = X[:, 2].min() - .5, X[:, 2].max() + .5
    h = 0.01
    #以0.01为间隔，对整幅图像进行网格式划分，返回值为两个相同规格的矩阵，规格图像网格划分后对应的行数与列数
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #预测并设置Z的维度
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 画等高线，X和Y均为2维，与Z的规格相同，或者为一维也可，但要求len(X)=Z的列数, len(Y)=Z的行数
    #https://matplotlib.org/tutorials/colors/colormaps.html 此网站内置颜色
    plt.contourf(xx, yy, Z,cmap=plt.cm.RdBu)
    #画点，注意颜色这里不能为二维的
    plt.scatter(X[:, 1], X[:, 2], c=Y.reshape(200) ,cmap=ListedColormap(['#FF0000', '#0000FF']),alpha=0.5)

def calculate_loss(theta):
    z = X.dot(theta)
    probs = sigmoid(z)
    data_loss = np.power((probs-Y), 2)
    return 0.5*np.sum(data_loss)

def predict(theta, x):
    #插入一列
    x=np.insert(x, 0, 1, axis=1)
    z = x.dot(theta)
    probs = sigmoid(z)
    #阈值为0.5，0.5以上的标记置为1，否则置为0
    return np.array([int(item>0.5) for item in probs])

def sigmoid(x):
  return 1/(1+np.exp(-x))

def build_model(iterations=3):
    theta = np.zeros(3).reshape(3,1)
    for i in range (iterations):
        #计算h(x)
        hypothesis=sigmoid(X.dot(theta))

        #牛顿方法,向量的一阶导数
        first_derivative = X.T.dot(Y - hypothesis )
        #对角矩阵
        diag=np.diagflat( hypothesis*( hypothesis-np.ones(200).reshape(200,1) ) )
        #Hessain矩阵
        H=np.dot(X.T.dot(diag),X)
        #更新theta
        theta-=np.linalg.solve(H,first_derivative)

        # BGD算法，学习率为0.01
        # alpha=0.01
        # theta+= alpha*X.T.dot(Y-hypothesis)

        #输出损失函数
        print(calculate_loss(theta))
    return theta

#sklearn下的LogisticRegression，牛顿方法进行优化
clf=LogisticRegression(solver='newton-cg');
#这里也将Y变成一维的
clf.fit(X[:,1:], Y.ravel())
plt.subplot(122)
plot_decision_boundary(lambda x: clf.predict(x))
plt.title('logistic regression in sklearn')
plt.tight_layout()

# plt.show()

#自己的模型
theta = build_model()
plt.subplot(121)
plot_decision_boundary(lambda x:predict(theta, x))
plt.title("logistic regression with our method")
plt.tight_layout()
plt.show()
