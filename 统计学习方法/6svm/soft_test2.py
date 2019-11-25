import os
import numpy as np
import random as rnd
from matplotlib import pyplot as plt
class SVM(object):
    def __init__(self, max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001):
        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_quadratic
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon
        self.w=None
        self.b=None
    def fit(self, X, y):
        # Initialization
        n, features = X.shape[0], X.shape[1]
        alpha = np.zeros((n))
        self.b=0
        kernel = self.kernels[self.kernel_type]
        count=0
        for iter in range(self.max_iter):
            alpha_prev = np.copy(alpha)
            count+=1
            for j in range(0, n):
                i = self.get_rnd_int(0, n-1, j) # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                Kii=kernel(x_i,x_i)
                Kjj=kernel(x_j,x_j)
                Kij=kernel(x_i,x_j)
                eta=Kii+Kjj-2*Kij
                if eta == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/eta
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

                # #renew b
                # bi=-E_i-y_i*Kii*(alpha[i]-alpha_prime_i)-y_j*Kij*(alpha[j]-alpha_prime_j)+self.b
                # bj=-E_j-y_i*Kij*(alpha[i]-alpha_prime_i)-y_j*Kjj*(alpha[j]-alpha_prime_j)+self.b
                
                # if 0<alpha[i] and self.C>alpha[i]:
                #     self.b=bi
                # elif 0<alpha[j] and self.C>alpha[j]:
                #     self.b=bj
                # else:
                #     self.b=(bi+bj)/2

            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)
        print("b:",self.b)
        print("count:",count)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)
        print(alpha)
        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors
    def predict(self, X):
        return self.h(X, self.w, self.b)
    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)
    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha,y))
    # Prediction
    def gx(self, x, w, b):
        return np.sign(np.dot(w.T, x.T) + b).astype(int)
    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.gx(x_k, w, b) - y_k
    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))
    def get_rnd_int(self, a,b,z):
        i = z
        cnt=0
        while i == z and cnt<1000:
            i = rnd.randint(a,b)
            cnt=cnt+1
        return i
    # Define kernels
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)
    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)


def loadDataSet(fileName):
    #first column is label
    labels=[]
    datas=[]
    with open(fileName,'r') as fr:
        lines=fr.readlines()
        for line in lines:
            linestr=line.strip().split(',')
            labels.append(linestr[0])
            datas.append([linestr[1],linestr[2]])
        datas,labels=np.array(datas),np.array(labels)
        datas=datas.astype(float)
        labels=labels.astype(float)
    return datas,labels

def Plotdatas(X,y):
    plt.plot(X[:,0][y==0],X[:,1][y==0],"bs")
    plt.plot(X[:,0][y==1],X[:,1][y==1],"yo")
    plt.axis([0,5,0,6])
def PlotregressionLine(w,b,support_vector):
    x0=np.linspace(0,5,200)
    decision_boundary=-w[0]/w[1]*x0-b/w[1]

    margin=1/w[1]
    gutter_up=decision_boundary+margin
    gutter_down=decision_boundary-margin

    plt.scatter(support_vector[:,0],support_vector[:,1],s=180,facecolors='#FFAAAA')
    plt.plot(x0,decision_boundary,"k-",linewidth=2)
    plt.plot(x0,gutter_up,"k--",linewidth=2)
    plt.plot(x0,gutter_down,"k--",linewidth=2)

#利用sklearn
from sklearn.svm import SVC
def svm2():
    svm_clf = SVC(kernel="linear", C=float("inf"))
    svm_clf.fit(allDatas, alllabels)
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(0, 5, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)
def save_fig(fig_id,tight_layout=True):
    path=os.path.join(".","images",fig_id+".png")
    print("Saving figure",fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path,format='png',dpi=300)

if __name__=="__main__":
    allDatas,alllabels=loadDataSet("iris_data.txt")
    print(allDatas.shape)
    print(alllabels.shape)
    # plotScatter(allDatas,alllabels)
    # plt.show()
    svmclassifier=SVM( max_iter=10000, kernel_type='linear', C=2, epsilon=0.0001)
    n_support=svmclassifier.fit(allDatas,alllabels)
    print(svmclassifier.w)
    print(svmclassifier.b)
    print(n_support)
    print(len(n_support))
    plt.subplot(121)
    Plotdatas(allDatas,alllabels)
    PlotregressionLine(svmclassifier.w,svmclassifier.b,n_support)

    plt.subplot(122)
    Plotdatas(allDatas,alllabels)
    svm2()
    save_fig("result")
    plt.show()

