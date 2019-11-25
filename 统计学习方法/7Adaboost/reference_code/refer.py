import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

import  pandas as pd
import  matplotlib.pyplot as plt
from sklearn.datasets import make_hastie_10_2
'''
Thanks for the elements of statistic 
this is a adaboost of svm by JXinyee
I just want to try
'''
#compute the error_rate
def error_rate(y,pred):
    return sum(y!= pred)/len(y)
#compute the base estimator error_rate
def initclf(X_train,y_train,X_test,y_test,clf):
    clf.fit(X_train,y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_err = error_rate(y_train_pred,y_train)
    test_err = error_rate(y_test_pred,y_test)
    return train_err,test_err

#adaboost al..th
'''
M is the times of boost
clf is the base estimator
'''
def adaboost(X_train,y_train,X_test,y_test,M,clf):
    w = np.ones(len(X_train))/len(X_train)
    #刚开始总的分类器都是0
    n_train = len(X_train)
    n_test = len(y_train)
    pred_train,pred_test  = list(np.zeros(n_train)),list(np.zeros(n_test))
    for i in range(M):
        w1 = w*n_train
        clf.fit(X_train, y_train,sample_weight = w1)
        y_train_i = clf.predict(X_train)
        y_test_i = clf.predict(X_test)

        # miss is 8.1(b) 中的计算分类误差率要乘以w的
        miss = [int(i) for i in (y_train_i != y_train)]

        # miss2 是8.5中y*G(m)
        miss1 = [x if x == 1 else -1 for x in miss]
        #要注意np.dot()也可以一个ndarry 一个列表相乘 这里计算分类误差率和alpha_m
        error_m =np.dot(w,miss)
        #print(error_m)
        alpha_m = 0.5 *np.log((1-error_m)/error_m)
        #更新权重
        w = np.multiply(w,np.exp([-alpha_m * x for x in miss1 ]))

        #ensemble
        pred_train = [sum(x) for x in zip(pred_train,[alpha_m * i for i in y_train_i ])]
        pred_test = [sum(x) for x in zip(pred_test,[alpha_m * i for i in y_test_i ])]
    pred_train,pred_test = np.sign(np.array(pred_train)),np.sign(np.array(pred_test))
    return error_rate(pred_train,y_train), error_rate(pred_test,y_test)

#plot error curve
def plot_error_rate(er_train,er_test):
    df_err =pd.DataFrame([er_train,er_test]).T
    df_err.columns = ["Train","Test"]
    plot1 = df_err.plot(linewidth =3,figsize =(8,6),
                        color = ["lightblue","darkblue"],grid = True)
    plot1.set_xlabel('Number of iterations', fontsize=12)
    plot1.set_xticklabels(range(0, 450, 50))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of iterations', fontsize=16)
    plt.axhline(y=er_test[0], linewidth=1, color='red', ls='dashed')

#just do it
if __name__ =='__main__':
    x,y  = make_hastie_10_2()
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    svm_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                      max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
    er_train,er_test= initclf(X_train,y_train,X_test,y_test,svm_clf)
    er_train,er_test= [er_train],[er_test]
    xrange =(10,410,10)
    for i in xrange:
        er_train_i,er_test_i = adaboost(X_train,y_train,X_test,y_test,i,svm_clf)
        er_train.append(er_train_i)
        er_test.append(er_test_i)
    plot_error_rate(er_train,er_test)
