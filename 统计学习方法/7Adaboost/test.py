#Adaboost 以SVM 算法为例
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
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
class Adaboost(object):
    def __init__(self,M):
        self.X=None
        self.y=None
        self.alphas=[]
        self.classifiers=[]
        #self.classifierName=classifierName
        #self.classifier=None
        self.M=M
        pass
    def getClassifier(self,classifiername):
        if classifiername=="svm":
            from sklearn.svm import SVC
            #默认用rfb核函数
            rbf_kernel_svm_clf=Pipeline([
                ("scaler",StandardScaler()),
                ("svm_clf",SVC(kernel="rbf",gamma=5,C=0.001))
            ])
            return rbf_kernel_svm_clf
        elif classifiername=="knn":
            from sklearn.neighbors import KNeighborsClassifier
            #默认最近邻
            knn_clf=Pipeline([
                ("scaler",StandardScaler()),
                ("knn_clf",KNeighborsClassifier(n_neighbors=1))
            ])
            return knn_clf
        #logistc

        #decision tree

        #etc
        else:
            return None




    def getZm(self,D,alpha,classifier):
        Zm=0
        for j in range(len(self.X)):
            W_mj=D[j]
            Gmxj=classifier.predict([self.X[j]])[0]
            yj=self.y[j]
            Zm+=W_mj*np.exp(-alpha*yj*Gmxj)
        return Zm
    def f(self,xs):
        result=0
        for m in range(self.M):
            classifier=pickle.loads(self.classifiers[m])
            result+=self.alphas[m]*classifier.predict(xs)
        return result
    def G(self,xs):
        return np.sign(self.f(xs))

    def fit(self,classifiername,train_group,train_labels):
        self.X=train_group
        self.y=train_labels
        X=self.X
        y=self.y
        assert len(X)==len(y)
        n=len(X)
        D_m=np.ones(n)/n
        #self.D[:]=1/n
        #from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        #classifier_m=SVC(kernel="linear",gamma=5,C=0.001)  #目前发现只有直接用SVC才可以设置sample_weight
        classifier_m=SVC(kernel="linear")
        #classifier_m=self.getClassifier(classifiername)
        if classifier_m:
            for m in range(self.M):
                Weights=D_m*n
                print("weights",Weights)
                classifier_m.fit(X,y,sample_weight=Weights)
                y_predict=classifier_m.predict(X)
                print("y_predict",y_predict)
                # miss is 8.1(b) 中的计算分类误差率要乘以w的
                miss = [int(i) for i in (y_predict != y)]

                # # miss2 是8.5中y*G(m)
                # miss1 = [x if x == 1 else -1 for x in miss]
                #要注意np.dot()也可以一个ndarry 一个列表相乘 这里计算分类误差率和alpha_m
                em =np.dot(D_m,miss)
                alpha_m=1/2*np.log((1-em)/em)
                self.classifiers.append(pickle.dumps(classifier_m))
                self.alphas.append(alpha_m)

                D_mplus1=np.zeros(n)
                for i in range(n):
                    Zm=self.getZm(D_m,alpha_m,classifier_m)
                    W_mi=D_m[i]
                    yi=y[i]
                    Gmxi=classifier_m.predict([X[i]])[0]
                    w_new_i=W_mi/Zm *np.exp(-alpha_m*yi*Gmxi)
                    D_mplus1[i]=w_new_i              
                D_m=D_mplus1
                print("误差率:",em)

            print(pickle.loads(self.classifiers[0]).predict(X))
            print(pickle.loads(self.classifiers[1]).predict(X))
            print(pickle.loads(self.classifiers[2]).predict(X))
            y_predict=self.G(X)
            print("alphas:",self.alphas)
            print("error rate:",sum(y_predict!=y)/n)
            #print(sum(miss[:]n))
            
        else:
            print("fault classifier")



if __name__=="__main__":
    train_group,train_labels=loadDataSet("testdata.txt")
    train_labels[train_labels[:]==0]=-1
    adaboost=Adaboost(3)
    adaboost.fit("svm",train_group,train_labels)
