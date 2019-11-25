import numpy as np
from matplotlib import pyplot as plt
class Logistic:
    def __init__(self):
        self.data_array=None
        self.label_array=None
        self.weights=None
        pass
    def loaddata(self,fileName="data.txt"):
        data_mat=[]
        label_mat=[]
        fr=open(fileName)
        for line in fr.readlines():
            line_arr=line.strip().split()
            data_mat.append([float(line_arr[0]),float(line_arr[1]),1.0])
            label_mat.append(int(line_arr[-1]))
        
        self.data_array=np.array(data_mat).reshape((len(data_mat),3))
        self.label_array=np.array(label_mat).reshape(len(data_mat),1)
    
    def sigmoid(self,inX):
        return 1.0/(1+np.exp(-inX)) #等价于 np.exp(inX)/(1+np.exp(inx))

    #批梯度下降
    def batchgd_way(self,n,learningrate=0.01,max_iter=500,thresh=0.1):
        weights=np.ones((n,1))
        for _ in range(max_iter):
            pix=self.sigmoid(np.dot(self.data_array,weights))
            error=self.label_array-pix
            weights=weights+learningrate*np.dot(np.transpose(self.data_array),error)
            errornorm=np.linalg.norm(error)
            print(errornorm)
            if(errornorm<thresh): break
        return weights
    #牛顿法
    def newton(self,N,n,max_iter=3,thresh=0.1):
        weights=np.zeros((n,1))  #初始值要正确
        for _ in range(max_iter):
            t=np.dot(self.data_array,weights)
            pix=self.sigmoid(t)
            print("pix",pix)
            #计算向量的一阶导数
            error=self.label_array-pix
            first_derivative=np.dot(np.transpose(self.data_array),error)

            #对角矩阵
            diag=np.diagflat(pix*(pix-np.ones((N,1))))

            #Hessain矩阵
            H=np.dot(np.dot(np.transpose(self.data_array),diag),self.data_array)
            # H_inv=np.linalg.inv(H)
            # weights+=np.dot(H_inv,first_derivative)
            #注意到 Hx=first_derivative  -> x=H_inv*first_derivative
            weights-=np.linalg.solve(H,first_derivative)            
            errornorm=np.linalg.norm(error)
            print(errornorm)
            if(errornorm<thresh): break
        return weights


    def train(self):

        self.loaddata()
        n=self.data_array.shape[1]
        return self.batchgd_way(n,learningrate=0.01,max_iter=1000)
        #return self.newton(self.data_array.shape[0], n)

    def plotresults(self):
        weights=self.train()
        x_1,y_1,x_2,y_2=[],[],[],[]
        for i in range(self.label_array.shape[0]):
            if self.label_array[i]==0:
                x_1.append(self.data_array[i,0])
                y_1.append(self.data_array[i,1])
            else:
                x_2.append(self.data_array[i,0])
                y_2.append(self.data_array[i,1])
        
        plt.plot(x_1,y_1,'bo',x_2,y_2,'r^')
        linex=np.linspace(min(self.data_array[:,0])-1,max(self.data_array[:,0])+1,50)
        liney=(-weights[0]*linex-weights[2])/weights[1]
        plt.plot(linex,liney)
        plt.show()
        
                
if __name__=="__main__":
    logistic=Logistic()
    weights=logistic.plotresults()
    print(weights)