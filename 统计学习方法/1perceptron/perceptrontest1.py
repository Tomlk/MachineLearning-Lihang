import numpy as np
import matplotlib.pyplot as plt
def perceptron(train_group,train_labels,w0,b0,delta):
    Maxdelta_time=5*1e+4
    w=w0
    b=b0
    n=train_group.shape[0]
    finished=False
    delta_time=0
    while finished==False and delta_time<Maxdelta_time:
        index=0
        while index<n:
            if train_labels[index]*(w.dot(train_group[index])+b)<=0:
                w=w+delta*train_labels[index]*train_group[index]
                b=b+delta*train_labels[index]
                break
            else:
                index+=1
        if index>=n:
            finished=True
        delta_time+=1
    print(delta_time)
    return w,b

def createData():
    group=np.array([[3,3],[4,3],[1,1],[1,0],[3,0],[4,1]])
    labels=np.array([1,1,-1,-1,-1,1])
    return group,labels
def plotResults(train_group,labels,w,b):
    #plot the points
    X=train_group
    y=labels
    x1min,x1max=min(X[:,0]),max(X[:,0])
    x2min,x2max=min(X[:,1]),max(X[:,1])
    assert len(X)==len(y)
    plt.plot(X[:,0][y==-1],X[:,1][y==-1],"bs")
    plt.plot(X[:,0][y==1],X[:,1][y==1],"yo")
    plt.axis([x1min-1,x1max+1,x2min-1,x2max+1])
    #plot line
    x0=np.linspace(x1min-1,x1max+1,200)
    decision_doundary=-w[0]/w[1]*x0-b/w[1]
    plt.plot(x0,decision_doundary,"k-",linewidth=2)
    plt.savefig("result.png",format='png',dpi=300)
    plt.show()


if __name__=="__main__":
    train_group,labels=createData()
    w0=np.arange(train_group.shape[1])
    b0=0
    w,b=perceptron(train_group,labels,w0,b0,0.2)
    plotResults(train_group,labels,w,b)



