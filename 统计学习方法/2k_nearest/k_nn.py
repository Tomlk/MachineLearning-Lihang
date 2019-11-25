import numpy as np
def createData():
    train_group=np.array([[1,2,3],[0,0,5],[5,6,7],[6,6,10],[9,4,6],\
        [-2,-4,-5],[-1,-2,-3],[-5,-4,-3]])
    train_labels=np.array([1,1,2,2,2,3,3,3])
    return train_group,train_labels

def k_nn(train_group,train_labels,p,k,X):
    #连接可以索引到类别标签
    datas=np.c_[train_group,train_labels]
    classify_result=[]
    for x in X:
        a=np.sum(np.power(abs(x-train_group[:,:]),p),axis=1)
        b=np.power(a,1/p)
        #为了找到该样本，连接索引
        b=np.c_[b,np.arange(len(b))]
        #按距离大小排序
        b=b[b[:,0].argsort()]     
        indexs=b[:k,-1].astype(int)
        #这些最近样本的分类标签
        types=datas[indexs,-1]
        #返回出现次数最多的标签
        classify_result.append(np.argmax(np.bincount(types)))
    return classify_result

if __name__=="__main__":
    Train_group,Train_labels=createData()
    X=np.array([[1,1,1],[-5,-4,-3]])
    result=k_nn(Train_group,Train_labels,2,3,X)
    print(result)