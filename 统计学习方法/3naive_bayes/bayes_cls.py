import numpy as np
def createData():
    train_group=np.array([[1,2,3],[0,0,5],[5,6,7],[6,6,10],[9,4,6],\
        [-2,-4,-5],[-1,-2,-3],[-5,-4,-3]])
    train_labels=np.array([1,1,2,2,2,3,3,3])
    return train_group,train_labels

def bayes_cls(train_group,train_labels,X,LANGDA):
    result=[] #分类结果
    set_labels=np.unique(train_labels) #统计总共有多少种label
    K=set_labels.shape[0]
    N=train_labels.shape[0]  #样本个数
    n=len(train_group[1])  #输入维度
    print(set_labels)
    Pset_label={}  #字典存储每个可能y的先验概率
    for i in set_labels:
        #Pset_label.append((sum(train_labels==i)+LANGDA)/(N+K*LANGDA)) 
        Pset_label[i]=(sum(train_labels==i)+LANGDA)/(N+K*LANGDA)#对应于先验概率的贝叶斯估计
    print(Pset_label)  #计算先验概率

    #转秩求每个输入维度有多少种取值
    train_groupT=np.transpose(train_group)
    listX=[]
    for listarray in train_groupT:
        listXj=np.unique(listarray)
        listX.append(listXj)
    print(listX)  #每个维度的取值范围为Sj

    #计算每个以y为条件 x的概率
    ck_p_disk={}
    Regulation_p_ck_disk={}
    #P_langda_y=[]

    for x in X:
        for ck in set_labels:
            i=0
            pck=np.random.random(n)
            while i<n:#维度
                #pck=np.array(np.arange(n))
                j=0
                a=0
                while j<N: #训练集个数
                    if(train_labels[j]==ck and train_group[j][i]==x[i]):
                        a+=1
                    j+=1

                sumI=a+LANGDA#分子
                pck[i]=sumI/(sum(train_labels==ck)+len(listX[i])*LANGDA)
                i+=1
            #算出来总量
            #print(pck)
            compution=1
            for m in pck:
                compution*=m
            ck_p_disk[ck]=compution  #只要计算当前x 不用对所有 可能出现的x计算
            Regulation_p_ck_disk[ck]=compution*Pset_label[ck]  #乘上先验概率
        result.append(max(Regulation_p_ck_disk,key=Regulation_p_ck_disk.get))
    return result




if __name__=="__main__":
    train_group,labels=createData()
    print(train_group)
    print(labels)
    X=np.array([[1,2,1],[-2,-4,-5],[-5,-4,-3]])
    classify_result=bayes_cls(train_group,labels,X,1)
    print(classify_result)
