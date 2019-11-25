import numpy as np

# class ListNode:
#     def __init__(self,value,children):
#         self.value=value
#         self.children=children
#     def addchidren(self,_children):
#         self.children=_children
def createData():
    train_group=np.array([[0,0,0],[0,0,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    train_labels=np.array([0,0,0,1,0,1])
    return train_group,train_labels

class C4_5:
    def __init__(self,train_group,train_labels,_epsilon=0):
        # self.D=train_group
        # self.labels=train_labels
        self.datas=np.c_[train_group,train_labels]
    
        self.label_dic={}
        i=0
        while i<self.datas.shape[1]-1:
            self.label_dic[i]=i
            i+=1
        #self.A=train_group.shape[1]-1
        self.episilon=_epsilon
    
    def getCurrentHD(self,datas):#根据当前子数据集进行计算经验熵H(D)
        #print(datas)
        currentD=datas[:,:-1]
        currentlabels=datas[:,-1]

        #print(currentD)
        #print(currentlabels)
        if currentD.shape[0]!=currentlabels.shape[0]:
            print("error dimension not correspond")
            return
        Ck=[]
        currentlabels_=list(currentlabels)
        for i in set(currentlabels_):
            Ck.append(currentlabels_.count(i))
        HD=0
        D_=currentlabels.shape[0]
        for i in Ck:
            temp=i/D_
            HD-= temp *np.log2(temp)
        return HD
    def getCurrentConditonalHDA(self,datas,A_index):#根据当前子数据集和特征(输入维度数)来计算此时的经验条件熵
        currentD=datas[:,:-1]
        #currentlabels=datas[:,-1]
        lA=[]
        Di=[]
        Di_list=[]
        for x in currentD:
            lA.append(x[A_index])
        for i in set(lA):           
            Di.append(lA.count(i))

            l=np.array([])
            j=0
            while j<currentD.shape[0]:
                if currentD[j][A_index]==i:
                    l=np.r_[l,datas[j]]
                j+=1
            l=np.reshape(l,(-1,datas.shape[1]))
            Di_list.append(l)
        
        HDA=0
        D_=currentD.shape[0]
        if len(Di)!=len(Di_list):
            print("error dimension not correspond")
            return
        index=0
        while index<len(Di):
            HDA+=(Di[index]/D_)*self.getCurrentHD(Di_list[index])
            index+=1
        return HDA
    
    def get_gDA(self,datas,A_index):
        return self.getCurrentHD(datas)\
            -self.getCurrentConditonalHDA(datas,A_index)
    def get_HAD(self,datas,A_index):
        currentD=datas[:,:-1]
        #currentlabels=datas[:,-1]
        D_=currentD.shape[0]
        A_list=[]
        for i in currentD[:][A_index]:
            A_list.append(i)
        Di_list=[]
        for i in set(A_list):
            Di_list.append(A_list.count(i))
        HAD=0
        for i in Di_list:
            temp=i/D_
            HAD-=temp*np.log2(temp)
        return HAD
    
    def get_gRDA(self,datas,A_index):
        return self.get_gDA(datas,A_index)/self.get_HAD(datas,A_index)


    def chooseA_index(self,datas):
        maxget_gRDA=0
        value=0   #特征索引
        i=0
        while i<datas.shape[1]-1:
            if i==0:
                maxget_gRDA=self.get_gRDA(datas,i)
                i+=1
                continue
            if maxget_gRDA<self.get_gRDA(datas,i):
                value=i
                maxget_gRDA=self.get_gRDA(datas,i)
            i+=1
        return value

    def splitDatas(self,datas,A_index,value):
        retDatas=[]
        for featVec in datas:
            if featVec[A_index]==value:
                #reduceFeatVec=featVec[:A_index]
                reduceFeatVec=np.delete(featVec,A_index)
                #reduceFeatVec.extend(featVec[A_index+1:])
                retDatas.append(reduceFeatVec)
        retDatas=np.array(retDatas)
        return retDatas#返回不含划分特征的子集

    def create_decisionTree(self,datas,label_dic):
        label_dic_=label_dic.copy()
        classlist=[example[-1] for example in datas]
        #类别相同，停止划分
        if classlist.count(classlist[-1])==len(classlist):
            return classlist[-1]
        #特征只有1个,返回实例数最大的类Ck作为该结点的类标记
        #print(classlist[0])
        if datas.shape[1]==2:
            return np.argmax(np.bincount(classlist))
        bestFeat=self.chooseA_index(datas)
        bestFeatLabel=label_dic_[bestFeat]
        myTree={bestFeatLabel:{}}
        del(label_dic_[bestFeat])
        featValues=[example[bestFeat] for example in datas]
        uniqueVals=set(featValues)
        for value in uniqueVals:
            subLabels=label_dic_
            myTree[bestFeatLabel][value]=self.create_decisionTree(self.splitDatas(datas,bestFeat,value),subLabels)
        return myTree
    
    def classify(self,input_tree,labels_str,x):
        firstStr=list(input_tree.keys())[0]
        secondDict=input_tree[firstStr]
        featIndex=firstStr
        #featIndex=labels_str.get(firstStr)
        for key in secondDict.keys():
            if x[featIndex]==key:
                if type(secondDict[key]).__name__=='dict':#还没到叶结点
                    class_label=self.classify(secondDict[key],labels_str,x)
                else:
                    class_label=secondDict[key]
        return class_label

    def get_result(self,input_tree,labels_str,X):
        result_labels=[]
        for x in X:
            result_labels.append(self.classify(input_tree,labels_str,x))
        return result_labels
    
    def start(self,testX):
        print("before",self.label_dic)
        Mytree=self.create_decisionTree(self.datas,self.label_dic)
        print("after",self.label_dic)
        result=self.get_result(Mytree,self.label_dic,testX)
        print(result)

if __name__=="__main__":
    train_group,labels=createData()
    print(train_group)
    print(labels)
    X=np.array([[1,1,1],[0,1,0],[1,0,1]])
    id3_tree=C4_5(train_group,labels)
    id3_tree.start(X)       





 
        