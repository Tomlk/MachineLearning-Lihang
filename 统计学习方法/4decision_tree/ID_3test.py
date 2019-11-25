#ID3算法，以《机器学习》（周志华板）表4.3西瓜数据集为例
import numpy as np

#标准化数据
dic={"浅白":1,"稍蜷":1,"清脆":1,"清晰":1,"平坦":1,"硬滑":1,\
    "青绿":2,"蜷缩":2,"浊响":2,"稍糊":2,"稍凹":2,"软粘":0,\
        "乌黑":3,"硬挺":3,"沉闷":3,"模糊":3,"凹陷":3,"是":1,"否":0}

# def createData():
#     train_group=np.array([[0,0,0],[0,0,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
#     train_labels=np.array([0,0,0,1,0,1])
#     return train_group,train_labels

#加载训练数据构建决策树
def loadData(filename:str):
    label_names=[]
    train_datas=[]
    with open(filename,'r') as fr: 
        lines=fr.readlines()
        firstlineFlag=True
        for line in lines:
            linestr=line.strip().split(',')
            if firstlineFlag:
                label_names=linestr
                firstlineFlag=False
            else:
                train_datas.append(linestr)
    # print(label_names)
    # print(train_datas)
    label_names=label_names[:-1]
    train_datasNormal=[]
    for listdata in train_datas:
        nlist=[]
        for s in listdata:
            if s not in dic.keys():
                nlist.append(float(s))
            else:
                nlist.append(dic[s])
        train_datasNormal.append(nlist)
    train_datasNormal=np.array(train_datasNormal)
    train_group=train_datasNormal[:,:-1]
    train_labels=train_datasNormal[:,-1]
    return label_names,train_group,train_labels

#加载测试数据，没有需要预测的标签
def load_testDatas(filename):
    test_datas=[]
    with open(filename,'r') as fr: 
        lines=fr.readlines()
        firstlineFlag=True
        for line in lines:
            linestr=line.strip().split(',')
            if firstlineFlag:
                firstlineFlag=False
            else:
                test_datas.append(linestr)
    return test_datas

#算法类
class ID_3:
    def __init__(self,train_group,train_labels,label_names,_epsilon=0):
        self.datas=np.c_[train_group,train_labels] #再次整合

        self.raw_label_list=label_names
        self.label_list=label_names

        self.Mytree=None  #决策树
        self.episilon=_epsilon  #调节参数，这里不需要
        self.decision_point={}  #对于连续特征数据的划分点保存
    
    def getCurrentHD(self,datas):#根据当前子数据集进行计算经验熵H(D)
        currentD=datas[:,:-1]
        currentlabels=datas[:,-1]
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
        if self.label_list[A_index]=="密度" or self.label_list[A_index]=="含糖率": #连续特征值单独讨论
            discreteflag=False
        else:
            discreteflag=True      
        if discreteflag:
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
        else:  #连续
            # currentD=datas[:,:-1]
            currentD=datas
            list_a=list(currentD[:,A_index])
            list_a.sort()
            Ta=[]#Ta 候选划分点集合
            for i in range(len(list_a)-1):
                Ta.append((list_a[i]+list_a[i+1])/2)
            #print(Ta)
            
            HDAlist=[]
            for t in Ta:
                D_t_below=[]
                D_t_above=[]
                for x in currentD:
                    if x[A_index]<=t:
                        D_t_below.append(x)
                    else:
                        D_t_above.append(x)
                D_size=currentD.shape[0]
                D_t_below=np.array(D_t_below) #小于划分点的
                D_t_above=np.array(D_t_above) #大于划分点的
                thisHDA=len(D_t_below)/D_size *self.getCurrentHD(D_t_below)+len(D_t_above)/D_size *self.getCurrentHD(D_t_above)
                HDAlist.append(thisHDA)
            decision_point=Ta[HDAlist.index(min(HDAlist))]          
            if self.label_list[A_index] not in self.decision_point:
                self.decision_point[self.label_list[A_index]]=decision_point
                print(decision_point)
            return min(HDAlist) #最小化HDA 能使gDA最大

        
    
    def get_gDA(self,datas,A_index):
        return self.getCurrentHD(datas)\
            -self.getCurrentConditonalHDA(datas,A_index)

    def chooseA_index(self,datas):
        value=0   #特征索引
        i=0
        gDA_list=[]
        for i in range(datas.shape[1]-1):
            gDA=self.get_gDA(datas,i)
            gDA_list.append(gDA)
        value=gDA_list.index(max(gDA_list))
        #print(gDA_list)
        return value

    #数据分离
    def splitDatas(self,datas,A_index,value,discreteFlag=True,aboveFlag=False): 
        retDatas=[]
        #离散特征
        if discreteFlag:
            for featVec in datas:
                if featVec[A_index]==value:               
                    reduceFeatVec=np.delete(featVec,A_index)
                    retDatas.append(reduceFeatVec)
            retDatas=np.array(retDatas)
        #连续特征
        else:
            if aboveFlag==False:
                for featVec in datas:
                    if featVec[A_index]<=value:
                        reduceFeatVec=np.delete(featVec,A_index)
                        retDatas.append(reduceFeatVec)
            else:
                for featVec in datas:
                    if featVec[A_index]>value:
                        reduceFeatVec=np.delete(featVec,A_index)
                        retDatas.append(reduceFeatVec)
            retDatas=np.array(retDatas)
        return retDatas#返回不含划分特征的子集

    #递归创建决策树
    def create_decisionTree(self,datas,label_list):
        self.label_list=label_list.copy()
        label_list_=label_list.copy()
        classlist=[example[-1] for example in datas]
        #类别相同，停止划分
        if classlist.count(classlist[-1])==len(classlist):
            return classlist[-1]
        #特征只有1个,返回实例数最大的类Ck作为该结点的类标记
        #print(classlist[0])
        if datas.shape[1]==2:
            return np.argmax(np.bincount(classlist))
        #当前最佳特征的索引
        bestFeat=self.chooseA_index(datas)
        bestFeatLabel=label_list_[bestFeat]
        myTree={bestFeatLabel:{}}
        del(label_list_[bestFeat])
        featValues=[example[bestFeat] for example in datas]
        uniqueVals=set(featValues)

        #离散形式单独讨论
        if bestFeatLabel=="密度" or bestFeatLabel=="含糖率":
            subLabels=label_list_
            decison_value=self.decision_point[bestFeatLabel]
            myTree[bestFeatLabel]["low"]\
                =self.create_decisionTree(self.splitDatas(datas,bestFeat,decison_value,False,False),subLabels)
            myTree[bestFeatLabel]["high"]\
                =self.create_decisionTree(self.splitDatas(datas,bestFeat,decison_value,False,True),subLabels)
        else:
            for value in uniqueVals:
                subLabels=label_list_
                myTree[bestFeatLabel][value]=self.create_decisionTree(self.splitDatas(datas,bestFeat,value),subLabels)
        return myTree
    
    def classify(self,input_tree,label_list,x):
        firstStr=list(input_tree.keys())[0]
        secondDict=input_tree[firstStr]
        featIndex=label_list.index(firstStr)
        for key in secondDict.keys():
            if x[featIndex]==key:
                if type(secondDict[key]).__name__=='dict':#还没到叶结点
                    class_label=self.classify(secondDict[key],label_list,x)
                else:
                    class_label=secondDict[key]
        return class_label

    def get_result(self,input_tree,label_list,X):
        result_labels=[]
        for x in X:
            result_labels.append(self.classify(input_tree,label_list,x))
        return result_labels
    
    def train(self):
        self.Mytree=self.create_decisionTree(self.datas,self.label_list)

    def predict(self,testX):
        ChangedXs=[]
        for raw_x in testX:
            length=len(raw_x)
            x=[]
            for i in range(len(raw_x)):
                fea=raw_x[i]
                if fea in dic.keys():
                    x.append(dic[fea])
                else:
                    if i==length-2:  #密度
                        if float(fea)<=self.decision_point["密度"]:
                            x.append("low")
                        else:
                            x.append("high")
                    elif i==length-1:   #含糖率
                        if float(fea)<=self.decision_point["含糖率"]:
                            x.append("low")
                        else:
                            x.append("high")
            ChangedXs.append(x)     
        result=self.get_result(self.Mytree,self.raw_label_list,ChangedXs)
        resultnew=[]
        for r in result:
            if r==1:
                resultnew.append("是")
            else:
                resultnew.append("否")
        return resultnew

if __name__=="__main__":
    label_names,train_group,train_labels=loadData("watermelondata.txt")
    testX=load_testDatas("watermelontestdata.txt")
    id3_tree=ID_3(train_group,train_labels,label_names)
    id3_tree.train()
    print("决策树:",id3_tree.Mytree)
    print("判别结果",id3_tree.predict(testX))

    #id3_tree.start(X)       





 
        