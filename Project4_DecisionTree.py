import pandas as pd
from math import log
import operator

filename='iris.txt'
# 读取数据
def getData(filename):
    # 用pandas读取txt文件
    data=pd.read_csv(filename,header=None)
    # 组织成list
    data_list=data.values.tolist()
    return data_list

#计算信息熵
def calcomentropy(data):
    # 初始化信息熵
    comentropy=0.0
    # 字典存储所有鸢尾花的种类
    All_labels={}
    # 获取数据集的长度
    length=len(data)
    for row in data:
        # 种类为每一行的最后一个元素
        label=row[-1]
        # 如果该类不存在，则在字典中添加，并计数为0
        if(label not in All_labels.keys()):
            All_labels[label]=0
        # 每次遍历都将对应的种类数+1
        All_labels[label]+=1
    # 计算信息熵
    for i in All_labels:
        # 计算每个类别的概率，即权重
        weight=float(All_labels[i])/length
        comentropy-=weight*log(weight,2)
    # print('该节点的信息熵为：',comentropy)
    return comentropy

# 分类后的小于某个value的数据
def getNewDataLess(data, num, value):
    newdata1 = []
    for row in data:
        row_list = list(row)
        # 对于连续值的计算：将每个可能的value都用来计算，找出最合适的那个value最为该特征的最优分割
        if row_list[num] <= value:
            newdata1.append(row_list)
    # print(newdata1)
    return newdata1

# 分类后的大于某个value的数据
def getNewDataGreater(data, num, value):
    newdata2 = []
    for row in data:
        row_list = list(row)
        if row_list[num] > value:
            newdata2.append(row_list)
    # print(newdata2)
    return newdata2

# 选取最优特征
def ChooseBestFeature(data):
    # 计算特征数量
    feature_num = len(data[0]) - 1
    # 计算数据集的信息熵
    base_comentroy = calcomentropy(data)
    # 初始化最大信息增益
    max_gain = 0.0
    # 初始化最优特征
    best_feature = -1
    # 初始化每个特征中最优分割点的值
    best_feature_value = -1
    for i in range(feature_num):
        # 确定某一特征下所有可能的取值,set函数去除重复值
        feature_list = set([example[i] for example in data])
        # 初始化每个特征中最优分割点的信息增益
        simple_best_gain = 0.0
        simple_best_feature_value=0.0
        for value in feature_list:
            # print('-------------------------------')
            # print('当前的属性为 ',i,' 值为 ',value)
            # 初始化信息熵
            comentropy=0.0
            # 抽取在该特征的每个取值下其他特征的值组成新的子数据集
            new_data_less = getNewDataLess(data, i, value)
            new_data_greater = getNewDataGreater(data, i, value)
            # 计算该特征下的每一个取值的比重
            weight_less = float(len(new_data_less)/len(data))
            weight_greater = float(len(new_data_greater) / len(data))
            if(weight_greater==0):
                comentropy += weight_less * calcomentropy(new_data_less)
            else:
                # 计算该特征下每一个取值的子数据集的信息熵
                comentropy += (weight_less * calcomentropy(new_data_less) + weight_greater * calcomentropy(new_data_greater))
            # 对于一个连续数值的特征，要找到其内部最优的分割点的熵作为该特征的最大信息增益
            simple_gain=base_comentroy-comentropy
            if(simple_gain > simple_best_gain):
                simple_best_gain=simple_gain
                simple_best_feature_value=value
        print("第%d个特征<=%s,对应的最大信息增益值是%f"%((i+1),simple_best_feature_value,simple_best_gain))
        # 求出每个特征的最大信息增益，再各自比较
        gain=simple_best_gain
        if (gain >= max_gain):
            max_gain = gain
            best_feature = i
            best_feature_value=simple_best_feature_value
    print("第%d个特征的信息增益最大，其特征的取值为<=%s,对应的信息增益值是%f"%((best_feature+1),best_feature_value,max_gain))
    return best_feature,best_feature_value,max_gain

# dataSet=getData(filename)
# best_feature,best_feature_value,max_gain=ChooseBestFeature(dataSet)


# 停止条件：如果纯度达到95%或数据数目小于等于5
def StopCondition(data):
    feature_count={}#创建字典
    nums=len(data)
    for row in data:
        label=row[-1]
        if label not in feature_count.keys():
            # 如果现阶段的字典中缺少这一类的特征，创建到字典中并令其值为0
            feature_count[label]=0
        #  循环一次，在对应的种类的数量上加一
        feature_count[label] +=1
    sortedcount=sorted(feature_count.items(),key=operator.itemgetter(1),reverse=True)#operator.itemgetter(1)是抓取其中第2个数据的值
    print("—"*200)
    print("当前数据的分类及其对应的个数为：",sortedcount)
    #利用sorted方法对class count进行排序，并且以key=operator.itemgetter(1)作为排序依据降序排序因为用了（reverse=True）
    percent=float(sortedcount[0][1]/nums)
    # 如果停止则返回True、数量最大的类别号、纯度、叶子大小
    # 反之返回False
    if(nums<=5 or percent>=0.95):
        return True,sortedcount[0][0],percent,nums
    else:
        return False,0,0,0


#特征对应标签
features={0:'花萼长度',1:'花萼宽度',2:'花瓣长度',3:'花瓣宽度'}
#决策树创建函数
def createtree(data,tree):

    #获取是否停止建树
    stop,label,percent,nums=StopCondition(data)
    print("isStop?:",stop)

    #如果不停止，则为树节点，则输出条件、信息增益
    if(stop==False):
        best_feature, best_feature_value, max_gain=ChooseBestFeature(data)
        tree["判断条件"]=str(features[best_feature])+"<="+str(best_feature_value)
        tree["信息增益"]=max_gain
        #对符合条件和不符合条件继续进行判断，建树
        ytree=tree.setdefault("yes", {})
        ntree=tree.setdefault("no",{})
        # 左右子树分开，分别递归
        yesData=getNewDataLess(data,best_feature,best_feature_value)
        noData=getNewDataGreater(data,best_feature,best_feature_value)
        print("左子树数据集为:",yesData)
        print("右子树数据集为：",noData)
        createtree(yesData,ytree)
        createtree(noData,ntree)

    # 如果停止则为叶子节点，输出类别，纯度，大小
    else:
        tree["类别"]=label
        tree["纯度"]=percent
        tree["大小"]=nums
    return tree

#决策树字典
DecisionTree={}
dataSet= getData(filename)
myTree = createtree(dataSet, DecisionTree)
print("—"*200)
print("决策树为：",myTree)