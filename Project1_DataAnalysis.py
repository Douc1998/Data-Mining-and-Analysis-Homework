import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, seaborn
import math

# 读取文件
def getData(filename):
    data = pd.read_csv(filename, header=None, usecols=[0, 1, 2, 3,4,5,6,7,8,9])
    datalist = data.values.tolist()
    # 获取属性个数
    feature_nums = len(datalist[0])
    dataSet = []
    # 获取每个属性下的数据
    for i in range(feature_nums):
        feature_list = [example[i] for example in datalist]
        dataSet.append(feature_list)
    # datalist为行式，dataSet为列式
    return datalist,dataSet

# 计算多元均值向量
def getMeanVector(data):
    # 存放均值向量
    MeanVector=[]
    # 求每个属性的均值
    for item in data:
        average=0.0
        for value in item:
            average+=value
        average/=len(item)
        MeanVector.append(average)
    return MeanVector

# 内积计算协方差矩阵
def getCovarianceMatrix1(data,MeanVector):
    # 中心化
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]-=MeanVector[i]
    data=np.array(data)
    # 获得data矩阵的转置
    dataT=data.transpose()
    # 计算协方差矩阵
    covariance_matrix = np.dot(data,dataT)/(len(dataT)-1)
    #打印协方差矩阵
    print("内积计算协方差矩阵为：","—"*200)
    print(covariance_matrix)

    # 可视化绘图
    seaborn.heatmap(covariance_matrix, center=0, annot=True, xticklabels=list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    yticklabels=list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).set_ylim([10, 0])
    plt.title("CovarianceMatrix by Dot product")
    plt.show()
    return covariance_matrix

# 外积计算协方差矩阵
def getCovarianceMatrix2(data,MeanVector):
    # 中心化
    getFeatureNum=0
    for item in data:
        getFeatureNum=len(item)
        for i in range(len(item)):
            item[i]-=MeanVector[i]

    # 协方差矩阵初始化为n*n的全为0的矩阵，n为特征数
    covariance_matrix=np.zeros(shape=(getFeatureNum,getFeatureNum))
    # outer即为将data[i]的列向量模式和data[i]的行向量模式相乘，得到一个n*n的矩阵
    # 每次把得到的矩阵的值都加在一起，最后就可以得到协方差矩阵
    for i in range(len(data)):
        covariance_matrix+=(np.outer(data[i],data[i])/(len(data)-1))

    # 打印外积计算的协方差矩阵
    print("外积计算协方差矩阵为：", "—" * 200)
    print(covariance_matrix)

    # 可视化绘图
    seaborn.heatmap(covariance_matrix, center=0, annot=True, xticklabels=list([1,2,3,4,5,6,7,8,9,10]), yticklabels=list([1,2,3,4,5,6,7,8,9,10])).set_ylim([10,0])
    plt.title("CovarianceMatrix by Outer product")
    plt.show()
    return covariance_matrix

#结算属性1和属性2的相关系数和绘制属性1，2之间的散点图
def Draw(filename):
    data = pd.read_csv(filename, header=None, usecols=[0, 1])
    x=data[0]
    y=data[1]
    # 计算相关系数
    # 先计算均值用于中心化
    x_mean=x.mean()
    y_mean=y.mean()
    # 计算方差
    x_variance=0.0
    y_variance=0.0
    for i in range(len(x)):
        x_variance+=(x[i]-x_mean)**2
        y_variance+=(y[i]-y_mean)**2
    # 计算属性1和属性2的协方差
    # 先中心化
    for i in range(len(x)):
        x[i]-=x_mean
        y[i]-=y_mean
    # 内积计算协方差
    dataY=np.array(y).transpose()
    dataX=np.array(x)
    cov=np.dot(dataY,dataX)/(len(x)-1)
    # 去除量纲，计算相关系数
    p=cov/(math.sqrt(x_variance*y_variance))
    print("—"*200)
    print("属性1和属性2的相关系数为：",p)

    # 绘制散点图
    fig = plt.figure()
    # 1×1网格，第一子图
    ax1 = fig.add_subplot(111)
    # 标题
    ax1.set_title('The Scatter plot of Feature1 and Feature2 ')
    # 坐标轴
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    # 绘制
    plt.scatter(x, y,c='grey',marker='o')
    plt.show()

# 正态分布函数
def Gaussian(x, mu, sigma):
    left = 1 / (np.sqrt(2 * math.pi) * sigma)
    right = np.exp(-(x - mu)**2 / (2 * sigma * sigma))
    gaus=left*right
    return gaus

# 绘制属性1在正态分布下概率密度图
def GaussianAndDraw(filename):
    data = pd.read_csv(filename, header=None, usecols=[0])
    data=np.array(data)
    # 获取均值，即数学期望
    mean=float(data.mean())
    # 获取标准差
    std=float(data.std())

    x = np.arange(mean-200, mean+200, 1)
    y=Gaussian(x,mean,std)
    # 绘制高斯分布
    plt.plot(x, y)
    plt.text(120, 0.008, "Mean=%f\n\nVariance=%f"%(mean,std*std))
    plt.title('Feature1_Gaussian')
    plt.xlabel('Feature1')
    plt.ylabel('Probability')
    plt.show()

# 计算最大最小方差
def get_MaxandMin_Variance(filename):
    data = pd.read_csv(filename, header=None, usecols=[0,1,2,3,4,5,6,7,8,9])
    Variance=[]
    # 获取每个属性的方差
    for i in range(10):
        Variance.append(float(data[i].std())**2)
    print(Variance)
    Max_Variance=-1
    Max_num=-1
    Min_Variance=999999999
    Min_num=-1
    for i in range(len(Variance)):
        if(Variance[i]>=Max_Variance):
            Max_Variance=Variance[i]
            Max_num=i
        if(Variance[i]<=Min_Variance):
            Min_Variance=Variance[i]
            Min_num=i
    print('—'*200)
    print("方差最大的为第",Max_num+1,'个属性，','方差为：',Max_Variance)
    print("方差最小的为第", Min_num+1, '个属性，', '方差为：', Min_Variance)

# 计算最大和最小的协方差及其对应的属性对
def get_MaxandMin_Covariance(covariance_matrix):
    # 标记初始化
    Max_num1=-1
    Max_num2=-1
    Max_covariance=-999999
    Min_num1=-1
    Min_num2=-1
    Min_covariance=999999
    for i in range(len(covariance_matrix)):
        for j in range(len(covariance_matrix)):
            # 在协方差矩阵中，对角线上为各属性的方差，因此不算，必须i!=j
            if(i!=j):
                if (covariance_matrix[i][j] > Max_covariance):
                    Max_covariance = covariance_matrix[i][j]
                    Max_num1 = i
                    Max_num2 = j
                if (covariance_matrix[i][j] < Min_covariance):
                    Min_covariance = covariance_matrix[i][j]
                    Min_num1 = i
                    Min_num2 = j
    print('—'*200)
    print("协方差最大的为第",Max_num1+1,'个属性和第',Max_num2+1,'个属性,其协方差为：',Max_covariance)
    print("协方差最小的为第", Min_num1 + 1, '个属性和第', Min_num2 + 1, '个属性,其协方差为：', Min_covariance)




if __name__ == '__main__':
    filename = 'magic.txt'
    datalist,dataSet=getData(filename)
    # 多元均值向量
    print("多元均值向量为：","—"*200)
    print(getMeanVector(dataSet))
    MeanVector = getMeanVector(dataSet)
    # 内积计算协方差矩阵
    covariance_matrix=getCovarianceMatrix1(dataSet,MeanVector)
    # 外积计算协方差矩阵
    getCovarianceMatrix2(datalist,MeanVector)
    # # 属性1和2的相关系数以及两者的散点图绘制
    Draw(filename)
    # 绘制属性1的正态分布概率密度函数曲线
    GaussianAndDraw(filename)
    # 输出最大和最小的方差及其对应的属性
    get_MaxandMin_Variance(filename)
    # 输出最大和最小的协方差及其对应的属性对
    get_MaxandMin_Covariance(covariance_matrix)