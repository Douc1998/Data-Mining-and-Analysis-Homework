import numpy as np
import pandas as pd
import math

def getData(filename):
    data=pd.read_csv(filename,header=None,usecols=[0,1,2,3])
    data=np.array(data)
    return data

# 归一化和中心化
def NormalizeAndCenter(data):

    # 转置，方便计算归一化和中心化
    dataT=data.transpose()
    # 归一化
    for i in range(len(dataT)):
        max=np.max(dataT[i])
        min=np.min(dataT[i])
        dataT[i]=(dataT[i]-min)/(max-min)

    # 转置回来
    data=dataT.transpose()
    # 计算均值向量
    MeanVector=[]
    for i in range(len(dataT)):
        MeanVector.append(dataT[i].mean())
    # print("均值向量为：",MeanVector)
    # 中心化
    for value in data:
        value-=MeanVector
    # print("中心化和归一化后的矩阵为：",data)
    return data

# 齐二次核方法
# K(x,y)=(x*y)^2，x和y为两个向量
def Kernel(Vec1,Vec2):
    K=0.0
    for i in range(len(Vec1)):
        K+=Vec1[i]*Vec2[i]
    return K*K

# 输入时利用核函数计算归一化，中心化的核矩阵
def getKernelMatrix1(data):
    Kernel_Matrix=[]
    # 计算核矩阵
    for i in range(len(data)):
        row=[]
        for j in range(len(data)):
            # 两两计算
            row.append(Kernel(data[i],data[j]))
        Kernel_Matrix.append(row)
    Kernel_Matrix=np.array(Kernel_Matrix)
    # 将核矩阵中心化，归一化
    New_Kernel_Matrix=NormalizeAndCenter(Kernel_Matrix)
    print("利用输入空间的核函数计算并中心化和归一化后的核矩阵为：","—"*200)
    # 获取核矩阵的行列数
    print("核矩阵行数：",len(New_Kernel_Matrix))
    print("核矩阵列数：",len(New_Kernel_Matrix[0]))
    print(New_Kernel_Matrix)
    return New_Kernel_Matrix

# 齐二次核方法对应的高维映射为：
# 鸢尾花数据集为4维，即x=(x1,x2,x3,x4)
# 因此其高维映射为：X=（x1^2,x2^2,x3^2,x4^2,sqrt(2)*x1*x2,sqrt(2)*x1*x3,sqrt(2)*x1*x4,sqrt(2)*x2*x3,sqrt(2)*x2*x4,sqrt(2)*x3*x4)
def Mapping(vec):
    New_vec=[]
    for i in range(len(vec)):
        for j in range(i,len(vec)):
            # 自身相乘和两两相乘的区别
            if(i==j):
                New_vec.append(vec[i]*vec[j])
            else:
                New_vec.append(math.sqrt(2)*vec[i]*vec[j])
    return New_vec

# 高维特征空间计算内积得到核矩阵
def getKernelMatrix2(data):

    # 映射后的高维矩阵
    HighDimensional_Matrix=[]
    for i in range(len(data)):
        New_vec=Mapping(data[i])
        HighDimensional_Matrix.append(New_vec)

    # 计算内积
    HighDimensional_Matrix=np.array(HighDimensional_Matrix)
    HighDimensional_Matrix_T=HighDimensional_Matrix.transpose()
    Kernel_Matrix=np.dot(HighDimensional_Matrix,HighDimensional_Matrix_T)
    # # 将核矩阵中心化，归一化
    New_Kernel_Matrix=NormalizeAndCenter(Kernel_Matrix)
    print("先映射到高维空间再求内积并中心化和归一化后的核矩阵为：","—"*200)
    # 获取核矩阵的行列数
    print("核矩阵行数：",len(New_Kernel_Matrix))
    print("核矩阵列数：",len(New_Kernel_Matrix[0]))
    print(New_Kernel_Matrix)
    return New_Kernel_Matrix

if __name__ == '__main__':
    filename='iris.txt'
    data=getData(filename)
    # 打印核方法直接计算得到的核矩阵
    # 为防止计算过程中的根号取近似值对后续计算的影响，矩阵中每个数保留10位小数
    Matrix1=np.around(getKernelMatrix1(data),decimals=10)
    # 打印先映射到高维空间再内积计算得到的核矩阵
    Matrix2=np.around(getKernelMatrix2(data),decimals=10)
    # 判断矩阵中对应位置的数是否相同
    isSame=(Matrix1==Matrix2).all()
    print("—"*200)
    print("两个方法计算的核矩阵相同吗：",isSame)

