import pandas as pd
import math

filename='iris.txt'

# 读取数据
def getData(filename):
    # 用pandas读取txt文件,只读前四列属性
    data=pd.read_csv(filename,usecols=[0,1,2,3],header=None)
    # 组织成list
    data_list=data.values.tolist()
    return data_list

# data=getData(filename)
# print(data)

# 计算两个四维点之间的距离
def getDistance(Vec1,Vec2):
    length=len(Vec1)
    Dist=0.0
    for i in range(length):
        Dist+= (Vec1[i]-Vec2[i])*(Vec1[i]-Vec2[i])
    return math.sqrt(Dist)

# distance=getDistance([1,2,3,4],[3,3,4,5])
# print(distance)

# 四维高斯核函数
def Gaussian_kernel(Vec1,Vec2,h):
    # pow()的第二个参数 2=d/2，其中d=4，即四维空间
    left=1/math.pow(2*math.pi,2)
    # 四维向量下的距离公式
    dist=getDistance(Vec1,Vec2)
    right=math.exp(-(dist*dist)/(2*(h*h)))
    K=left*right
    return K

# 获取距离中心点距离小于带宽的所有点并暂时标记
def getPoints(data,visit_temp,point,h):
    nums=len(data)
    pointslist=[]
    for i in range(nums):
        dist=getDistance(point,data[i])
        if(0<dist<=h):
            visit_temp[i]=1
            pointslist.append(data[i])
    return pointslist


# MeanShift均值漂移函数
def MeanShift(center,pointslist,h):
    # 记录新中心的四个纬度，初始化
    # axis[i]表示第i个纬度漂移后的值，即新中心的值
    axis1 = 0.0
    axis2 = 0.0
    axis3 = 0.0
    axis4 = 0.0
    all_weight=0.0
    newcenter=[]
    for vec in pointslist:
        weight=Gaussian_kernel(center,vec,h)
        #加上每个样点在不同纬度上的贡献
        axis1 += weight * vec[0]
        axis2 += weight * vec[1]
        axis3 += weight * vec[2]
        axis4 += weight * vec[3]
        #计算总权重
        all_weight += weight
    #将漂移后的每个纬度的值均一化
    axis1 = axis1 / all_weight
    axis2 = axis2 / all_weight
    axis3 = axis3 / all_weight
    axis4 = axis4 / all_weight
    #漂移后的新中心
    newcenter.append(axis1)
    newcenter.append(axis2)
    newcenter.append(axis3)
    newcenter.append(axis4)
    return newcenter

# 计算吸引子的核密度
# center为中心点向量，pointslist中包含了在距离中心点h范围内的所有样点向量
def getDensity(center,pointslist,h):
    nums=len(pointslist)
    density=0.0
    # 求出每个点对中心点密度的贡献值
    for i in range(nums):
        density+=(Gaussian_kernel(center,pointslist[i],h)/(math.pow(h,4)*nums))
    return density

# 装载分类后的密度吸引子及其对应的样点
centerpoints=[]
samplepoints=[]
# 全局变量用来标记第count个密度吸引子
count=0

# 找到所有符合阈值的密度吸引子函数:数据集、真实visit标记、暂时visit标记、带宽、密度阈值、收敛阈值
# 分真实visit和暂时visit用于：先在暂时visit中标记该密度吸引子关联的样点，如果最后的密度吸引子的密度不符合要求，
# 则不对真实visit修改，反之，对真实visit修改
def getAllCenters(data,visit,visit_temp,h,d,e):
    nums = len(data)
    center=[]

    # 在数据集中随机选取一个没有被标记的点作为起始点
    for i in range(nums):
        if(visit[i]==0):
            visit_temp[i]=1
            center=data[i]
            break
    # 如果数据集中已经没有数据了则停止
    if(len(center)==0):
        return False
    # 不断进行均值漂移，直到最后漂移量小于阈值e
    isStop=False
    while(isStop==False):
        pointslist=getPoints(data,visit_temp,center,h)
        # 获得新的中心点
        newcenter=MeanShift(center,pointslist,h)
        dist=getDistance(center,newcenter)
        # 停止标志，中心点和新的中心点距离小于一个阈值e
        if(dist<=e):
            isStop=True
        # 变换新的中心点
        center=newcenter

    # 得到最后一次的中心周围的样点及其核密度
    pointslist=getPoints(data,visit_temp,center,h)
    density=getDensity(center,pointslist,h)

    # 判断密度是否大于阈值，如果大于，则把最后漂移的中心点加入到中心点群centerpoints中，
    # 其对应的样点加入到样点群samplepoints对应的位置中
    global count
    if(density>=d):
        count+=1
        print("第", count, "个吸引子的密度为：", density)
        addpoints=[]
        centerpoints.append(center)
        for i in range(len(visit_temp)):
            # 把符合要求的点都放入集合中，并在visit数组标记该点
            if(visit_temp[i]==1 and visit[i]==0):
                addpoints.append(data[i])
                visit[i]=1
            # 不论是否符合核密度要求，都把对应的visit_temp清零
            visit_temp[i]=0
        samplepoints.append(addpoints)
    return True

# 计算每个密度吸引子可以密度直达的其他密度吸引子集合
def getArrival(centers,h):
    AllArrival=[]
    nums=len(centers)
    for i in range(nums):
        # 用于存放i可以密度直达的点
        MyArrival=[]
        for j in range(nums):
            # 如果两者距离小于等于带宽，则说明密度直达，把j加入到i的数组中
            if (getDistance(centers[i], centers[j]) <= h):
                MyArrival.append(j)
        # 把每一个吸引子可以密度直达的吸引子集合放置在一起，形成一个密度直达矩阵
        AllArrival.append(MyArrival)
    return AllArrival

# 结果数组，全局变量
center_result=[]
sample_result=[]

# 合并密度吸引子，如果两两密度吸引子密度直达，即加入同一个簇中
# i用于标记当前处理的密度吸引子编号
# AllArrival则是用来装载每个密度吸引子可以密度直达的其他密度吸引子集合
# Sign用来标记每个密度吸引子所属的簇编号
# center_visit为每个密度吸引子是否已经被加入某个簇中，是为1，否为0
# centers为密度吸引子集合，samples为每个密度吸引子对应的样点集合
def MergeCenters(i,AllArrrival,Sign,center_visit,centers,samples):
    label=Sign[i]
    # 如果当前的密度吸引子还没有所属的簇，则新建一个簇，并给该密度吸引子的Sign标记
    if(label==0):
        center_result.append([])
        sample_result.append([])
        label=len(sample_result)-1
        Sign[i]=label
    # 对该密度吸引子密度直达的其他吸引子进行查找和标记，并添加到相应的簇中
    for j in range(len(AllArrrival[i])):
        # 密度吸引子i可密度直达的吸引子j如果没有被添加过，
        # 则就给它的Sign标记密度吸引子i所属的簇，并把密度吸引子j和其对应的样点加入到对应的簇
        if(center_visit[AllArrrival[i][j]]==0):
            center_result[Sign[i]].append(centers[AllArrrival[i][j]])
            sample_result[Sign[i]].extend(samples[AllArrrival[i][j]])
            center_visit[AllArrrival[i][j]] =1
            Sign[AllArrrival[i][j]]=label
            # 递归查找并添加
            MergeCenters(AllArrrival[i][j],AllArrrival,Sign,center_visit,centers,samples)

# 执行各函数形成完成的DENCLUE算法
def Run(data,visit,visit_temp,h,d,e):
    sign=True
    count=0

    # 找出所有的密度吸引子
    while(sign==True):
        sign=getAllCenters(data,visit,visit_temp,h,d,e)

    # 制作一些visit和sign等标记数组，所有值为0
    num=len(centerpoints)
    center_visit=[]
    Sign=[]
    for i in range(num):
        center_visit.append(0)
        Sign.append(0)

    # 获得每个密度吸引子可以密度直达的其他密度吸引子集合
    AllArrival=getArrival(centerpoints,h)

    # 密度可达的密度吸引子们合并到相同簇中
    for i in range(len(AllArrival)):
        MergeCenters(i,AllArrival,Sign,center_visit,centerpoints,samplepoints)


if __name__ == '__main__':
    dataSet = getData(filename)
    nums = len(dataSet)
    Visit = []
    Visit_temp = []
    for i in range(nums):
        Visit.append(0)
        Visit_temp.append(0)

    # 执行DENCLUE算法，带宽h=1,密度阈值为0.001，收敛阈值为0.3
    Run(dataSet,Visit,Visit_temp, 1, 0.001, 0.3)

    # 输出聚类相关信息
    print("—" * 100)
    print("密度吸引子个数：", len(centerpoints))
    for j in range(len(samplepoints)):
        print("第",j+1,"个密度吸引子周围的样点数量为：",len(samplepoints[j]))
    print("—" * 100)
    print("密度吸引子为：", centerpoints)
    print("各密度吸引子周围的样点：",samplepoints)


    # 对每个簇中的密度吸引子们求均值作为该簇的中心
    center_Average=[]
    for i in range(len(center_result)):
        x0=0.0
        x1=0.0
        x2=0.0
        x3=0.0
        for j in range(len(center_result[i])):
            x0 += center_result[i][j][0]
            x1 += center_result[i][j][1]
            x2 += center_result[i][j][2]
            x3 += center_result[i][j][3]
        x0 = x0 / len(center_result[i])
        x1 = x1 / len(center_result[i])
        x2 = x2 / len(center_result[i])
        x3 = x3 / len(center_result[i])
        center_Average.append([x0,x1,x2,x3])


    print("—" * 100)
    print("聚类簇的个数：", len(center_Average))
    print("聚类簇中心为：", center_Average)
    print("—"*100)
    for i in range(len(sample_result)):
        print("第",i+1,"个聚类簇中心为：", center_Average[i])
        print("第",i+1,"个簇的密度吸引子数量为：",len(center_result[i]))
        print("第",i+1,"个簇的密度吸引子为：", center_result[i])
        print("第",i+1,"个簇的样点数量为：",len(sample_result[i]))
        print("第",i+1,"个簇的样点为：", sample_result[i])
        print("—"*100)












