import matplotlib.pyplot as plt
import random
import numpy as np
import math
from sklearn import datasets
from kneed import KneeLocator
import pandas as pd

def loadDataSet(fileName, splitChar='\t'):
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet

# 计算两个点之间的欧式距离，参数为两个元组
def dist(t1, t2):
    dis = math.sqrt((np.power((t1[0] - t2[0]), 2) + np.power((t1[1] - t2[1]), 2)))
    # print("两点之间的距离为："+str(dis))
    return dis

# DBSCAN算法，参数为数据集，Eps为指定半径参数，MinPts为制定邻域密度阈值
def dbscan(Data, Eps, MinPts):
    num = len(Data)  # 点的个数
    print("点的个数："+str(num))
    unvisited = [i for i in range(num)]  # 没有访问到的点的列表
    # print(unvisited)
    visited = []  # 已经访问的点的列表
    C = [-1 for i in range(num)]
    # C为输出结果，默认是一个长度为num的值全为-1的列表
    # 用k来标记不同的簇，k = -1表示噪声点
    k = -1
    # 如果还有没访问的点,则unvisited
    while len(unvisited) > 0:
        # 随机选择一个unvisited对象
        p = random.choice(unvisited)
        unvisited.remove(p)
        visited.append(p)
        # N为p的epsilon邻域中的对象的集合
        N = []
        for i in range(num):
            if (dist(Data[i], Data[p]) <= Eps):  # 判断该点是否在p的epsilon邻域中
                N.append(i)
        # 如果p的epsilon邻域中的对象数大于指定阈值，说明p是一个核心对象
        if len(N) >= MinPts:
            k = k + 1
            # print(k)
            C[p] = k
            # 对于p的epsilon邻域中的每个对象pi
            for pi in N:
                if pi in unvisited:
                    unvisited.remove(pi)
                    visited.append(pi)
                    # 找到pi的邻域中的核心对象，将这些对象放入N中
                    # M是位于pi的邻域中的点的列表
                    M = []
                    for j in range(num):
                        if (dist(Data[j], Data[pi]) <= Eps):  # and (j!=pi):
                            M.append(j)
                    if len(M) >= MinPts:
                        for t in M:
                            if t not in N:
                                N.append(t)
                # 若pi不属于任何簇，C[pi] == -1说明C中第pi个值没有改动
                if C[pi] == -1:
                    C[pi] = k
        # 如果p的epsilon邻域中的对象数小于指定阈值，说明p是一个噪声点
        else:
            C[p] = -1

    return C

def knee_point_search(x, y):
    # 转为list以支持负号索引
    x, y = x.tolist(), y.tolist()
    output_knees = []
    for curve in ['convex', 'concave']:
        for direction in ['increasing', 'decreasing']:
            model = KneeLocator(x=x, y=y, curve=curve, direction=direction, online=False)
            if model.knee != x[0] and model.knee != x[-1]:
                output_knees.append((model.knee, model.knee_y, curve, direction))

    if output_knees.__len__() != 0:
        print('发现拐点！')
        print(output_knees)
    else:
        print('未发现拐点！')

def select_MinPts(data,k):
    k_dist = []
    for i in range(data.shape[0]):
        dist = (((data[i] - data)**2).sum(axis=1)**0.5)
        dist.sort()
        k_dist.append(dist[k])
    return np.array(k_dist)

k = 3

# 数据集
dataSet = loadDataSet('redForClustering.txt', splitChar=',')

k_dist = select_MinPts(np.array(dataSet),k)
k_dist.sort()
k_dist = k_dist[::-1]

# 绘制散点图
plt.scatter(np.arange(100),k_dist[0:100],s=5,c='red')
# plt.scatter(np.arange(k_dist.shape[0]),k_dist,s=5,c='red')
# 绘制折线图
plt.plot(np.arange(100),k_dist[0:100])
# plt.plot(np.arange(k_dist.shape[0]),k_dist)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.title("红葡萄酒数据点第k最近邻距离排序折线图")
plt.xlabel("数据量")
plt.ylabel("距离")
plt.grid()  # 显示网格

knee_point_search(np.arange(k_dist.shape[0]),k_dist)

# 进行DBSCAN，返回聚类列表
C = dbscan(dataSet, 11, 10.04)
indexList = []
for i in range(len(set(C))):
    indexList.append([])

# 获得各簇的索引列表
for i in range(len(C)):
    indexList[C[i]+1].append(i)
print(indexList)

x = []
y = []
for data in dataSet:
    x.append(data[0])
    y.append(data[1])
plt.figure(figsize=(8, 6), dpi=80)
plt.scatter(x, y, c=C, marker='o')
plt.show()

print("红葡萄酒中的离群数据（劣质酒）索引为：")
print(indexList[0])

# 数据集
dataSet2 = loadDataSet('whiteForClustering.txt', splitChar=',')

k_dist2 = select_MinPts(np.array(dataSet2),k)
k_dist2.sort()
k_dist2 = k_dist2[::-1]

# # 绘制散点图
# plt.scatter(np.arange(100),k_dist2[0:100],s=5,c='red')
# # plt.scatter(np.arange(k_dist.shape[0]),k_dist,s=5,c='red')
# # 绘制折线图
# plt.plot(np.arange(100),k_dist2[0:100])
# # plt.plot(np.arange(k_dist.shape[0]),k_dist)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.title("白葡萄酒数据点第k最近邻距离排序折线图")
# plt.xlabel("数据量")
# plt.ylabel("距离")
# plt.grid()  # 显示网格

knee_point_search(np.arange(k_dist2.shape[0]),k_dist2)

# 进行DBSCAN，返回聚类列表
C2 = dbscan(dataSet2, 26, 11.98)

indexList2 = []
for i in range(len(set(C2))):
    indexList2.append([])

# 获得各簇的索引列表
for i in range(len(C2)):
    indexList2[C2[i]+1].append(i)
print(indexList2)

# x2 = []
# y2 = []
# for data in dataSet2:
#     x2.append(data[0])
#     y2.append(data[1])
# plt.figure(figsize=(8, 6), dpi=80)
# plt.scatter(x2, y2, c=C2, marker='o')
# plt.show()

print("白葡萄酒中的离群数据（劣质酒）索引为：")
print(indexList2[0])