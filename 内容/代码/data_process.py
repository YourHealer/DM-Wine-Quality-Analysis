import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imshow
from sklearn import preprocessing

# 原始数据
data = pd.read_csv('F:/学习/数据仓库与数据挖掘/作业/数据/winequalityN.csv')
data.columns = ['type', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                'sulphates', 'alcohol', 'quality']

# 去重后的红葡萄酒、白葡萄酒数据
data1 = pd.read_csv('F:/学习/数据仓库与数据挖掘/作业/数据/red.csv')
data2 = pd.read_csv('F:/学习/数据仓库与数据挖掘/作业/数据/white.csv')
data1.columns = ['type', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                'sulphates', 'alcohol', 'quality']
data2.columns = ['type', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                'sulphates', 'alcohol', 'quality']

# 分类绘制去重后数据的箱线图：红葡萄酒 白葡萄酒
# fig, ax = plt.subplots(1,3,constrained_layout=True, figsize=(14, 4),)
# fixed acidity   volatile   acidity citric acid
# residual sugar   chlorides   free sulfur dioxide
# total sulfur dioxide   density   pH
# sulphates	  alcohol   quality

# x11 = np.array(data1['fixed acidity']).tolist()
# x12 = np.array(data2['fixed acidity']).tolist()
# axesSub = sns.boxplot(data=[x11, x12], width=0.5, ax=ax[0])
# x = ['red wine', 'white wine']
# axesSub.set_xticks(range(len(x)), x)
# axesSub.set_title('fixed acidity')
# axesSub.grid()
#
# x21 = np.array(data1['volatile acidity']).tolist()
# x22 = np.array(data2['volatile acidity']).tolist()
# axesSub = sns.boxplot(data=[x21, x22], width=0.5, ax=ax[1])
# x = ['red wine', 'white wine']
# axesSub.set_xticks(range(len(x)), x)
# axesSub.set_title('volatile acidity')
# axesSub.grid()
#
# x31 = np.array(data1['citric acid']).tolist()
# x32 = np.array(data2['citric acid']).tolist()
# axesSub = sns.boxplot(data=[x31, x32], width=0.5, ax=ax[2])
# x = ['red wine', 'white wine']
# axesSub.set_xticks(range(len(x)), x)
# axesSub.set_title('citric acid')
# axesSub.grid()

# x11 = np.array(data1['residual sugar']).tolist()
# x12 = np.array(data2['residual sugar']).tolist()
# axesSub = sns.boxplot(data=[x11, x12], width=0.5, ax=ax[0])
# x = ['red wine', 'white wine']
# axesSub.set_xticks(range(len(x)), x)
# axesSub.set_title('residual sugar')
# axesSub.grid()
#
# x21 = np.array(data1['chlorides']).tolist()
# x22 = np.array(data2['chlorides']).tolist()
# axesSub = sns.boxplot(data=[x21, x22], width=0.5, ax=ax[1])
# x = ['red wine', 'white wine']
# axesSub.set_xticks(range(len(x)), x)
# axesSub.set_title('chlorides')
# axesSub.grid()
#
# x31 = np.array(data1['free sulfur dioxide']).tolist()
# x32 = np.array(data2['free sulfur dioxide']).tolist()
# axesSub = sns.boxplot(data=[x31, x32], width=0.5, ax=ax[2])
# x = ['red wine', 'white wine']
# axesSub.set_xticks(range(len(x)), x)
# axesSub.set_title('free sulfur dioxide')
# axesSub.grid()

# x11 = np.array(data1['total sulfur dioxide']).tolist()
# x12 = np.array(data2['total sulfur dioxide']).tolist()
# axesSub = sns.boxplot(data=[x11, x12], width=0.5, ax=ax[0])
# x = ['red wine', 'white wine']
# axesSub.set_xticks(range(len(x)), x)
# axesSub.set_title('total sulfur dioxide')
# axesSub.grid()
#
# x21 = np.array(data1['density']).tolist()
# x22 = np.array(data2['density']).tolist()
# axesSub = sns.boxplot(data=[x21, x22], width=0.5, ax=ax[1])
# x = ['red wine', 'white wine']
# axesSub.set_xticks(range(len(x)), x)
# axesSub.set_title('density')
# axesSub.grid()
#
# x31 = np.array(data1['pH']).tolist()
# x32 = np.array(data2['pH']).tolist()
# axesSub = sns.boxplot(data=[x31, x32], width=0.5, ax=ax[2])
# x = ['red wine', 'white wine']
# axesSub.set_xticks(range(len(x)), x)
# axesSub.set_title('pH')
# axesSub.grid()

# x11 = np.array(data1['sulphates']).tolist()
# x12 = np.array(data2['sulphates']).tolist()
# axesSub = sns.boxplot(data=[x11, x12], width=0.5, ax=ax[0])
# x = ['red wine', 'white wine']
# axesSub.set_xticks(range(len(x)), x)
# axesSub.set_title('sulphates')
# axesSub.grid()
#
# x21 = np.array(data1['alcohol']).tolist()
# x22 = np.array(data2['alcohol']).tolist()
# axesSub = sns.boxplot(data=[x21, x22], width=0.5, ax=ax[1])
# x = ['red wine', 'white wine']
# axesSub.set_xticks(range(len(x)), x)
# axesSub.set_title('alcohol')
# axesSub.grid()
#
# x31 = np.array(data1['quality']).tolist()
# x32 = np.array(data2['quality']).tolist()
# axesSub = sns.boxplot(data=[x31, x32], width=0.5, ax=ax[2])
# x = ['red wine', 'white wine']
# axesSub.set_xticks(range(len(x)), x)
# axesSub.set_title('quality')
# axesSub.grid()
# plt.show()

# 分类了解数据的数据特征
# 红酒、白酒
# y = []
# x = ['Mean', 'Std', 'Min', 'Max']
# for col in data2.columns:
#     tmp = []
#     if col != 'type':
#         # tmp.append(data[col].count()/6497.0)  # 非空值比例valid
#         tmp.append(data2[col].mean()) # 均值
#         tmp.append(data2[col].std())  # 标准差
#         tmp.append(data2[col].min())  # 最小值
#         tmp.append(data2[col].max())  # 最大值
#         y.append(tmp)
#
# fig = plt.figure(figsize=(12, 6))
# color_list = ['seagreen', 'cornflowerblue', 'lightsalmon', 'orangered']
# for i in range(0, 6):
#     # for j in range(len(x)):
#     ax = plt.subplot(2, 3, i+1)
#     plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.5,
#                         hspace=0.15)  # wspace 子图横向间距， hspace 代表子图间的纵向距离，left 代表位于图像不同位置
#     b = ax.barh(x, y[i + 6], color=color_list)
#     # plt.axis('off')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     # ax.spines['left'].set_visible(False)
#     for rect in b:
#         w = rect.get_width()
#         ax.text(w, rect.get_y() + rect.get_height() / 2, '%.3f' % w, ha='left', va='center',fontsize=10)
#     # ax.set_yticks(range(len(x)))
#     # ax.set_yticklabels(x)
#     ax.set_xticks([])
#     plt.title(data.columns[i + 7])
#
# plt.show()

# 分类处理3σ
# 红酒的各属性3σ左右边界列表
# list1 = []
# for col in data1.columns:
#     if col != 'type':
#         left=data1[col].mean()-3*data1[col].std()
#         right=data1[col].mean()+3*data1[col].std()
#         if(left<0):
#             left=0
#         list1.append([left,right])

# 白酒的各属性3σ左右边界列表
# list2 = []
# for col in data2.columns:
#     if col != 'type':
#         left=data2[col].mean()-3*data2[col].std()
#         right=data2[col].mean()+3*data2[col].std()
#         if(left<0):
#             left=0
#         list2.append([left,right])
#
# print("红葡萄酒的3σ区间为:",end='')
# print(list1)
# print("白葡萄酒的3σ区间为:",end='')
# print(list2)

# 获取在范围内的数据
# (left<num)&(num<right)的数据为合理的
# 红酒中各属性不在3σ范围内的数据索引列表
# list11=[]
# cnt=-1
# for col in data1.columns:
#     if col != 'type':
#         cnt += 1
#         mylist=[]
#         for i in range(len(data1[col])):
#             if(list1[cnt][1] < data1[col][i]) | (data1[col][i] < list1[cnt][0]):
#                 mylist.append(i)
#         list11.append(mylist)

# for i in range(len(list11)):
#     print(data1.columns[i+1],end=': ')
#     print(list11[i])

# 红酒中各属性不在3σ范围内的数据索引列表
# list21=[]
# cnt=-1
# for col in data2.columns:
#     if col != 'type':
#         cnt += 1
#         mylist=[]
#         for i in range(len(data2[col])):
#             if(list2[cnt][1] < data2[col][i]) | (data2[col][i] < list2[cnt][0]):
#                 mylist.append(i)
#         list21.append(mylist)
#
# # for i in range(len(list21)):
# #     print(data2.columns[i+1],end=': ')
# #     print(list21[i])
# # 获得的异常值索引都在list11 和 list21里
#
# indexTobeDelete = set()
# for i in list11[1]:
#     if(data1['volatile acidity'][i] > 1.2):
#         indexTobeDelete.add(i)
#
# for i in list11[3]:
#     if(data1['residual sugar'][i] <= 45): #非甜葡萄酒
#         if(data1['citric acid'][i] > 1):
#             indexTobeDelete.add(i)
#     else:
#         if(data1['citric acid'][i] > 2):
#             indexTobeDelete.add(i)
#
# print("红葡萄酒中不符合国标要求的数据索引为：")
# print(indexTobeDelete)
#
# indexTobeDelete2 = set()

# for i in list21[3]:
#     if(data2['residual sugar'][i] <= 45): #非甜葡萄酒
#         if(data2['citric acid'][i] > 1):
#             indexTobeDelete2.add(i)
#     else:
#         if(data2['citric acid'][i] > 2):
#             indexTobeDelete2.add(i)
#
# print("白葡萄酒中不符合国标要求的数据索引为：")
# print(indexTobeDelete2)
#
# print("红葡萄酒数据原先实例数： %d" % data1.shape[0])
# data1 = data1.drop(index=indexTobeDelete,axis=0)
# print("红葡萄酒数据删去异常值后实例数： %d" % data1.shape[0])
#
# data1.to_csv(path_or_buf="C:/Users/ASUS/Desktop/redwineNew.csv")
# data2.to_csv(path_or_buf="C:/Users/ASUS/Desktop/whitewineNew.csv")

# PCA主成分分析
# data是一个矩阵  left_dim是要缩减到的维度
def pca(data, left_dim):
    # mean_vector是每一列的均值向量
    mean_vector = np.mean(data, axis=0)
    # normal_data是减掉均值后中心化得到的矩阵
    normal_data = data - mean_vector
    # Covmatrix是对中心化后的数据计算得到的协方差矩阵
    Covmatrix = np.cov(normal_data, rowvar=False)
    # eig_val, eig_vec分别是对协方差矩阵进行对角化算出的特征值与特征向量
    eig_val, eig_vec = np.linalg.eig(Covmatrix)
    print(eig_val)
    eig_val_sorted = sorted(eig_val)[::-1]
    print(eig_val_sorted)

    # 贡献率
    contriRate = eig_val_sorted / np.sum(eig_val_sorted)
    # 累积贡献率
    sumContriRate = np.cumsum(contriRate)
    print(sumContriRate)
    # 前两个主成分解释了原数据99.5%的方差，所以用两个主成分几乎可以完全代替原来11个变量。

    # eigIndexOld是根据特征值从小到大排序得到的特征值索引列表
    eigIndexOld = np.argsort(eig_val)
    # 切片操作基本表达式：object[start_index : end_index : step]  eigIndexNew获得最大的left_dim个特征值的索引列表
    eigIndexNew = eigIndexOld[:-(1+left_dim):-1]
    # eig_face是根据最大的几个特征值筛选出来的特征向量组成的用于降维的矩阵eig_face n×r维 （原维度,新维度）
    eig_tool = eig_vec[:, eigIndexNew]
    # new_data是利用矩阵乘法将中心化后的数据进行降维后得到的结果 数据量*新维度
    new_data = np.dot(normal_data, eig_tool)
    # 返回 用于降维的矩阵、排序后的特征值向量、降维后的矩阵
    return eig_tool, eig_val_sorted, new_data

# # 红酒数据
# # val_data是数据量*原维度的矩阵
# val_data=pd.read_csv("F:/学习/数据仓库与数据挖掘/作业/数据/redwineNew.csv")
# # 除去“属性”字段
# val_data=val_data.drop(columns='type')
# # 除去“质量”字段
# val_data=val_data.drop(columns='quality')
#
# dim = 2
# #  数据量*新维度
# eig_face, eig_val_sorted, new_data = pca(val_data,left_dim=dim)
# # 主成分矩阵
# print(eig_face)
# # 新矩阵
# print(new_data)
#
# # 同样的数据绘制散点图和折线图
# plt.scatter(range(1, val_data.shape[1] + 1), eig_val_sorted)
# plt.plot(range(1, val_data.shape[1] + 1), eig_val_sorted)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.title("字段索引值-特征值折线图")
# plt.xlabel("字段索引值")
# plt.ylabel("特征值")
# plt.grid()  # 显示网格
# plt.show()  # 显示图形
#
# pd.DataFrame(new_data).to_csv("F:/学习/数据仓库与数据挖掘/作业/数据/redwinePCA.csv",index=0,header=['PC1','PC2'])

# # 白酒数据
# # val_data是数据量*原维度的矩阵
# val_data=pd.read_csv("F:/学习/数据仓库与数据挖掘/作业/数据/whitewineNew.csv")
# # 除去”属性“字段
# val_data=val_data.drop(columns='type')
# # 除去“质量”字段
# val_data=val_data.drop(columns='quality')
#
# dim = 2
# #  数据量*新维度
# eig_tool, eig_val_sorted, new_data = pca(val_data,left_dim=dim)
# # 主成分矩阵
# print(eig_tool)
# # 新矩阵
# print(new_data)
#
# # 同样的数据绘制散点图和折线图
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.scatter(range(1, val_data.shape[1] + 1), eig_val_sorted)
# plt.plot(range(1, val_data.shape[1] + 1), eig_val_sorted)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.title("字段索引值-特征值折线图")
# plt.xlabel("字段索引值")
# plt.ylabel("特征值")
# plt.grid()  # 显示网格
# plt.show()  # 显示图形
#
# pd.DataFrame(new_data).to_csv("F:/学习/数据仓库与数据挖掘/作业/数据/whitewinePCA1.csv",index=0,header=['PC1','PC2'])

# val_data_red=pd.read_csv("F:/学习/数据仓库与数据挖掘/作业/数据/redwineNew.csv")
# val_data_white=pd.read_csv("F:/学习/数据仓库与数据挖掘/作业/数据/whitewineNew.csv")
# # 除去”属性“字段
# val_data_red=val_data_red.drop(columns='type')
# val_data_white=val_data_white.drop(columns='type')
# # 除去“质量”字段
# val_data_red=val_data_red.drop(columns='quality')
# val_data_white=val_data_white.drop(columns='quality')
#
# Zscore_red = pd.DataFrame(preprocessing.scale(val_data_red))
# Zscore_white = pd.DataFrame(preprocessing.scale(val_data_white))
# Zscore_red.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
#                 'sulphates', 'alcohol']
# Zscore_white.columns = Zscore_red.columns
# print(Zscore_red)
# print(Zscore_white)