import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 原始数据
data = pd.read_csv('F:/学习/数据仓库与数据挖掘/作业/数据/winequalityN.csv')
data.columns = ['type', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                'sulphates', 'alcohol', 'quality']

# 分类：红葡萄酒 白葡萄酒
data1 = pd.read_csv('F:/学习/数据仓库与数据挖掘/作业/数据/red_wine_original.csv')
data2 = pd.read_csv('F:/学习/数据仓库与数据挖掘/作业/数据/white_wine_original.csv')
data1.columns = ['type', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                'sulphates', 'alcohol', 'quality']


# 统计特征：条形图
y = []
x = ['Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
for col in data.columns:
    tmp = []
    if col != 'type':
        # tmp.append(data[col].count()/6497.0)  # 非空值比例valid
        tmp.append(data[col].mean())  # 均值
        tmp.append(data[col].std())  # 标准差
        tmp.append(data[col].min())  # 最小值
        tmp.append(data[col].quantile(0.25))  # 25%
        tmp.append(data[col].median())  # 50% 中位数
        tmp.append(data[col].quantile(0.75))  # 75%
        tmp.append(data[col].max())  # 最大值
        y.append(tmp)

fig = plt.figure(figsize=(12, 6))
color_list = ['seagreen', 'cornflowerblue', 'lightsalmon', 'darksalmon', 'coral', 'tomato', 'orangered']
for i in range(0, 6):
    # for j in range(len(x)):
    ax = plt.subplot(2, 3, i+1)
    plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.3,
                        hspace=0.15)  # wspace 子图横向间距， hspace 代表子图间的纵向距离，left 代表位于图像不同位置
    b = ax.barh(x, y[i], color=color_list)
    # plt.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    for rect in b:
        w = rect.get_width()
        ax.text(w, rect.get_y() + rect.get_height() / 2, '%.3f' % w, ha='left', va='center')
    # ax.set_yticks(range(len(x)))
    # ax.set_yticklabels(x)
    ax.set_xticks([])
    plt.title(data.columns[i + 1])

plt.show()

# 箱线图
# fig, ax =plt.subplots(1,3,constrained_layout=True, figsize=(12, 4))
# axesSub = sns.boxplot(data=data['fixed acidity'], width=0.4, ax=ax[0])
# axesSub.set_title('fixed acidity')
# axesSub.grid()
# axesSub = sns.boxplot(data=data['volatile acidity'], width=0.4, ax=ax[1])
# axesSub.set_title('volatile acidity')
# axesSub.grid()
# axesSub = sns.boxplot(data=data['citric acid'], width=0.4, ax=ax[2])
# axesSub.set_title('citric acid')
# axesSub.grid()
#
# plt.show()

# 分类绘制箱线图：红葡萄酒 白葡萄酒
fig, ax =plt.subplots(1,3,constrained_layout=True, figsize=(12, 4))
x11 = np.array(data1['fixed acidity'])
x12 = np.array(data2['fixed acidity'])
axesSub = sns.boxplot(data=[x11, x12], width=0.5, ax=ax[0])
x = ['red wine', 'white wine']
axesSub.set_xticks(range(len(x)), x)
axesSub.set_title('fixed acidity')
axesSub.grid()
x21 = np.array(data1['volatile acidity'])
x22 = np.array(data2['volatile acidity'])
axesSub = sns.boxplot(data=[x21, x22], width=0.5, ax=ax[1])
x = ['red wine', 'white wine']
axesSub.set_xticks(range(len(x)), x)
axesSub.set_title('volatile acidity')
axesSub.grid()
x31 = np.array(data1['citric acid'])
x32 = np.array(data2['citric acid'])
axesSub = sns.boxplot(data=[x31, x32], width=0.5, ax=ax[2])
x = ['red wine', 'white wine']
axesSub.set_xticks(range(len(x)), x)
axesSub.set_title('citric acid')
axesSub.grid()
plt.show()

# 饼图
# # 删除缺失值>1的实例
# data = data.dropna(thresh=12)
# # 用特征指标的中位数填充含1个缺失值的列
# for col in data.columns:
#     if data[col].isnull().sum(axis=0) != 0:
#         data[col] = data[col].fillna(data[col].median())
# # 去重，计算重复率
# data_before = data.shape[0]
# data = data.drop_duplicates()
# data.to_csv('winequalityPreprocessed.csv')
# data_after = data.shape[0]
# print(data_after/data_before)
# sizes = [data_after/data_before, 1-data_after/data_before]
# labels = ['Unique values', 'Duplicate values']
# colors = ['lightblue', 'steelblue']
# explode = (0, 0.15)
# plt.pie(sizes, labels=labels, autopct='%3.1f%%',
#         explode=explode, colors=colors, shadow=True, startangle=90)
# plt.title('Repeatability')
# plt.show()


# 直方图+密度图
# i = 0
# plt.figure(figsize=(12, 3))
# for col in data.columns:
#     if 1 <= i < 4:
#         ax = plt.subplot(1, 3, i)
#         data[col].hist(color='darkseagreen', bins = 50)
#         # data[col].plot(kind='hist', color='darkseagreen', edgecolor = 'seagreen',
#         #                density=True, bins = 30)
#         # data[col].plot(kind='kde', color='darkgreen')
#         plt.title(col)
#     i += 1
# plt.show()


# 热力图
corr = data.corr()
# print(x)
fig = plt.figure(figsize = (9,9))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
# with sns.axes_style("white"):
#      ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True)
sns.heatmap(corr, annot=True, vmax=1,vmin = 0, mask=mask,
            xticklabels= True, yticklabels= True, square=True, cmap="Greens")  # YlGnBu
plt.title('Wine Dataset Heat Map')
# plt.savefig('D:/Study/数据仓库与数据挖掘/大作业/可视化图片/heatmap3.png')
plt.show()