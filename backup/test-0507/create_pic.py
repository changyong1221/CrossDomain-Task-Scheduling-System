#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

# 设置汉字格式
# plt.rcParams['font.sans-serif'] = ['Times new roman']  # 指定默认字体
# plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号


def plot_data_loss():
    data = pd.read_csv("DQN3/test/loss.txt", header=None)
    data.columns = ['loss']
    data_list = data['loss'].tolist()
    # 指定画布大小
    plt.figure(figsize=(10, 4))
    # 设置图标标题
    plt.title(u"testing loss")
    # 设置图表x轴名称和y轴名称
    plt.xlabel(u"epoch")
    plt.ylabel("loss")
    # 将数据进行处理，以便于画图（需要将DataFrame转为List）
    # x轴数据，只需一份即可
    linewidth = 2.5
    plt.plot(data_list, linewidth=linewidth)
    # 设置图例
    # plt.legend(loc='best')
    # 保存图片
    plt.savefig(f"DQN3/test/testing_loss.png")
    # 展示折线图
    # plt.show()

def plot_data_reward():
    data = pd.read_csv("DQN3/test/reward.txt", header=None)
    data.columns = ['reward']
    data_list = data['reward'].tolist()
    # 指定画布大小
    plt.figure(figsize=(10, 4))
    # 设置图标标题
    plt.title(u"testing reward")
    # 设置图表x轴名称和y轴名称
    plt.xlabel(u"epoch")
    plt.ylabel("reward")
    # 将数据进行处理，以便于画图（需要将DataFrame转为List）
    # x轴数据，只需一份即可
    linewidth = 2.5
    data_list = data_list[:10000]
    plt.plot(data_list, linewidth=linewidth)
    # 设置图例
    # plt.legend(loc='best')
    # 保存图片
    plt.savefig(f"DQN3/test/testing_reward_3.png")
    # 展示折线图
    # plt.show()

def plot_data_q_value():
    data = pd.read_csv("DQN3/test/q_value.txt", header=None)
    data.columns = ['q_value']
    data_list = data['q_value'].tolist()
    # 指定画布大小
    plt.figure(figsize=(10, 4))
    # 设置图标标题
    plt.title(u"testing q_value")
    # 设置图表x轴名称和y轴名称
    plt.xlabel(u"epoch")
    plt.ylabel("q_value")
    # 将数据进行处理，以便于画图（需要将DataFrame转为List）
    # x轴数据，只需一份即可
    linewidth = 2.5
    plt.plot(data_list, linewidth=linewidth)
    # 设置图例
    # plt.legend(loc='best')
    # 保存图片
    plt.savefig(f"DQN3/test/testing_q_value.png")
    # 展示折线图
    # plt.show()

def plot_action():
    data = pd.read_csv("DQN3/test/action.txt", header=None, delimiter='\t')
    data.columns = ['task_mi', 'action']
    data_list = data['action'].tolist()
    # 指定画布大小
    plt.figure(figsize=(10, 4))
    # 设置图标标题
    plt.title(u"testing action")
    # 设置图表x轴名称和y轴名称
    plt.xlabel(u"epoch")
    plt.ylabel("action")
    # 将数据进行处理，以便于画图（需要将DataFrame转为List）
    # x轴数据，只需一份即可
    # 设置柱形图的柱子不同颜色，最高的柱子与最低的柱子突出显示
    x_axis_data = [str(i) for i in range(20)]
    y_axis_data = []
    for i in range(20):
        y_axis_data.append(data_list.count(i))

    color_list = []
    for i in range(len(x_axis_data)):
        color_list.append('lightskyblue')

    # 画柱形图
    plt.bar(x=x_axis_data, height=y_axis_data, width=0.5, color=color_list, alpha=0.8)

    # 在柱形图上显示具体数值，ha参数控制水平对齐方式，va控制垂直对齐方式
    # zip()将可迭代的对象中的对应元素打包成一个元组，然后返回这些元组组成的列表
    # 例：zip([1, 2, 3], [4, 5, 6])返回[(1, 4), (2, 5), (3, 6)]
    z_xy = zip(x_axis_data, y_axis_data)
    for xx, yy in z_xy:
        plt.text(xx, yy-0.008, str(round(yy, 4)), ha='center', va='bottom', fontsize=12, rotation=45)
    # 设置图例
    # plt.legend(loc='best')
    # 保存图片
    plt.savefig(f"DQN3/test/testing_action.png")
    # 展示折线图
    # plt.show()

if __name__ == '__main__':
    plot_data_loss()
    plot_data_reward()
    plot_data_q_value()
    plot_action()