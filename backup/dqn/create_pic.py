#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

# 设置汉字格式
# plt.rcParams['font.sans-serif'] = ['Times new roman']  # 指定默认字体
# plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号


def plot_data_loss():
    data = pd.read_csv("5/loss.txt", header=None)
    data.columns = ['loss']
    data_list = data['loss'].tolist()
    # 指定画布大小
    plt.figure(figsize=(10, 4))
    # 设置图标标题
    plt.title(u"Training loss")
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
    plt.savefig(f"5/training_loss.png")
    # 展示折线图
    # plt.show()

def plot_data_reward():
    data = pd.read_csv("5/reward.txt", header=None)
    data.columns = ['reward']
    data_list = data['reward'].tolist()
    # 指定画布大小
    plt.figure(figsize=(10, 4))
    # 设置图标标题
    plt.title(u"Training reward")
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
    plt.savefig(f"5/training_reward_3.png")
    # 展示折线图
    # plt.show()

def plot_data_q_value():
    data = pd.read_csv("5/q_value.txt", header=None)
    data.columns = ['q_value']
    data_list = data['q_value'].tolist()
    # 指定画布大小
    plt.figure(figsize=(10, 4))
    # 设置图标标题
    plt.title(u"Training q_value")
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
    plt.savefig(f"5/training_q_value.png")
    # 展示折线图
    # plt.show()

if __name__ == '__main__':
    plot_data_loss()
    plot_data_reward()
    plot_data_q_value()
