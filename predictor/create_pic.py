#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 设置汉字格式
# plt.rcParams['font.sans-serif'] = ['Times new roman']  # 指定默认字体
# plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号


def plot_data_loss():
    data = pd.read_csv("history.csv", header=0, index_col=0)
    loss_list = data['loss'].tolist()
    val_loss_list = data['val_loss'].tolist()
    # 指定画布大小
    plt.figure(figsize=(10, 4))
    # 设置图标标题
    plt.title(u"loss graph")
    # 设置图表x轴名称和y轴名称
    plt.xlabel(u"epoch")
    plt.ylabel("loss")
    # 将数据进行处理，以便于画图（需要将DataFrame转为List）
    # x轴数据，只需一份即可x
    linewidth = 2.5
    plt.plot(loss_list, linewidth=linewidth, label='train loss')
    # plt.plot(val_loss_list, linewidth=linewidth, label='val loss')
    # 设置图例
    plt.legend(loc='best')
    # 保存图片
    plt.savefig(f"loss_graph_0.png")
    # 展示折线图
    # plt.show()


def plot_data_acc():
    data = pd.read_csv("history.csv", header=0, index_col=0)
    loss_list = data['accuracy'].tolist()
    val_loss_list = data['val_accuracy'].tolist()
    # 指定画布大小
    plt.figure(figsize=(10, 4))
    # 设置图标标题
    plt.title(u"accuracy graph")
    # 设置图表x轴名称和y轴名称
    plt.xlabel(u"epoch")
    plt.ylabel("accuracy")
    # 将数据进行处理，以便于画图（需要将DataFrame转为List）
    # x轴数据，只需一份即可x
    linewidth = 2.5
    plt.plot(loss_list, linewidth=linewidth, label='train accuracy')
    # plt.plot(val_loss_list, linewidth=linewidth, label='val accuracy')
    # 设置图例
    plt.legend(loc='best')
    # 保存图片
    plt.savefig(f"accuracy_graph_0.png")
    # 展示折线图
    # plt.show()

if __name__ == '__main__':
    plot_data_loss()
    plot_data_acc()