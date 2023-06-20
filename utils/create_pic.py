import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def save_to_pic(data_dir, dest_dir, plt_config, show=False):
    """Draw line chart

    file_format:    1.2
                    1.3
                    1.4
                    1.5

    """
    # 1. read data
    data = pd.read_csv(data_dir, header=None)

    # 2. plt configure
    plt.figure(figsize=(10, 6))
    plt.title(plt_config.title)
    plt.xlabel(plt_config.xlabel)
    plt.ylabel(plt_config.ylabel)
    y_axis_data = data[0].tolist()
    num = len(y_axis_data)
    x_axis_data = [i for i in range(num)]
    plt.plot(x_axis_data, y_axis_data)
    plt.savefig(dest_dir)
    if show:
        plt.show()
    plt.close()


def save_to_pic_from_list(data_list, dest_dir, plt_config, show=False):
    """Draw line chart from data_list

    data_list format: [1.1, 1.2, 1.3, 1.4]

    """
    # plt.rcParams["font.family"] = "Arial"
    # plt.rcParams["font.size"] = "18"
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams["font.size"] = "18"

    plt.figure(figsize=(13, 6))
    plt.title(plt_config.title)
    plt.xlabel(plt_config.xlabel)
    plt.ylabel(plt_config.ylabel)
    y_axis_data = data_list
    num = len(y_axis_data)
    x_axis_data = [i for i in range(num)]
    plt.plot(x_axis_data, y_axis_data)

    plt.tick_params(labelsize=20)
    plt.savefig(dest_dir)
    if show:
        plt.show()
    plt.close()


def save_compare_pic_from_vector(data_vector, labels, dest_dir, plt_config, show=False):
    """Draw line chart from data_list

    data_list format: [[1.1, 1.2, 1.3, 1.4],
                       [2.1, 2.2, 2.3, 2.4]]

    """
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = "20"
    plt.figure(figsize=(13, 7))
    # plt.title(plt_config.title)
    plt.xlabel(plt_config.xlabel, fontproperties="SimHei", fontsize=20)
    plt.ylabel(plt_config.ylabel, fontproperties="SimHei", fontsize=20)
    # x_axis_data = plt_config.x_axis_data

    n_schedulers = len(data_vector)
    linewidth = 3
    # color_list = ['blue', 'red', 'slategrey', 'orange', 'lightskyblue', 'green', 'lightgrey', 'slateblue']
    color_list = ['green', 'red', 'slategrey', 'orange', 'lightskyblue', 'green', 'lightgrey', 'slateblue']

    for i in range(n_schedulers):
        if i % 2 == 0:
            plt.plot(data_vector[i], label=labels[i], linewidth=linewidth, color=color_list[i], linestyle="-")
        else:
            plt.plot(data_vector[i], label=labels[i], linewidth=linewidth, color=color_list[i], linestyle="-")
        # plt.plot(x_axis_data, data_vector[i], label=labels[i], linewidth=linewidth, color=color_list[i])

    # 设置图例
    plt.legend(loc='best')
    plt.tick_params(labelsize=20)
    plt.savefig(dest_dir)
    if show:
        plt.show()
    plt.close()


def save_to_histogram(data_dir, dest_dir, plt_config, show=False):
    """Draw histogram

    file_format:    1.2
                    1.3
                    1.4
                    1.5

    """
    # 1. read data
    data = pd.read_csv(data_dir)

    # 2. plt configure
    plt.figure(figsize=(10, 6))
    plt.title(plt_config.title)
    plt.xlabel(plt_config.xlabel)
    plt.ylabel(plt_config.ylabel)
    y_axis_data = data[0].tolist()
    x_axis_data = plt_config.x_axis_data

    # 设置柱形图的柱子不同颜色，最高的柱子与最低的柱子突出显示
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

    plt.savefig(dest_dir)
    if show:
        plt.show()
    plt.close()


def save_to_histogram_from_list(data_list, dest_dir, plt_config, show=False, show_text=True):
    """Draw histogram

    file_format:    1.2
                    1.3
                    1.4
                    1.5

    """
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = "20"
    plt.figure(figsize=(13, 7))
    plt.title(plt_config.title)
    plt.xlabel(plt_config.xlabel, fontproperties="SimHei", fontsize=20)
    plt.ylabel(plt_config.ylabel, fontproperties="SimHei", fontsize=20)
    y_axis_data = data_list
    x_axis_data = plt_config.x_axis_data

    # 设置柱形图的柱子不同颜色，最高的柱子与最低的柱子突出显示
    color_list = []
    for i in range(len(x_axis_data)):
        color_list.append('lightskyblue')

    # 画柱形图
    plt.bar(x=x_axis_data, height=y_axis_data, width=0.5, color=color_list, alpha=0.8)

    # 在柱形图上显示具体数值，ha参数控制水平对齐方式，va控制垂直对齐方式
    # zip()将可迭代的对象中的对应元素打包成一个元组，然后返回这些元组组成的列表
    # 例：zip([1, 2, 3], [4, 5, 6])返回[(1, 4), (2, 5), (3, 6)]
    # if show_text:
    #     z_xy = zip(x_axis_data, y_axis_data)
    #     for xx, yy in z_xy:
    #         if yy == 0:
    #             continue
    #         plt.text(xx, yy - 100, round(yy, 2), ha='center', va='bottom', fontsize=24, rotation=45)
            # plt.text(xx, yy - 40, round(yy, 2), ha='center', va='bottom', fontsize=10, rotation=45)
            # plt.text(xx, yy - 10, str(round(yy, 2)) + '%', ha='center', va='bottom', fontsize=11, rotation=45)

    plt.savefig(dest_dir)
    if show:
        plt.show()
    plt.close()


def save_two_histogram_from_vector(data_vector, dest_path, plt_config, show=False, show_text=True):
    """Draw histogram

    file_format:    1.2 2.2 3.2
                    1.3 2.3 3.3
                    1.4 2.4 3.4
                    1.5 2.5 3.5

    """
    x_axis_data = plt_config.x_axis_data
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(plt_config.x_axis_data))
    
    width = 0.3
    ax1 = ax.bar(x - width / 2, height=data_vector[0], width=width, alpha=0.8, label='cpu-time')
    ax2 = ax.bar(x + width / 2, height=data_vector[1], width=width, alpha=0.8, label='io-time')
    ax.legend()

    ax.set_title(plt_config.title)
    ax.set_xlabel(plt_config.xlabel)
    ax.set_ylabel(plt_config.ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_axis_data)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    if show_text:
        autolabel(ax1)
        autolabel(ax2)
    fig.tight_layout()
    plt.savefig(dest_path)
    if show:
        plt.show()
    plt.close()
    

def save_multiple_histogram_from_vector(data_vector, dest_path, plt_config, show=False, show_text=True):
    """Draw histogram

    file_format:    1.2 2.2 3.2
                    1.3 2.3 3.3
                    1.4 2.4 3.4
                    1.5 2.5 3.5

    """
    x_axis_data = plt_config.x_axis_data
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(plt_config.x_axis_data))
    
    width = 0.3
    ax1 = ax.bar(x - width, height=data_vector[0], width=width, alpha=0.8, label='cpu-intensive')
    ax2 = ax.bar(x, height=data_vector[1], width=width, alpha=0.8, label='io-intensive')
    ax3 = ax.bar(x + width, height=data_vector[2], width=width, alpha=0.8, label='simple')
    ax.legend()

    ax.set_title(plt_config.title)
    ax.set_xlabel(plt_config.xlabel)
    ax.set_ylabel(plt_config.ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_axis_data)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    if show_text:
        autolabel(ax1)
        autolabel(ax2)
        autolabel(ax3)
    fig.tight_layout()
    plt.savefig(dest_path)
    if show:
        plt.show()
    plt.close()


def save_multiple_histogram_compare_pic_from_vector(data_vector, dest_dir, plt_config, show=False, show_text=True):
    """Draw histogram

    file_format:    1.2 2.2 3.2
                    1.3 2.3 3.3
                    1.4 2.4 3.4
                    1.5 2.5 3.5

    """
    x_axis_data = plt_config.x_axis_data
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(plt_config.x_axis_data))
    
    width = 0.2
    ax1 = ax.bar(x - width, height=data_vector[0], width=width, alpha=0.8, label='cpu-intensive')
    ax2 = ax.bar(x, height=data_vector[1], width=width, alpha=0.8, label='io-intensive')
    ax3 = ax.bar(x + width, height=data_vector[2], width=width, alpha=0.8, label='simple')
    ax.legend()

    ax.set_title(plt_config.title)
    ax.set_xlabel(plt_config.xlabel)
    ax.set_ylabel(plt_config.ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_axis_data)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    if show_text:
        autolabel(ax1)
        autolabel(ax2)
        autolabel(ax3)
    fig.tight_layout()
    plt.savefig(dest_dir)
    if show:
        plt.show()
    plt.close()


def save_to_pie_from_list(data_list, dest_dir, plt_config, show=False):
    """Draw pie

    list format: [1.1, 1.2, 1.3]

    """
    plt.figure(figsize=(10, 6))
    plt.title(plt_config.title)
    y_axis_data = data_list
    labels = plt_config.labels

    # 设置柱形图的柱子不同颜色，最高的柱子与最低的柱子突出显示
    color_list = ["#d5695d", "#5d8ca8", "#65a479"]

    # 画饼图
    plt.pie(y_axis_data, labels=labels, colors=color_list, autopct='%.2f')

    plt.savefig(dest_dir)
    if show:
        plt.show()
    plt.close()
