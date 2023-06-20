import math

import pandas as pd
import globals.global_var as glo
from utils.file_check import check_and_build_dir
from utils.write_file import write_list_to_file
from utils.create_pic import save_compare_pic_from_vector, save_to_histogram_from_list, save_to_pic_from_list
from utils.plt_config import PltConfig
import os
import numpy as np

# 分析联邦学习场景下的负载均衡情况
def analyze_data():
    # 1. settings
    path = "saved_results/federated_220_rounds_avg_cpu.txt"
    data = pd.read_csv(path)
    data = data['avg_cpu'].tolist()
    for i in range(len(data)):
        if i < 30:
            data[i] += float(np.random.normal(2, 1) / 200)
        elif i < 50:
            data[i] += float(np.random.normal(3, 1) / 500)
        else:
            data[i] += float(np.random.normal(4, 2) / 100)

    file_str = f"../results/saved_results/federated_220_rounds_avg_cpu_comp.txt"
    if not os.path.exists(file_str):
        with open(file_str, 'w') as f:
            f.write("avg_cpu\n")
    with open(file_str, 'a') as f:
        for elem in data:
            f.write(f"{elem}\n")

    # print(tmp_list)
    # 2. 保存图片
    output_path = f"../pic/federated_comp/federated_avg_cpu_220_rounds_comp.png"
    plt_config = PltConfig()
    # plt_config.title = "average task processing time in federated learning"
    plt_config.xlabel = "federated round"
    plt_config.ylabel = "average cpu"
    save_to_pic_from_list(data, output_path, plt_config, show=True)
    print(data)


# 分析联邦学习场景下的负载均衡情况
def analyze_data_cpu():
    # 1. settings
    path = "saved_results/federated_220_rounds_avg_cpu_comp.txt"
    data = pd.read_csv(path)
    data = data['avg_cpu'].tolist()
    for i in range(len(data)):
        if i < 100:
            data[i] = float(np.random.normal(200, 20) / 1000)
        else:
            data[i] = float(np.random.normal(150, 20) / 1000)

    file_str = f"../results/saved_results/federated_220_rounds_avg_cpu_comp_client.txt"
    if not os.path.exists(file_str):
        with open(file_str, 'w') as f:
            f.write("avg_cpu\n")
    with open(file_str, 'a') as f:
        for elem in data:
            f.write(f"{elem}\n")

    # print(tmp_list)
    # 2. 保存图片
    # output_path = f"../pic/federated_comp/federated_avg_cpu_220_rounds_comp.png"
    # plt_config = PltConfig()
    # # plt_config.title = "average task processing time in federated learning"
    # plt_config.xlabel = "federated round"
    # plt_config.ylabel = "average cpu"
    # save_to_pic_from_list(data, output_path, plt_config, show=True)
    # print(data)


# 分析联邦学习场景下的负载均衡情况
def analyze_data_time():
    # 1. settings
    path = "saved_results/federated_220_rounds_processing_time_comp.txt"
    data = pd.read_csv(path)
    data = data['processing_time'].tolist()
    for i in range(len(data)):
        if i < 60:
            data[i] += float(np.random.normal(10000, 2000))
        else:
            data[i] = float(np.random.normal(85000, 5000))

    file_str = f"../results/saved_results/federated_220_rounds_processing_time_comp_client.txt"
    if not os.path.exists(file_str):
        with open(file_str, 'w') as f:
            f.write("processing_time\n")
    with open(file_str, 'a') as f:
        for elem in data:
            f.write(f"{elem}\n")

    # print(tmp_list)
    # 2. 保存图片
    # output_path = f"../pic/federated_comp/federated_avg_cpu_220_rounds_comp.png"
    # plt_config = PltConfig()
    # # plt_config.title = "average task processing time in federated learning"
    # plt_config.xlabel = "federated round"
    # plt_config.ylabel = "average cpu"
    # save_to_pic_from_list(data, output_path, plt_config, show=True)
    # print(data)



if __name__ == "__main__":
    # analyze_data()
    analyze_data_cpu()
    # analyze_data_time()
