import pandas as pd
import globals.global_var as glo
from utils.file_check import check_and_build_dir
from utils.write_file import write_list_to_file
from utils.create_pic import save_compare_pic_from_vector, save_to_histogram_from_list, save_to_pic_from_list
from utils.plt_config import PltConfig
import os

# 分析联邦学习场景下的平均任务处理时间（算法对比）
def analyze_federated_task_processing_time_results():
    # 1. settings
    path = "results/saved_results/"
    path_list = [
            path + "federated_220_rounds_processing_time_comp.txt",
            path + "federated_220_rounds_processing_time.txt"
        ]

    show_vector = []
    # 1. read data
    for data_path in path_list:
        data = pd.read_csv(data_path)
        print(len(data))
        show_vector.append(data['processing_time'].to_list())

    # 2. 保存图片
    output_path = f"pic/federated_comp/federated_task_processing_time_comp_version.png"
    plt_config = PltConfig()
    plt_config.title = "task processing time in federated learning"
    plt_config.xlabel = "federated round"
    plt_config.ylabel = "task processing time"
    labels = ["fed-avg", "fed-ad-pa"]
    save_compare_pic_from_vector(show_vector, labels, output_path, plt_config, show=False)


if __name__ == "__main__":
    analyze_federated_task_processing_time_results()