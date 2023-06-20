import numpy as np
import pandas as pd
import sys
sys.path.append("/home/scy/CrossDomain-Task-Scheduling-System")
from utils.read_data import read_line_elems_from_file
from utils.plt_config import PltConfig
from utils.create_pic import save_to_histogram_from_list, save_to_pic_from_list


def show_task_type_distribution(file_path):
    data = pd.read_csv(file_path, delimiter='\t')
    data.columns = ['id', 'commit_time', 'length', 'cpu_uti', 'size']
    size_list = data['size'].tolist()
    task_num = len(data)
    cpu_tasks_num = 0
    io_tasks_num = 0
    for i in range(task_num):
        if size_list[i] != 10:
            io_tasks_num += 1
        else:
            cpu_tasks_num += 1
    print("cpu_tasks_num: ", cpu_tasks_num)
    print("io_tasks_num:", io_tasks_num)

    # 2. plot data
    x_axis_data = ['CPU-intensive tasks', 'IO-intensive tasks']
    y_axis_data = [cpu_tasks_num, io_tasks_num]

    plt_config = PltConfig()
    # plt_config.title = "任务负载类型分布"
    plt_config.xlabel = "任务负载类型"
    plt_config.ylabel = "任务数目"
    plt_config.x_axis_data = x_axis_data
    dest_dir = f"task_type_distribution_on_Alibaba_dataset.png"
    save_to_histogram_from_list(y_axis_data, dest_dir, plt_config, show=True)

if __name__ == '__main__':
    # filepath = "../dataset/Alibaba/Alibaba-Cluster-trace-500000-multiple-test.txt"
    filepath = "../dataset/Alibaba/Alibaba-Cluster-trace-300000-IO-test.txt"
    show_task_type_distribution(filepath)
