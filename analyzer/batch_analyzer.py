import numpy as np
import pandas as pd
import sys
sys.path.append("/home/scy/CrossDomain-Task-Scheduling-System")
import globals.global_var as glo
from utils.file_check import check_and_build_dir
from utils.write_file import write_list_to_file
from utils.create_pic import save_compare_pic_from_vector, save_to_histogram_from_list, save_multiple_histogram_compare_pic_from_vector
from utils.plt_config import PltConfig


# 展示任务类型的分布，使用并列柱状图，仅展示单个机器的对比
def plot_task_type_distribution_comp():
    schedulers = ["DQNScheduler", "RoundRobinScheduler", "EarliestScheduler",
                  "WeightedRandomScheduler", "HeuristicScheduler"]
    labels = ["DRL-TA", "RR", "Earliest", "WeightedRandom", "GA"]
    
    data_vector = [[] for i in range(3)]
    for i, scheduler in enumerate(schedulers):
        data_path = f"../results/task_run_results/{glo.current_dataset}{glo.records_num}/{scheduler}/{scheduler}_task_run_results2.txt"
        data = pd.read_csv(data_path, header=None, delimiter='\t')
        data.columns = ['task_id', 'task_mi', 'machine_id', 'machine_mips', 'transfer_time', 'wait_time',
                        'execute_time', 'process_time', 'task_type']
        machine_id = 19
        data = data[data['machine_id'] == machine_id]
        num_cpu_intensive = len(data[data['task_type'] == 'cpu-intensive'])
        num_io_intensive = len(data[data['task_type'] == 'io-intensive'])
        num_simple = len(data[data['task_type'] == 'simple'])
        data_vector[0].append(num_cpu_intensive)
        data_vector[1].append(num_io_intensive)
        data_vector[2].append(num_simple)
    # 1. 任务类型的分布
    dest_path = f"task_type_distribution_comparison_on_{glo.current_dataset}{glo.records_num}_machine{machine_id}.png"
    plt_config = PltConfig()
    plt_config.title = f"comparison of task type distribution on {glo.current_dataset}{glo.records_num} machine{machine_id}"
    plt_config.xlabel = "different algorithms"
    plt_config.ylabel = "number of tasks"
    plt_config.x_axis_data = labels
    save_multiple_histogram_compare_pic_from_vector(data_vector, dest_path, plt_config, show=False)
    
    
def plot_task_batch_results():
    schedulers = ["DQNScheduler", "RoundRobinScheduler", "EarliestScheduler",
                  "WeightedRandomScheduler", "HeuristicScheduler"]
    labels = ["DRL-TA", "RR", "Earliest", "WeightedRandom", "GA"]

    avg_process_time_vector = []
    makespan_vector = []
    avg_work_time_vector = []
    # Alibaba1000000 [370, 420],
    # GoCJ [300, 350],
    slice_start = 300
    slice_end = 350
    check_and_build_dir(f"./slice{slice_start}-slice{slice_end}")
    for i, scheduler in enumerate(schedulers):
        data_path = f"../results/task_run_results/{glo.current_dataset}{glo.records_num}/{scheduler}/task_batches/" \
                    f"task_batches_run_results2.txt"
        data = pd.read_csv(data_path, header=None, delimiter='\t')
        data.columns = ['avg_process_time', 'makespan', 'avg_work_time']
        avg_processing_time_data = data['avg_process_time'].tolist()[slice_start:slice_end]
        makespan_data = data['makespan'].tolist()[slice_start:slice_end]
        avg_work_time_data = data['avg_work_time'].tolist()[slice_start:slice_end]
        avg_process_time_vector.append(avg_processing_time_data)
        makespan_vector.append(makespan_data)
        avg_work_time_vector.append(avg_work_time_data)

    # 1. task_batch平均任务处理时间
    dest_path = f"slice{slice_start}-slice{slice_end}/batch_average_task_processing_time_comparison_on_{glo.current_dataset}{glo.records_num}_slice{slice_start}-slice{slice_end}_4.png"
    plt_config = PltConfig()
    plt_config.xlabel = "Task batch数量"
    plt_config.ylabel = "平均任务处理时间"
    plt_config.x_axis_data = [i for i in range(len(makespan_data))]
    save_compare_pic_from_vector(avg_process_time_vector, labels, dest_path, plt_config, show=False)

    # 2. task_batch makespan
    dest_path = f"slice{slice_start}-slice{slice_end}/batch_makespan_comparison_on_{glo.current_dataset}{glo.records_num}_slice{slice_start}-slice{slice_end}_4.png"
    plt_config = PltConfig()
    plt_config.xlabel = "Task batch数量"
    plt_config.ylabel = "Makespan of batch (秒)"
    plt_config.x_axis_data = [i for i in range(len(makespan_data))]
    save_compare_pic_from_vector(makespan_vector, labels, dest_path, plt_config, show=False)

    # 3. task_batch平均机器工作时间
    dest_path = f"slice{slice_start}-slice{slice_end}/batch_average_machine_worktime_comparison_on_{glo.current_dataset}{glo.records_num}_slice{slice_start}-slice{slice_end}_4.png"
    plt_config = PltConfig()
    plt_config.xlabel = "Task batch数量"
    plt_config.ylabel = "平均机器实际工作时间 (秒)"
    plt_config.x_axis_data = [i for i in range(len(makespan_data))]
    save_compare_pic_from_vector(avg_work_time_vector, labels, dest_path, plt_config, show=False)


def plot_average_task_processing_time_comparison():
    schedulers = ["DQNScheduler", "RoundRobinScheduler", "HeuristicScheduler", "EarliestScheduler", "WeightedRandomScheduler"]
    labels = ["DRL-TA", "RR", "GA", "Earliest", "WeightedRandom"]
    
    # 获取总任务数
    # 计算每个调度算法的单位时间吞吐量
    # 单位时间吞吐量 = 总任务数 / 总任务makespan
    total_avg_task_processing_time_list = []
    for i, scheduler in enumerate(schedulers):
        data_path = f"../results/task_run_results/{glo.current_dataset}{glo.records_num}/{scheduler}/" \
                    f"{scheduler}_task_run_results.txt"
        data = pd.read_csv(data_path, header=None, delimiter='\t')
        data.columns = ['task_id', 'task_mi', 'task_size', 'machine_id', 'machine_mips', 'transfer_time', 'wait_time',
                        'execute_time', 'process_time', 'task_type']
        task_num = len(data)
        print("task_num: ", task_num)
        transfer_time_mean = data['transfer_time'].mean()
        wait_time_mean = data['wait_time'].mean()
        execute_time_mean = data['execute_time'].mean()
        process_time_mean = data['process_time'].mean()
        print(f"{scheduler}_transfer_time_mean: {transfer_time_mean}")
        print(f"{scheduler}_wait_time_mean: {wait_time_mean}")
        print(f"{scheduler}_execute_time_mean: {execute_time_mean}")
        print(f"{scheduler}_process_time_mean: {process_time_mean}")
        total_avg_task_processing_time_list.append(process_time_mean)
    for i, elem in enumerate(total_avg_task_processing_time_list):
        total_avg_task_processing_time_list[i] /= 1000
    dest_path = f"total_average_task_processing_time_comparison_on_{glo.current_dataset}{glo.records_num}.png"
    plt_config = PltConfig()
    plt_config.title = f"total average task processing time comparison on {glo.current_dataset}{glo.records_num}"
    plt_config.xlabel = "调度算法"
    plt_config.ylabel = "平均任务处理时间 (1e3)"
    plt_config.x_axis_data = labels
    save_to_histogram_from_list(total_avg_task_processing_time_list, dest_path, plt_config, show=False, show_text=True)


def plot_machine_work_time_std_comparison():
    schedulers = ["DQNScheduler", "RoundRobinScheduler", "EarliestScheduler",
                  "WeightedRandomScheduler", "HeuristicScheduler"]
    labels = ["DRL-TA", "RR", "Earliest", "WR", "GA"]

    # 获取总任务数
    # 计算每个调度算法的单位时间吞吐量
    # 单位时间吞吐量 = 总任务数 / 总任务makespan
    machine_num = 20
    machine_work_time_std_list = []
    for i, scheduler in enumerate(schedulers):
        machine_work_time_list = []
        path = f"../results/machine_status_results/{glo.current_dataset}{glo.records_num}/{scheduler}"
        for machine_id in range(machine_num):
            work_time_path = f"{path}/{machine_id}_batch_status.txt"
            work_time_data = pd.read_csv(work_time_path, header=None, delimiter='\t')
            work_time_data.columns = ['makespan', 'work_time']
            machine_work_time = work_time_data['work_time'].tolist()[-1]
            machine_work_time_list.append(machine_work_time)
        machine_std = np.std(np.array(machine_work_time_list))
        machine_work_time_std_list.append(machine_std)
        print(f"{scheduler} machine_work_time_std: {machine_std}")
    
    for i in range(len(labels)):
        machine_work_time_std_list[i] /= 1000
    dest_path = f"machine_work_time_std_comparison_on_{glo.current_dataset}{glo.records_num}.png"
    plt_config = PltConfig()
    plt_config.xlabel = "调度算法"
    plt_config.ylabel = "机器实际工作时间标准差 (1e3)"

    plt_config.x_axis_data = labels
    save_to_histogram_from_list(machine_work_time_std_list, dest_path, plt_config, show=False, show_text=True)


if __name__ == "__main__":
    glo.current_dataset = "Alibaba"
    # glo.records_num = 500000
    glo.records_num = 300000
    plot_machine_work_time_std_comparison()
    plot_average_task_processing_time_comparison()
