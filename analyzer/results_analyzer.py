import math
import numpy as np
import pandas as pd
import globals.global_var as glo
from utils.create_pic import save_to_pic_from_list, save_to_histogram_from_list, save_to_pie_from_list
from utils.plt_config import PltConfig


def analyze_task_results():
    # 1. settings
    path = "../results/task_run_results/client-10000/DQNScheduler_task_run_results_test.txt"
    DQN_data = pd.read_csv(path, header=None, delimiter='\t')
    DQN_data.columns = ['task_id', 'machine_id', 'transfer_time', 'wait_time', 'execute_time', 'process_time']
    test_records_num = 2000
    machine_num = 20
    total_records_num = len(DQN_data)
    epoch_num = total_records_num // test_records_num

    # 2. task processing time
    avg_processing_time_list = []
    for epoch in range(epoch_num):
        epoch_data = DQN_data[test_records_num*epoch: test_records_num*(epoch+1)]
        avg_processing_time = epoch_data['process_time'].mean()
        avg_processing_time_list.append(avg_processing_time)
    print(avg_processing_time_list)
    dest_path = f"../pic/task_run_results/avg_processing_time_{epoch_num}epoch_{test_records_num}records.png"
    plt_config = PltConfig()
    plt_config.title = f"average task processing time in {epoch_num} federated rounds"
    plt_config.xlabel = "federated rounds"
    plt_config.ylabel = "average task processing time"
    save_to_pic_from_list(avg_processing_time_list, dest_path, plt_config, show=True)

    # 3. task to machine map
    for epoch in range(epoch_num):
        epoch_data = DQN_data[test_records_num*epoch: test_records_num*(epoch+1)]
        machine_assignment_list = []
        for machine_id in range(machine_num):
            machine_assignment_list.append(len(epoch_data[epoch_data['machine_id'] == machine_id]))
        dest_path = f"../pic/machine_assignment/machine_assignment_round_{epoch}.png"
        plt_config = PltConfig()
        plt_config.title = f"task to machine assignment map in round-{epoch}"
        plt_config.xlabel = "machine id"
        plt_config.ylabel = "task number"
        plt_config.x_axis_data = [i for i in range(machine_num)]
        save_to_histogram_from_list(machine_assignment_list, dest_path, plt_config, show=False)


def analyze_machine_results():
    # 1. settings
    path = "../results/machine_status_results/client-10000/DQNScheduler/"
    machine_num = 20
    federated_rounds = 10

    # 1. machine work time
    for epoch in range(federated_rounds):
        for machine_id in range(machine_num):
            data_path = f"{path}/{epoch}/{machine_id}_status_test.txt"
            machine_data = pd.read_csv(data_path, header=None, delimiter='\t')
            machine_data.columns = ['work_time', 'cpu_uti', 'mem_uti', 'band_uti']


def compute_avg_task_process_time():
    """Compute average task process time of different scheduling algorightm
    """
    idx = 1
    schedulers = ["RR", "DQN"]
    data_path = [
        "../results/task_run_results/RoundRobinScheduler/RoundRobinScheduler_task_run_results.txt",
        "../results/task_run_results/DQNScheduler/DQNScheduler_task_run_results.txt",
    ]
    data_path = data_path[idx]
    data = pd.read_csv(data_path, header=None, delimiter='\t')
    data.columns = ['task_id', 'task_mi', 'machine_id', 'machine_mips', 'transfer_time', 'wait_time',
                       'execute_time', 'process_time']
    transfer_time_mean = data['transfer_time'].mean()
    wait_time_mean = data['wait_time'].mean()
    execute_time_mean = data['execute_time'].mean()
    process_time_mean = data['process_time'].mean()
    print(f"{schedulers[idx]}_transfer_time_mean: {transfer_time_mean}")
    print(f"{schedulers[idx]}_wait_time_mean: {wait_time_mean}")
    print(f"{schedulers[idx]}_execute_time_mean: {execute_time_mean}")
    print(f"{schedulers[idx]}_process_time_mean: {process_time_mean}")
    pie_list = [round(transfer_time_mean, 2), round(wait_time_mean, 2), round(execute_time_mean, 2)]
    output_path = f"../pic/task_run_results/{schedulers[idx]}_time_distribution_20machine_2000tasks.png"
    plt_config = PltConfig()
    plt_config.title = f"Task processing time distribution using {schedulers[idx]}"
    plt_config.labels = ["transfer_time", "wait_time", "execute_time"]
    save_to_pie_from_list(pie_list, output_path, plt_config, show=True)


def compute_task_to_machine_map():
    # 1. settings
    idx = 1
    schedulers = ["RR", "DQN"]
    data_path = [
        "../results/task_run_results/RoundRobinScheduler/RoundRobinScheduler_task_run_results.txt",
        "../results/task_run_results/DQNScheduler/DQNScheduler_task_run_results.txt",
        # "../results/task_run_results/client-10000/DQNScheduler_task_run_results_test.txt"
    ]
    path = data_path[idx]
    data = pd.read_csv(path, header=None, delimiter='\t')
    data.columns = ['task_id', 'task_mi', 'machine_id', 'machine_mips', 'transfer_time', 'wait_time', 'execute_time',
                    'process_time']
    test_records_num = 2000
    machine_num = 20
    total_records_num = len(data)
    epoch_num = total_records_num // test_records_num

    machine_assignment_list = []
    for machine_id in range(machine_num):
        machine_assignment_list.append(len(data[data['machine_id'] == machine_id]))
    dest_path = f"../pic/machine_assignment/{schedulers[idx]}_machine_assignment_20machine_2000tasks.png"
    plt_config = PltConfig()
    plt_config.title = f"task to machine assignment map using {schedulers[idx]}"
    plt_config.xlabel = "machine id"
    plt_config.ylabel = "task number"
    plt_config.x_axis_data = [i for i in range(machine_num)]
    save_to_histogram_from_list(machine_assignment_list, dest_path, plt_config, show=True)


def compute_machine_status_results():
    RR_data_path = '../results/machine_status_results/RoundRobinScheduler/'
    RR_data = pd.read_csv(RR_data_path, header=None, delimiter='\t')
    RR_data.columns = ['task_id', 'task_mi', 'machine_id', 'machine_mips', 'transfer_time', 'wait_time',
                       'execute_time', 'process_time']
    RR_transfer_time_mean = RR_data['transfer_time'].mean()
    RR_wait_time_mean = RR_data['wait_time'].mean()
    RR_execute_time_mean = RR_data['execute_time'].mean()
    RR_process_time_mean = RR_data['process_time'].mean()
    print(f"RR_transfer_time_mean: {RR_transfer_time_mean}")
    print(f"RR_wait_time_mean: {RR_wait_time_mean}")
    print(f"RR_execute_time_mean: {RR_execute_time_mean}")
    print(f"RR_process_time_mean: {RR_process_time_mean}")
    pie_list = [round(RR_transfer_time_mean, 2), round(RR_wait_time_mean, 2), round(RR_execute_time_mean, 2)]
    output_path = "../pic/task_run_results/RR_time_distribution_20machine_2000tasks.png"
    plt_config = PltConfig()
    plt_config.title = "Task processing time distribution using RR"
    plt_config.labels = ["transfer_time", "wait_time", "execute_time"]
    save_to_pie_from_list(pie_list, output_path, plt_config, show=True)


def compute_avg_task_process_time_by_name(scheduler_name):
    """Compute average task process time of different scheduling algorightm
    """
    data_path = glo.results_path_list[scheduler_name]
    data = pd.read_csv(data_path, header=None, delimiter='\t')
    data.columns = ['task_id', 'task_mi', 'machine_id', 'machine_mips', 'transfer_time', 'wait_time', 'execute_time',
                    'process_time']
    process_time_mean = data['process_time'].mean()
    print(f"{scheduler_name}'s average task processing time: {process_time_mean} s")


def show_task_run_results_distribution():
    pass


def show_task_process_time_results():
    """Display task processing time
    """
    pass


def show_machine_utilization_std():
    """Display machine utilization std
    """
    pass


if __name__ == '__main__':
    # analyze_task_results()
    # analyze_machine_results()
    # compute_avg_task_process_time()
    compute_task_to_machine_map()
