import math
import pandas as pd
import globals.global_var as glo
from utils.create_pic import save_to_pic_from_list, save_to_histogram_from_list
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
    schedulers = ["DQNScheduler", "RoundRobinScheduler"]
    # RR_data_path = '../' + glo.results_path_list[schedulers[1]]
    # RR_data = pd.read_csv(RR_data_path, header=None, delimiter='\t')
    # RR_data.columns = ['task_id', 'machine_id', 'transfer_time', 'wait_time', 'execute_time', 'process_time']
    # RR_process_time_mean = RR_data['process_time'].mean()
    # print(f"RR_process_time_mean: {RR_process_time_mean}")

    DQN_data_path = '../results/task_run_results/client-0/DQNScheduler_task_run_results.txt'
    # DQN_data_path = '../results/task_run_results/client-10000/DQNScheduler_task_run_results_test.txt'
    DQN_data = pd.read_csv(DQN_data_path, header=None, delimiter='\t')
    DQN_data.columns = ['task_id', 'machine_id', 'transfer_time', 'wait_time', 'execute_time', 'process_time']
    DQN_process_time_mean = DQN_data['process_time'].mean()
    print(f"DQN_process_time_mean: {DQN_process_time_mean}")


def compute_task_to_machine_map():
    # 1. settings
    path = "../results/task_run_results/client-0/DQNScheduler_task_run_results.txt"
    # path = "../results/task_run_results/client-10000/DQNScheduler_task_run_results_test.txt"
    # path = "../results/task_run_results/RoundRobinScheduler/RoundRobinScheduler_task_run_results.txt"
    data = pd.read_csv(path, header=None, delimiter='\t')
    data.columns = ['task_id', 'machine_id', 'transfer_time', 'wait_time', 'execute_time', 'process_time']
    test_records_num = 2000
    machine_num = 20
    total_records_num = len(data)
    epoch_num = total_records_num // test_records_num

    machine_assignment_list = []
    for machine_id in range(machine_num):
        machine_assignment_list.append(len(data[data['machine_id'] == machine_id]))
    # dest_path = f"../pic/machine_assignment/RR_machine_assignment.png"
    # dest_path = f"../pic/machine_assignment/DQN_machine_assignment.png"
    dest_path = f"../pic/machine_assignment/DQN_machine_assignment_client_1.png"
    plt_config = PltConfig()
    plt_config.title = f"task to machine assignment map using DQN"
    plt_config.xlabel = "machine id"
    plt_config.ylabel = "task number"
    plt_config.x_axis_data = [i for i in range(machine_num)]
    save_to_histogram_from_list(machine_assignment_list, dest_path, plt_config, show=True)


def compute_avg_task_process_time_by_name(scheduler_name):
    """Compute average task process time of different scheduling algorightm
    """
    data_path = glo.results_path_list[scheduler_name]
    data = pd.read_csv(data_path, header=None, delimiter='\t')
    data.columns = ['task_id', 'machine_id', 'transfer_time', 'wait_time', 'execute_time', 'process_time']
    process_time_mean = data['process_time'].mean()
    print(f"{scheduler_name}'s average task processing time: {process_time_mean} s")


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
    compute_avg_task_process_time()
    compute_task_to_machine_map()
