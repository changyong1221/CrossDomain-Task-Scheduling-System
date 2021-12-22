import pandas as pd
import globals.global_var as glo


def compute_avg_task_process_time():
    """Compute average task process time of different scheduling algorightm
    """
    schedulers = ["DQNScheduler", "RoundRobinScheduler"]
    RR_data_path = '../' + glo.results_path_list[schedulers[1]]
    RR_data = pd.read_csv(RR_data_path, header=None, delimiter='\t')
    RR_data.columns = ['task_id', 'machine_id', 'transfer_time', 'wait_time', 'execute_time', 'process_time']
    RR_process_time_mean = RR_data['process_time'].mean()
    print(f"RR_process_time_mean: {RR_process_time_mean}")

    DQN_data_path = '../' + glo.results_path_list[schedulers[0]]
    DQN_data = pd.read_csv(DQN_data_path, header=None, delimiter='\t')
    DQN_data.columns = ['task_id', 'machine_id', 'transfer_time', 'wait_time', 'execute_time', 'process_time']
    DQN_process_time_mean = DQN_data['process_time'].mean()
    print(f"DQN_process_time_mean: {DQN_process_time_mean}")


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
    compute_avg_task_process_time()
