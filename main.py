import math
from core.domain import create_domains, create_multi_domain
from scheduler.RoundRobinScheduler import RoundRobinScheduler
from scheduler.DQNScheduler import DQNScheduler
from analyzer.results_analyzer import compute_avg_task_process_time_by_name
from utils.load_data import load_machines_from_file, load_task_batches_from_file
import globals.global_var as glo


# 根据机器性能分配任务数
def get_vm_tasks_capacity(machine_list):
    # 定义数组用于存储最终结果
    vm_tasks_capacity = []
    # 所有机器的总mips
    total_mips = 0
    for machine in machine_list:
        vm_tasks_capacity.append(0)
        total_mips += machine.mips
    # print("total_mips: ", total_mips)
    # 计算每个机器mips占总mips的比例
    for i, machine in enumerate(machine_list):
        vm_tasks_capacity[i] = math.ceil(((float)(machine.mips) / total_mips) * glo.records_num)
    # print("vm_tasks_capacity: ", vm_tasks_capacity)
    return vm_tasks_capacity


def inter_domain_scheduling():
    """Perform inter-domain task scheduling
    """
    # 1. create multi-domain system
    multi_domain_system_location = "北京市"
    multi_domain = create_multi_domain(0, multi_domain_system_location)

    # 2. create domains
    domain_num = 5
    location_list = ["北京市", "上海市", "莫斯科", "新加坡市", "吉隆坡"]
    domain_list = create_domains(location_list)

    # 3. add machines to domain
    machine_file_path = glo.machine_file_path
    machine_list = load_machines_from_file(machine_file_path)
    machine_num_per = len(machine_list) // domain_num
    for domain_id in range(domain_num):
        for i in range(machine_num_per):
            machine = machine_list[i + domain_id*machine_num_per]
            machine.set_location(domain_list[domain_id].longitude, domain_list[domain_id].latitude)
            domain_list[domain_id].add_machine(machine)

    # 4. clustering machines in each domain
    cluster_num = 3
    for domain in domain_list:
        domain.clustering_machines(cluster_num)

    # 5. add domain to multi-domain system
    for domain in domain_list:
        multi_domain.add_domain(domain)

    # 6. load tasks
    # task_file_path = glo.task_file_path
    task_file_path = f"dataset/GoCJ/GoCJ_Dataset_20000_train.txt"
    task_batch_list = load_task_batches_from_file(task_file_path, delimiter='\t')

    # 7. set scheduler for multi-domain system
    machine_num = len(machine_list)
    task_batch_num = len(task_batch_list)
    print("machine_num: ", machine_num)
    print("task_batch_num: ", task_batch_num)
    # scheduler = RoundRobinScheduler(machine_num)
    vm_task_capacity = get_vm_tasks_capacity(machine_list)
    scheduler = DQNScheduler(multi_domain.multidomain_id, machine_num, task_batch_num, vm_task_capacity,
                             is_federated=False)
    scheduler_name = scheduler.__class__.__name__
    glo.task_run_results_path = glo.results_path_list[scheduler_name]
    glo.current_scheduler = scheduler_name
    multi_domain.set_scheduler(scheduler)

    # 8. commit tasks to multi-domain system
    for batch in task_batch_list:
        multi_domain.commit_tasks(batch)

    # 9. reset multi-domain system
    multi_domain.reset()

    # 10. show statistics
    compute_avg_task_process_time_by_name(scheduler_name)


def model_test():
    """Perform inter-domain task scheduling
    """
    # 1. create multi-domain system
    multi_domain_system_location = "北京市"
    multi_domain = create_multi_domain(0, multi_domain_system_location)

    # 2. create domains
    domain_num = 5
    location_list = ["北京市", "上海市", "莫斯科", "新加坡市", "吉隆坡"]
    domain_list = create_domains(location_list)

    # 3. add machines to domain
    machine_file_path = glo.machine_file_path
    machine_list = load_machines_from_file(machine_file_path)
    machine_num_per = len(machine_list) // domain_num
    for domain_id in range(domain_num):
        for i in range(machine_num_per):
            machine = machine_list[i + domain_id*machine_num_per]
            machine.set_location(domain_list[domain_id].longitude, domain_list[domain_id].latitude)
            domain_list[domain_id].add_machine(machine)

    # 4. clustering machines in each domain
    cluster_num = 3
    for domain in domain_list:
        domain.clustering_machines(cluster_num)

    # 5. add domain to multi-domain system
    for domain in domain_list:
        multi_domain.add_domain(domain)

    # 6. load tasks
    # task_file_path = glo.task_file_path
    task_file_path = f"dataset/GoCJ/GoCJ_Dataset_2000_test.txt"
    glo.records_num = 2000
    task_batch_list = load_task_batches_from_file(task_file_path, delimiter='\t')

    # 7. set scheduler for multi-domain system
    machine_num = len(machine_list)
    task_batch_num = len(task_batch_list)
    print("machine_num: ", machine_num)
    print("task_batch_num: ", task_batch_num)
    # scheduler = RoundRobinScheduler(machine_num)
    vm_task_capacity = get_vm_tasks_capacity(machine_list)
    scheduler = DQNScheduler(multi_domain.multidomain_id, machine_num, task_batch_num, vm_task_capacity,
                             is_federated=True)
    scheduler_name = scheduler.__class__.__name__
    glo.task_run_results_path = glo.results_path_list[scheduler_name]
    glo.current_scheduler = scheduler_name
    multi_domain.set_scheduler(scheduler)

    # 8. commit tasks to multi-domain system
    glo.is_federated = True
    glo.is_test = True
    for batch in task_batch_list:
        multi_domain.commit_tasks(batch)

    # 9. reset multi-domain system
    multi_domain.reset()

    # 10. show statistics
    compute_avg_task_process_time_by_name(scheduler_name)


if __name__ == "__main__":
    glo.is_print_log = False
    # inter_domain_scheduling()
    model_test()
