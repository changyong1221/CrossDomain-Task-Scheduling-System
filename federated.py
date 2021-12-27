import time
import torch
import math
from core.domain import create_domains, create_multi_domain
from scheduler.RoundRobinScheduler import RoundRobinScheduler
from scheduler.DQNScheduler import DQNScheduler
from analyzer.results_analyzer import compute_avg_task_process_time_by_name
from utils.load_data import load_machines_from_file, load_task_batches_from_file, sample_tasks_from_file, \
    load_tasks_from_file
from utils.file_check import check_and_build_dir
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
        vm_tasks_capacity[i] = math.ceil(((float)(machine.mips) / total_mips) * glo.test_records_num)
    # print("vm_tasks_capacity: ", vm_tasks_capacity)
    return vm_tasks_capacity


def client_train(client_id):
    """Perform inter-domain task scheduling
    """
    # 1. create multi-domain system
    multi_domain_system_location = "北京市"
    multi_domain = create_multi_domain(client_id, multi_domain_system_location)

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
    task_file_path = f"dataset/GoCJ/client/GoCJ_Dataset_2000_client_{client_id}.txt"
    task_batch_list = sample_tasks_from_file(task_file_path, batch_size=128, delimiter='\t')

    # 7. set scheduler for multi-domain system
    machine_num = len(machine_list)
    task_batch_num = len(task_batch_list)
    # scheduler = RoundRobinScheduler(machine_num)
    vm_task_capacity = get_vm_tasks_capacity(machine_list)
    scheduler = DQNScheduler(multi_domain.multidomain_id, machine_num, task_batch_num, vm_task_capacity,
                             is_federated=True)
    scheduler_name = scheduler.__class__.__name__
    glo.current_scheduler = scheduler_name
    multi_domain.set_scheduler(scheduler)

    # 8. commit tasks to multi-domain system, training
    glo.is_test = False
    for task in task_batch_list:
        multi_domain.commit_tasks([task])
    # multi_domain.commit_tasks(task_batch_list)

    # 9. reset multi-domain system
    multi_domain.reset()

    # 10. show statistics
    # compute_avg_task_process_time_by_name(scheduler_name)


def federated_test():
    """Perform inter-domain task scheduling
    """
    # 1. create multi-domain system
    multi_domain_system_location = "北京市"
    client_id = 10000   # federated server
    multi_domain = create_multi_domain(client_id, multi_domain_system_location)

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
    task_file_path = f"dataset/GoCJ/GoCJ_Dataset_2000_test.txt"
    tasks_for_test = load_tasks_from_file(task_file_path, delimiter='\t')

    # 7. set scheduler for multi-domain system
    machine_num = len(machine_list)
    task_batch_num = len(tasks_for_test)
    # scheduler = RoundRobinScheduler(machine_num)
    vm_task_capacity = get_vm_tasks_capacity(machine_list)
    scheduler = DQNScheduler(multi_domain.multidomain_id, machine_num, task_batch_num, vm_task_capacity,
                             is_federated=True)
    scheduler_name = scheduler.__class__.__name__
    glo.current_scheduler = scheduler_name
    multi_domain.set_scheduler(scheduler)

    # 8. commit tasks to multi-domain system, training
    glo.is_test = True
    for task in tasks_for_test:
        multi_domain.commit_tasks([task])
    # multi_domain.commit_tasks(tasks_for_test)

    # 9. reset multi-domain system
    multi_domain.reset()

    # 10. show statistics
    # compute_avg_task_process_time_by_name(scheduler_name)


def test_federated():
    # Initialization
    start_time = time.time()
    n_clients = 10
    federated_rounds = 1
    init_federated_model()
    glo.is_federated = True
    glo.is_print_log = True

    # federated main
    print("federated learning start...")
    for epoch in range(federated_rounds):
        glo.federated_round = epoch
        print(f"Round {epoch}: ")
        for client_id in range(n_clients):
            client_train(client_id)
        fed_avg(n_clients)
        federated_test()

    print("federated learning finished.")
    end_time = time.time()
    print("Time used: %.2f s" % (end_time - start_time))


def fed_avg(clients_num):
    model_path_list = []
    for i in range(clients_num):
        model_path_list.append(f"save/client-{i}/eval.pth")

    # load client weights
    clients_weights_sum = None
    for model_path in model_path_list:
        cur_parameters = torch.load(model_path)
        if clients_weights_sum is None:
            clients_weights_sum = {}
            for key, var in cur_parameters.items():
                clients_weights_sum[key] = var.clone()
        else:
            for var in cur_parameters:
                clients_weights_sum[var] = clients_weights_sum[var] + cur_parameters[var]

    # fed_avg
    global_weights = {}
    for var in clients_weights_sum:
        global_weights[var] = (clients_weights_sum[var] / clients_num)
    global_model_path = f"save/global/global.pth"
    torch.save(global_weights, global_model_path)


def init_federated_model():
    machine_num = 20
    vm_task_capacity = []
    scheduler = DQNScheduler(multidomain_id=1, machine_num=machine_num, task_batch_num=1,
                             vm_task_capacity=vm_task_capacity,
                             is_federated=False)
    global_model_dir = "save/global"
    check_and_build_dir(global_model_dir)
    global_model_path = f"save/global/global.pth"
    scheduler.DRL.save_initial_model(global_model_path)


if __name__ == "__main__":
    test_federated()
