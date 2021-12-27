import time

from core.domain import create_domains, create_multi_domain
from scheduler.RoundRobinScheduler import RoundRobinScheduler
from scheduler.DQNScheduler import DQNScheduler
from analyzer.results_analyzer import compute_avg_task_process_time_by_name
from utils.load_data import load_machines_from_file, load_task_batches_from_file, sample_tasks_from_file, \
    load_tasks_from_file
from utils.file_check import check_and_build_dir
import globals.global_var as glo


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
    task_file_path = f"../dataset/GoCJ/client/GoCJ_Dataset_2000_client_{client_id}.txt"
    task_batch_list = sample_tasks_from_file(task_file_path, delimiter='\t')

    # 7. set scheduler for multi-domain system
    machine_num = len(machine_list)
    task_batch_num = len(task_batch_list)
    # scheduler = RoundRobinScheduler(machine_num)
    scheduler = DQNScheduler(multi_domain.multidomain_id, machine_num, task_batch_num, is_federated=True)
    scheduler_name = scheduler.__class__.__name__
    glo.task_run_results_path = glo.results_path_list[scheduler_name]
    glo.current_scheduler = scheduler_name
    multi_domain.set_scheduler(scheduler)

    # 8. commit tasks to multi-domain system, training
    glo.is_test = False
    multi_domain.commit_tasks(task_batch_list)

    # 9. reset multi-domain system
    multi_domain.reset()

    # 10. federated test
    glo.is_test = True
    task_file_path = f"../dataset/GoCJ/GoCJ_Dataset_2000_test.txt"
    tasks_for_test = load_tasks_from_file(task_file_path)
    multi_domain.commit_tasks(tasks_for_test)
    multi_domain.reset()

    # 10. show statistics
    # compute_avg_task_process_time_by_name(scheduler_name)


def test_federated():
    # Initialization
    start_time = time.time()
    n_clients = 10
    federated_rounds = 1
    init_federated_model()

    # federated main



def fed_avg(clients_num):
    model_path_list = []
    for i in range(clients_num):
        model_path_list.append(f"../save/client-{i}")


def init_federated_model():
    machine_num = 20
    scheduler = DQNScheduler(multidomain_id=1, machine_num=machine_num, task_batch_num=1, is_federated=False)
    global_model_dir = "../save/global"
    check_and_build_dir(global_model_dir)
    global_model_path = f"../save/global/global.pth"
    scheduler.DRL.save_initial_model(global_model_path)


if __name__ == "__main__":
    test_federated()
