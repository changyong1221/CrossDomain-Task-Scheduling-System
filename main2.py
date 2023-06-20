import time
from core.domain import create_domains, create_multi_domain
from scheduler.RandomScheduler import RandomScheduler
from scheduler.WeightedRandomScheduler import WeightedRandomScheduler
from scheduler.EarliestScheduler import EarliestScheduler
from scheduler.RoundRobinScheduler import RoundRobinScheduler
from scheduler.HeuristicScheduler import HeuristicScheduler
from scheduler.DQNScheduler import DQNScheduler
from scheduler.DDPGScheduler import DDPGScheduler
from analyzer.results_analyzer import compute_avg_task_process_time_by_name
from utils.load_data import load_machines_from_file, load_task_batches_from_file
from utils.state_representation import get_machine_kind_list
import globals.global_var as glo


def inter_domain_scheduling(idx):
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
    glo.records_num = 300000
    glo.current_dataset = "Alibaba"
    task_file_path = f"dataset/Alibaba/Alibaba-Cluster-trace-{glo.records_num}-IO-test.txt"
    task_batch_list = load_task_batches_from_file(task_file_path, delimiter='\t')

    # 7. set scheduler for multi-domain system
    machine_num = len(machine_list)
    task_batch_num = len(task_batch_list)
    print("machine_num: ", machine_num)
    print("task_batch_num: ", task_batch_num)

    """ Scheduler Selection """
    # D3QN
    # epsilon_dec = 1.0
    epsilon_dec = 0.9993
    # epsilon_dec = 0.9996
    # epsilon_dec = 0.9996
    # epsilon_dec = 0.996
    prob = 0.5
    balance_prob = 0.9
    is_test = True

    glo.is_federated = False
    glo.is_test = True
    
    machine_kind_num_list, machine_kind_idx_range_list = get_machine_kind_list(machine_list)
    
    if idx == 0:
        scheduler = DQNScheduler(multi_domain.multidomain_id, machine_list, machine_num, task_batch_num, machine_kind_num_list,
                                 machine_kind_idx_range_list, is_federated=glo.is_federated, is_test=is_test, epsilon_decay=epsilon_dec, prob=prob, balance_prob=balance_prob)
    elif idx == 1:
        scheduler = RoundRobinScheduler(machine_num)
    elif idx == 2:
        scheduler = WeightedRandomScheduler(machine_list)
    elif idx == 3:
        scheduler = EarliestScheduler()
    else:
        scheduler = HeuristicScheduler(machine_list)

    scheduler_name = scheduler.__class__.__name__
    glo.current_scheduler = scheduler_name
    multi_domain.set_scheduler(scheduler)

    # 8. commit tasks to multi-domain system
    for batch in task_batch_list:
        multi_domain.commit_tasks(batch)

    # 9. reset multi-domain system
    multi_domain.reset()

    # 10. show statistics
    # compute_avg_task_process_time_by_name(scheduler_name)


if __name__ == "__main__":
    for i in range(0, 5):
        start_time = time.time()
        glo.is_print_log = False
        inter_domain_scheduling(idx=i)

        end_time = time.time()
        print("Time used: %.2f s" % (end_time - start_time))