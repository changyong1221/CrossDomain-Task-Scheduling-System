from core.domain import create_one_domain
from core.cluster import create_one_cluster
from scheduler.RoundRobinScheduler import RoundRobinScheduler
from scheduler.DQNScheduler import DQNScheduler
from utils.load_data import load_machines_from_file, load_task_batches_from_file
import globals.global_var as glo


def intra_cluster_scheduling():
    """Perform intra-domain task scheduling
    """
    # 1. create one domain and one cluster
    cluster = create_one_cluster()

    # 2. add machines to cluster
    machine_file_path = glo.machine_file_path
    machine_list = load_machines_from_file(machine_file_path)
    for machine in machine_list:
        cluster.add_machine(machine)

    # 3. load tasks
    task_file_path = glo.task_file_path
    task_batch_list = load_task_batches_from_file(task_file_path)

    # 4. set scheduler for cluster
    machine_num = len(machine_list)
    task_batch_num = len(task_batch_list)
    scheduler = RoundRobinScheduler(len(cluster.machine_list))
    # scheduler = DQNScheduler(machine_num, task_batch_num)
    scheduler_name = scheduler.__class__.__name__
    glo.task_run_results_path = glo.results_path_list[scheduler_name]
    cluster.set_scheduler(scheduler)

    # 5. commit tasks to cluster
    for batch in task_batch_list:
        cluster.commit_tasks(batch)

    # 6. reset cluster
    cluster.reset()


def inter_domain_scheduling():
    """Perform inter-domain task scheduling
    """
    # 1. create one domain
    domain = create_one_domain()

    # 2. add machines to domain
    machine_file_path = glo.machine_file_path
    machine_list = load_machines_from_file(machine_file_path)
    for machine in machine_list:
        domain.add_machine(machine)

    # 3. clustering machines in the domain
    cluster_num = 6
    domain.clustering_machines(cluster_num)

    # 4. load tasks
    task_file_path = glo.task_file_path
    task_batch_list = load_task_batches_from_file(task_file_path)

    # 5.

    # 4. set scheduler for cluster
    machine_num = len(machine_list)
    task_batch_num = len(task_batch_list)
    scheduler = RoundRobinScheduler(len(cluster.machine_list))
    # scheduler = DQNScheduler(machine_num, task_batch_num)
    scheduler_name = scheduler.__class__.__name__
    glo.task_run_results_path = glo.results_path_list[scheduler_name]
    cluster.set_scheduler(scheduler)

    # 5. commit tasks to cluster
    for batch in task_batch_list:
        cluster.commit_tasks(batch)

    # 6. reset cluster
    cluster.reset()


if __name__ == "__main__":
    intra_cluster_scheduling()
