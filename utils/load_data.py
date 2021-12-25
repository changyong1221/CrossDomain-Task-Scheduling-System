from core.machine import Machine
from core.task import TaskRunInstance


def load_tasks_from_file(file_path):
    """Load tasks from a given file

    Task dataset in the file should satisfy the following format in each line:
    task_id,commit_time,mi,cpu_utilization,size

    :return: a vector of tasks in shape(n,5), n is the number of records
    """
    task_list = []
    with open(file_path, 'r') as f:
        for line in f:
            task = [float(val) for val in line.rstrip().split(',')]
            task[0] = int(task[0])
            task_list.append(task)
        f.close()
    return task_list


def load_task_batches_from_file(file_path, delimiter='\t'):
    """Load task batches from a given file (A batch is tasks committed at the same time)

    Task dataset in the file should satisfy the following format in each line:
    task_id,commit_time,mi,cpu_utilization,size

    :return: a vector of task batches in shape(m,n,5), m is the number of batches, n is the number of records in batch
    """
    task_batches = []
    with open(file_path, 'r') as f:
        batch = []
        for line in f:
            task = [float(val) for val in line.rstrip().split(delimiter)]
            task[0] = int(task[0])
            if batch != [] and task[1] != batch[0].commit_time:
                task_batches.append(batch)
                batch = []
            batch.append(TaskRunInstance(task[0], task[1], task[2], task[3], task[4]))
        task_batches.append(batch)
        f.close()
    return task_batches


def load_machines_from_file(file_path):
    """Load machines from a given file

    Machine dataset in the file should satisfy the following format in each line:
    machine_id,mips,memory,bandwidth

    :return: a vector of Machine objects in shape(m,4), m is the number of records
    """
    machine_list = []
    with open(file_path, 'r') as f:
        for line in f:
            machine = [float(val) for val in line.rstrip().split(',')]
            machine[0] = int(machine[0])
            machine_list.append(Machine(machine[0], machine[1], machine[2], machine[3]))
        f.close()
    return machine_list


if __name__ == '__main__':
    pass
    # f_path = "../dataset/test/machine.txt"
    # m_list = load_machines_from_file(f_path)
    # print(m_list)
