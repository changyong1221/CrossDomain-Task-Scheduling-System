from utils.write_file import write_list_to_file
import globals.global_var as glo


class Task(object):
    def __init__(self, task_id, commit_time, mi, cpu_utilization, size):
        """Initialization
        """
        self.task_id = task_id
        self.commit_time = commit_time
        self.mi = mi
        self.cpu_utilization = cpu_utilization
        self.size = size

    def get_task_mi(self):
        """Return task mi of current task
        """
        return self.mi

    def get_task_cpu_utilization(self):
        """Return cpu utilization of current task
        """
        return self.cpu_utilization

    def get_task_size(self):
        """Return task size of current task
        """
        return self.size


class TaskRunInstance(Task):
    def __init__(self, task_id, commit_time, mi, cpu_utilization, size):
        """Initialization
        """
        super().__init__(task_id, commit_time, mi, cpu_utilization, size)
        self.task_transfer_time = 0
        self.task_waiting_time = 0
        self.task_executing_time = 0
        self.task_processing_time = 0
        self.is_done = False

    def run_on_machine(self, machine):
        """Run task on a specified machine

        传播时延 = 数据包大小(Mb) / 以太网链路速率(Mbps) + 传播距离(m) / 链路传播速率(m/s)
        默认链路传播速率 = 2.8 * 10^8 m/s
        """
        self.task_transfer_time = self.size / machine.get_bandwidth() + 0
        self.task_waiting_time = max(0, machine.get_finish_time() - self.commit_time)
        self.task_executing_time = self.mi / (machine.get_mips() * self.cpu_utilization)
        self.task_processing_time = self.task_transfer_time + self.task_waiting_time + self.task_executing_time
        machine.set_finish_time(self.commit_time + self.task_processing_time)
        self.is_done = True
        output_path = glo.task_run_results_path
        output_list = [self.task_id, machine.get_machine_id(), self.task_transfer_time, self.task_waiting_time,
                       self.task_executing_time, self.task_processing_time]
        write_list_to_file(output_list, output_path, mode='a+')

    def get_task_processing_time(self):
        """Return task processing time
        """
        if self.is_done:
            return self.task_processing_time
        else:
            return -1

