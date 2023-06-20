from utils.write_file import write_list_to_file
from utils.get_position import compute_distance_by_location
from utils.file_check import check_and_build_dir
from utils.log import print_log
import globals.global_var as glo
import numpy as np


class Task(object):
    def __init__(self, task_id, commit_time, mi, cpu_utilization, size):
        """Initialization
        """
        self.task_id = task_id
        self.commit_time = commit_time
        self.mi = mi
        self.cpu_utilization = cpu_utilization
        self.size = size

    def get_task_commit_time(self):
        """Return task commit time
        """
        return self.commit_time

    def get_task_mi(self):
        """Return task mi of current task
        """
        return self.mi

    def get_task_cpu_utilization(self):
        """Return cpu utilization of current task
        """
        return self.cpu_utilization

    def get_task_size(self):
        """Return task size (bytes) of current task
        """
        return self.size
    
    def get_array_data(self):
        """Return numpy array for task type prediction
        """
        return [self.mi, self.cpu_utilization, self.size]

class TaskRunInstance(Task):
    def __init__(self, task_id, commit_time, mi, cpu_utilization, size):
        """Initialization
        """
        super().__init__(task_id, commit_time, mi, cpu_utilization, size)
        self.task_transfer_time = 0
        self.task_waiting_time = 0
        self.task_executing_time = 0
        self.task_processing_time = 0
        self.task_type = "simple"
        self.is_done = False

    def run_on_machine(self, machine, multidomain_id):
        """Run task on a specified machine

        传播时延 = 数据包大小(Mb) / 以太网链路速率(Mbps) + 传播距离(m) / 链路传播速率(m/s)
        默认链路传播速率 = 2.8 * 10^8 m/s
        """
        line_transmit_time = compute_distance_by_location(machine.longitude,
                                                          machine.latitude,
                                                          glo.location_longitude,
                                                          glo.location_latitude) / glo.line_transmit_speed
        # print(f"line_transmit_time: {line_transmit_time} s")
        self.task_waiting_time += round(max(0, machine.get_transfer_finish_time() - self.commit_time))  # 等待开始传输
        self.task_transfer_time = round((self.size * 8) / machine.get_bandwidth() + line_transmit_time, 4)    # 任务传输时间
        machine.set_transfer_finish_time(self.commit_time + self.task_waiting_time + self.task_transfer_time)  # 更新机器的传输完成时间
        self.task_waiting_time += round(max(0, machine.get_execution_finish_time() - machine.get_transfer_finish_time()), 4)    # 等待开始执行
        self.task_executing_time = round(self.mi / (machine.get_mips() * self.cpu_utilization), 4)      # 任务执行时间
        self.task_processing_time = self.task_transfer_time + self.task_waiting_time + self.task_executing_time     # 任务总的处理时间
        machine.set_execution_finish_time(self.commit_time + self.task_processing_time)
        scheduler_name = glo.current_scheduler
        if self.task_executing_time > 2 * self.task_transfer_time:
            self.task_type = "cpu-intensive"
        elif self.task_transfer_time > 2 * self.task_executing_time:
            self.task_type = "io-intensive"
        if glo.is_federated:
            output_dir = f"results/task_run_results/federated/client-{multidomain_id}/{glo.federated_round}"
            check_and_build_dir(output_dir)
            output_path = output_dir + f"/{scheduler_name}_task_run_results.txt"
            if glo.is_test:
                output_dir = f"results/task_run_results/federated/federated_test/{glo.federated_round}"
                check_and_build_dir(output_dir)
                output_path = output_dir + f"/{scheduler_name}_task_run_results.txt"
                # output_path = output_dir + f"/{scheduler_name}_task_run_results_test.txt"
            output_list = [self.task_id, self.get_task_mi() / self.get_task_cpu_utilization(), self.get_task_size(), machine.get_machine_id(), machine.get_mips(),
                           self.task_transfer_time, self.task_waiting_time,
                           self.task_executing_time, self.task_processing_time, self.task_type]
            write_list_to_file(output_list, output_path, mode='a+')
        else:
            output_dir = f"results/task_run_results/{glo.current_dataset}{glo.records_num}/{scheduler_name}/"
            if glo.current_batch_size != 0:
                output_dir += f"{glo.current_batch_size}/"
            check_and_build_dir(output_dir)
            output_path = output_dir + f"{scheduler_name}_task_run_results.txt"
            output_list = [self.task_id, self.get_task_mi() / self.get_task_cpu_utilization(), self.get_task_size(), machine.get_machine_id(), machine.get_mips(),
                           self.task_transfer_time, self.task_waiting_time,
                           self.task_executing_time, self.task_processing_time, self.task_type]
            write_list_to_file(output_list, output_path, mode='a+')
        print_log(f"task({self.task_id}) finished, processing time: {round(self.task_processing_time, 4)} s")
        self.is_done = True

    def get_task_processing_time(self):
        """Return task processing time
        """
        if self.is_done:
            return self.task_processing_time
        else:
            return -1

    def get_task_waiting_time(self):
        """Return task waiting time
        """
        if self.is_done:
            return self.task_waiting_time
        else:
            return -1
            
    def get_task_executing_time(self):
        """Return task executing time
        """
        if self.is_done:
            return self.task_executing_time
        else:
            return -1
            
    def get_task_transfer_time(self):
        """Return task transfer time
        """
        if self.is_done:
            return self.task_transfer_time
        else:
            return -1
    
    def get_task_type(self):
        """Return task type
        """
        return self.task_type
