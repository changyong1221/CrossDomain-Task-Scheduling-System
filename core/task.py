from utils.write_file import write_list_to_file
from utils.get_position import compute_distance_by_location
from utils.file_check import check_and_build_dir
from utils.log import print_log
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

    def run_on_machine(self, machine, multidomain_id):
        """Run task on a specified machine

        传播时延 = 数据包大小(Mb) / 以太网链路速率(Mbps) + 传播距离(m) / 链路传播速率(m/s)
        默认链路传播速率 = 2.8 * 10^8 m/s
        """
        line_transmit_time = compute_distance_by_location(machine.longitude,
                                                          machine.latitude,
                                                          glo.location_longitude,
                                                          glo.location_latitude) / glo.line_transmit_speed
        # print_log(f"line_transmit_time: {line_transmit_time} s")
        self.task_transfer_time = round(self.size / machine.get_bandwidth() + line_transmit_time, 4)
        self.task_waiting_time = round(max(0, machine.get_finish_time() - self.commit_time), 4)
        self.task_executing_time = round(self.mi / machine.get_mips(), 4)
        # self.task_executing_time = self.mi / (machine.get_mips() * self.cpu_utilization)
        self.task_processing_time = self.task_transfer_time + self.task_waiting_time + self.task_executing_time
        machine.set_finish_time(self.commit_time + self.task_processing_time)
        self.is_done = True
        scheduler_name = glo.current_scheduler
        if glo.is_federated:
            output_dir = f"results/task_run_results/federated/client-{multidomain_id}/{glo.federated_round}"
            check_and_build_dir(output_dir)
            output_path = output_dir + f"/{scheduler_name}_task_run_results.txt"
            if glo.is_test:
                output_dir = f"results/task_run_results/federated/federated_test/{glo.federated_round}"
                check_and_build_dir(output_dir)
                output_path = output_dir + f"/{scheduler_name}_task_run_results.txt"
                # output_path = output_dir + f"/{scheduler_name}_task_run_results_test.txt"
            output_list = [self.task_id, self.get_task_mi(), machine.get_machine_id(), machine.get_mips(),
                           self.task_transfer_time, self.task_waiting_time,
                           self.task_executing_time, self.task_processing_time]
            write_list_to_file(output_list, output_path, mode='a+')
        else:
            output_dir = f"results/task_run_results/{glo.current_dataset}{glo.records_num}/{scheduler_name}/"
            if glo.current_batch_size != 0:
                output_dir += f"{glo.current_batch_size}/"
            check_and_build_dir(output_dir)
            output_path = output_dir + f"{scheduler_name}_task_run_results2.txt"
            output_list = [self.task_id, self.get_task_mi(), machine.get_machine_id(), machine.get_mips(),
                           self.task_transfer_time, self.task_waiting_time,
                           self.task_executing_time, self.task_processing_time]
            write_list_to_file(output_list, output_path, mode='a+')
        print_log(f"task({self.task_id}) finished, processing time: {round(self.task_processing_time, 4)} s")

    def get_task_processing_time(self):
        """Return task processing time
        """
        if self.is_done:
            return self.task_processing_time
        else:
            return -1

