from scheduler.scheduler import Scheduler
from utils.state_representation import get_state, get_machine_mips_weight, get_machine_bandwidth_weight
import numpy as np


class WeightedRandomScheduler(Scheduler):
    def __init__(self, machine_list):
        """Initialization
        """
        self.machine_mips_weight_list = get_machine_mips_weight(machine_list)
        self.machine_bandwidth_weight_list = get_machine_bandwidth_weight(machine_list)
        self.machine_weight_list = [((self.machine_mips_weight_list[i] + self.machine_bandwidth_weight_list[i]) * 0.5) for i in range(len(self.machine_mips_weight_list))]
        # self.accu_machine_weight_list = [np.sum(self.machine_mips_weight_list[:i+1]) for i in range(len(self.machine_mips_weight_list))]
        # self.accu_machine_weight_list = [np.sum(self.machine_bandwidth_weight_list[:i+1]) for i in range(len(self.machine_bandwidth_weight_list))]
        self.accu_machine_weight_list = [np.sum(self.machine_weight_list[:i+1]) for i in range(len(self.machine_weight_list))]

    def schedule(self, task_num):
        """Schedule tasks in weighted random way

        :return scheduling results, which is a list of machine id
        """
        # print(self.machine_mips_weight_list)
        # print(self.machine_bandwidth_weight_list)
        # print(self.machine_weight_list)
        scheduling_results = []
        # 对mips加权
        for task in range(task_num):
            rand_prob = np.random.uniform()
            for j, prob in enumerate(self.accu_machine_weight_list):
                if rand_prob <= prob:
                    scheduling_results.append(j)
                    break
        return scheduling_results