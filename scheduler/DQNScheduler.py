import numpy as np
from scheduler.scheduler import Scheduler
from model.dqn.dqn import DQN
from utils.log import print_log


class DQNScheduler(Scheduler):
    def __init__(self, multidomain_id, machine_num, task_batch_num, vm_task_capacity, is_federated=False):
        """Initialization

        input : a list of tasks
        output: scheduling results, which is a list of machine id
        """
        self.task_dim = 3
        self.machine_dim = 2

        self.state_all = []  # 存储所有的状态 [None,2+2*20]
        self.action_all = []  # 存储所有的动作 [None,1]
        self.reward_all = []  # 存储所有的奖励 [None,1]
        self.vm_task_capacity = vm_task_capacity

        self.double_dqn = True
        self.dueling_dqn = True
        self.optimized_dqn = False
        self.prioritized_memory = False
        self.DRL = DQN(multidomain_id, self.task_dim, machine_num, self.machine_dim, self.vm_task_capacity,
                       self.double_dqn, self.dueling_dqn, self.optimized_dqn, self.prioritized_memory, is_federated)
        self.DRL.max_step = task_batch_num
        self.cur_step = 0
        print_log("DQN网络初始化成功！")

    def schedule(self, task_instance_batch, machine_list):
        task_num = len(task_instance_batch)

        states = get_state(task_instance_batch, machine_list)
        self.state_all += states
        # self.state_all.append(states)
        machines_id = self.DRL.choose_action(np.array(states))  # 通过调度算法得到分配 id
        machines_id = machines_id.astype(int).tolist()
        return machines_id
        # if (step == 1): print_log("machines_id: " + str(machines_id))

    def learn(self, task_instance_batch, machines_id):
        for idx, task in enumerate(task_instance_batch):  # 便历新提交的一批任务，记录动作和奖励
            self.action_all.append([machines_id[idx]])
            # reward = task.get_task_mi() / task.get_task_processing_time() / 100
            reward = task.get_task_mi() / task.get_task_processing_time()
            self.reward_all.append([reward])  # 计算奖励

        # 减少存储数据量
        if len(self.state_all) > 20000:
            self.state_all = self.state_all[-10000:]
            self.action_all = self.action_all[-10000:]
            self.reward_all = self.reward_all[-10000:]

        # 如果使用prioritized memory
        if self.prioritized_memory:
            for i in range(len(task_instance_batch)):
                self.DRL.append_sample([self.state_all[-2 + i]], [self.action_all[-1 + i]],
                                  [self.reward_all[-1 + i]], [self.state_all[-1 + i]])

        # 先学习一些经验，再学习
        if self.cur_step > 10:
            # 截取最后10000条记录
            # print_log(type(self.state_all))
            # print_log(self.state_all)
            # array = np.array(self.state_all)
            # print_log(array)
            # print_log(type(array))
            new_state = np.array(self.state_all, dtype=np.float32)[-10000:-1]
            new_action = np.array(self.action_all, dtype=np.float32)[-10000:-1]
            new_reward = np.array(self.reward_all, dtype=np.float32)[-10000:-1]
            self.DRL.store_memory(new_state, new_action, new_reward)
            self.DRL.step = self.cur_step
            loss = self.DRL.learn()
            print_log(f"step: {self.cur_step}, loss: {loss}")
        self.cur_step += 1


# 通过任务和机器获取状态
def get_state(task_list, machine_list):
    commit_time = task_list[0].commit_time  # 当前批次任务的开始时间
    machines_state = []
    for machine in machine_list:
        machines_state.append(machine.get_mips())
        machines_state.append(max(machine.get_finish_time() - commit_time, 0))  # 等待时间
        # if (machine.next_start_time - start_time > 0):
        #     print_log("machines_state: ", machines_state)
    tasks_state = []
    for i, task in enumerate(task_list):
        task_state = []
        task_state.append(task.get_task_mi())
        task_state.append(task.get_task_cpu_utilization())
        task_state.append(task.get_task_mi() / machine_list[0].get_bandwidth())  # 传输时间
        task_state += machines_state  # 由于是DQN，所以一个任务状态加上多个虚拟机状态
        # if (i == 1): print_log(task_state)
        # print_log("task_state: ", task_state)
        tasks_state.append(task_state)
    # 返回值 [[[153.0, 0.79, 0.34, 600, 0, 600, 0, 500, 0, 500, 0, 400, 0, 400, 0, 300, 0, 300, 0, 200, 0, 200, 0]... ]]
        #           任务长度，任务利用率，任务传输时间，vm1_mips, vm1_waitTime, vm2....
    return tasks_state
