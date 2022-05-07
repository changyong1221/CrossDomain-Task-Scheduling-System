import math
import numpy as np
from scheduler.scheduler import Scheduler
from model.dqn.dqn import DQN
from utils.state_representation import get_state
from utils.log import print_log


class DQNScheduler(Scheduler):
    def __init__(self, multidomain_id, machine_num, task_batch_num, machine_kind_num_list, machine_kind_idx_range_list,
                 is_federated=False, epsilon_decay=0.998, prob=0.5, balance_prob=0.5):
        """Initialization

        input : a list of tasks
        output: scheduling results, which is a list of machine id
        """
        self.task_dim = 3
        self.machine_dim = 2

        self.state_all = []  # 存储所有的状态 [None,2+2*20]
        self.action_all = []  # 存储所有的动作 [None,1]
        self.reward_all = []  # 存储所有的奖励 [None,1]
        self.machine_kind_idx_range_list = machine_kind_idx_range_list

        self.double_dqn = True
        self.dueling_dqn = True
        self.optimized_dqn = False
        self.prioritized_memory = False
        self.DRL = DQN(multidomain_id, self.task_dim, machine_num, self.machine_dim, machine_kind_num_list,
                       self.machine_kind_idx_range_list,
                       self.double_dqn, self.dueling_dqn, self.optimized_dqn, self.prioritized_memory, is_federated, epsilon_decay, prob, balance_prob)
        self.DRL.max_step = task_batch_num
        self.cur_step = 0
        self.alpha = 0.5
        self.beta = 0.5
        self.C = 10
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

    def learn(self, task_instance_batch, machines_id, makespan, machine_list):
        reward_save_path = f"backup/test-0506/D3QN-OPT4/test/reward.txt"
        action_save_path = f"backup/test-0506/D3QN-OPT4/test/action.txt"
        for idx, task in enumerate(task_instance_batch):  # 便历新提交的一批任务，记录动作和奖励
            self.action_all.append([machines_id[idx]])
            with open(action_save_path, 'a+') as f:
                f.write(f"{task.get_task_mi()}\t{machines_id[idx]}\n")

            # reward = self.C / (self.alpha * math.log(task_item, 10) +
            #                   self.beta * math.log(makespan_item, 10))
            
            w = 1000          
            reward = math.log(task.get_task_mi() * w) / (self.alpha * math.log(task.get_task_processing_time() * w, 10) +
                              self.beta * math.log(makespan * w, 10))
            
            # reward = task.get_task_mi() / task.get_task_processing_time() / 100
            
            # reward = round(task.get_task_waiting_time(), 3)
            # reward = 10 / (math.log(task.get_task_waiting_time() + 10, 10))
            
            # if task.get_task_mi() == 59000 and machine_list[machines_id[idx]].get_mips() == 24000:
            #     reward = 3
            # elif task.get_task_mi() == 99000 and machine_list[machines_id[idx]].get_mips() == 4000:
            #     reward = 2
            # else:
            #     reward = -1
                               
            # reward = task.get_task_mi() / task.get_task_processing_time() / 100
            # reward = task.get_task_processing_time() / task.get_task_mi()
            with open(reward_save_path, 'a+') as f:
                f.write(str(round(reward, 3)) + "\n")
            # reward = task.get_task_mi() / task.get_task_processing_time() / 100
            # if task.get_task_mi() > 100000 and machines_id[idx] > 15:
            #     reward = 100
            # print("machine_id: ", machines_id[idx])
            # print("task_mi: ", task.get_task_mi())
            # print("task_processing_time: ", task.get_task_processing_time())
            # print("reward: ", reward)
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
        print("cur_step: ", self.cur_step)
        if self.cur_step > 400:
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
