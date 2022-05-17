import math
import numpy as np
from scheduler.scheduler import Scheduler
from model.dqn.dqn import DQN
from utils.state_representation import get_state, get_machine_weight
from utils.log import print_log
import torch


class DQNScheduler(Scheduler):
    def __init__(self, multidomain_id, machine_list, machine_num, task_batch_num, machine_kind_num_list, machine_kind_idx_range_list,
                 is_federated=False, epsilon_decay=0.998, prob=0.5, balance_prob=0.5):
        """Initialization

        input : a list of tasks
        output: scheduling results, which is a list of machine id
        """
        self.task_dim = 3
        self.machine_dim = 3
        self.machine_num = machine_num

        self.state_all = []  # 存储所有的状态 [None,2+2*20]
        self.action_all = []  # 存储所有的动作 [None,1]
        self.reward_all = []  # 存储所有的奖励 [None,1]
        self.machine_list = machine_list
        self.machine_weight_list = get_machine_weight(machine_list)
        self.machine_kind_idx_range_list = machine_kind_idx_range_list
        self.machine_assigned_task_list = [(int)(self.machine_weight_list[i] * 1000) for i in range(machine_num)]
        self.cur_weight_list = self.machine_weight_list[:]

        self.double_dqn = True
        self.dueling_dqn = True
        self.optimized_dqn = False
        self.prioritized_memory = False
        self.DRL = DQN(multidomain_id, self.task_dim, machine_num, self.machine_dim, machine_kind_num_list,
                       self.machine_kind_idx_range_list,
                       self.double_dqn, self.dueling_dqn, self.optimized_dqn, self.prioritized_memory, is_federated, epsilon_decay, prob, balance_prob, self.machine_weight_list)
        self.DRL.max_step = task_batch_num
        self.cur_step = 0
        self.replay_memory_size = 10000
        self.alpha = 0.5
        self.beta = 0.5
        self.C = 10
        self.last_p = 0
        print_log("DQN网络初始化成功！")

    def schedule(self, task_instance_batch):
        task_num = len(task_instance_batch)

        states = get_state(task_instance_batch, self.machine_list, self.cur_weight_list)
        self.state_all += states
        # self.state_all.append(states)
        machines_id = self.DRL.choose_action(np.array(states))  # 通过调度算法得到分配 id
        machines_id = machines_id.astype(int).tolist()
        return machines_id
        # if (step == 1): print_log("machines_id: " + str(machines_id))

    def learn(self, task_instance_batch, machines_id, makespan):
        # 设计奖励函数
        # print(self.machine_weight_list)
        # print(self.machine_assigned_task_list)
        # def compute_std(machine_weight_list, machine_assigned_task_list):
        #     machine_weight_tensor = torch.Tensor(machine_weight_list)
        #     machine_assigned_tasks_tensor = torch.Tensor(machine_assigned_task_list)
        #     weighted_machine_assigned_tasks_tensor = machine_assigned_tasks_tensor / machine_weight_tensor
        #     # print(f"weighted_machine_assigned_tasks_tensor: {weighted_machine_assigned_tasks_tensor}")
        #     tmp_std = torch.std(weighted_machine_assigned_tasks_tensor, dim=0)
        #     return tmp_std
            
        # m_std = compute_std(machine_weight_list, self.machine_assigned_task_list)
        
        
        reward_save_path = f"backup/test-0517/D3QN-OPT/train3/reward.txt"
        action_save_path = f"backup/test-0517/D3QN-OPT/train3/action.txt"
        info_save_path = f"backup/test-0517/D3QN-OPT/train3/info.txt"
        
        batch_reward_list = []
        for idx, task in enumerate(task_instance_batch):  # 便历新提交的一批任务，记录动作和奖励
            machine_id = machines_id[idx]
            self.action_all.append([machine_id])
            self.machine_assigned_task_list[machine_id] += 1
            with open(action_save_path, 'a+') as f:
                f.write(f"{machine_id}\n")

            # reward = self.C / (self.alpha * math.log(task_item, 10) +
            #                   self.beta * math.log(makespan_item, 10))
            # GoCJ数据集，w=10合适
            # Alibaba数据集，w=1000合适，否则会出现除0错误
            w = 1000
            # reward = math.log(w) / (math.log(task.get_task_processing_time() * w, 10))
            # reward = math.log(task.get_task_mi() * w) / (self.alpha * math.log(task.get_task_processing_time() * w, 10) +
            #                   self.beta * math.log(m_std * w, 10))
            # reward = math.log(task.get_task_mi() * w) / (self.alpha * math.log(task.get_task_processing_time() * w, 10) +
            #                   self.beta * math.log(makespan * w, 10))
            total_task_num = np.sum(self.machine_assigned_task_list)
            cur_weight = self.machine_assigned_task_list[machine_id] / total_task_num
            self.cur_weight_list[machine_id] = cur_weight
            weight_ratio = cur_weight / self.machine_weight_list[machine_id]
            # print(cur_weight)
            # if cur_weight > self.machine_weight_list[machine_id] * 1.1:
            #     reward_scale = 0.5
            # elif cur_weight < self.machine_weight_list[machine_id] * 0.8:
            #     reward_scale = 1.5
            # else:
            #     reward_scale = 1
            
            # m_wait_time = task.get_task_commit_time() - self.machine_list[machine_id].get_finish_time()
            # m_wait_elem = 0 if m_wait_time <= 10 else math.log(m_wait_time, 10)
            
            reward_scale = 1
            # if m_wait_elem > 1:
            #     reward_scale = m_wait_elem
            # print(reward_scale)
            # exit()
            # reward = reward_scale * (self.alpha * math.log(task.get_task_mi() * w, 10) / (math.log(task.get_task_executing_time() * w, 10)) +
            #                   self.beta * 20 / weight_ratio)
            reward = reward_scale * (math.log(task.get_task_mi() * w, 10) / (self.alpha * math.log(task.get_task_executing_time() * w, 10) +
                              self.beta * math.log(weight_ratio * w, 10)))
            with open(info_save_path, 'a+') as f:
                f.write(f"{round(reward,3)}\t{task.get_task_mi()}\t{task.get_task_executing_time()}\t{math.log(task.get_task_executing_time() * w, 10)}\t{math.log(weight_ratio * w, 10)}\t{weight_ratio}\t{reward_scale}\t{machine_id}\n")

            batch_reward_list.append(reward)
            # with open(reward_save_path, 'a+') as f:
            #     f.write(str(round(reward, 3)) + "\n")
            
            # reward = task.get_task_mi() / task.get_task_processing_time() / 100
            # if task.get_task_mi() > 100000 and machines_id[idx] > 15:
            #     reward = 100
            # print("machine_id: ", machines_id[idx])
            # print("task_mi: ", task.get_task_mi())
            # print("task_processing_time: ", task.get_task_processing_time())
            # print("reward: ", reward)
            self.reward_all.append([reward])  # 计算奖励
        
        reward_mean = np.mean(np.array(batch_reward_list))
        with open(reward_save_path, 'a+') as f:
            f.write(str(round(reward_mean, 3)) + "\n")
        # 减少存储数据量
        if len(self.state_all) > 2 * self.replay_memory_size:
            self.state_all = self.state_all[-self.replay_memory_size:]
            self.action_all = self.action_all[-self.replay_memory_size:]
            self.reward_all = self.reward_all[-self.replay_memory_size:]

        # 如果使用prioritized memory
        if self.prioritized_memory:
            for i in range(len(task_instance_batch)):
                self.DRL.append_sample([self.state_all[-2 + i]], [self.action_all[-1 + i]],
                                  [self.reward_all[-1 + i]], [self.state_all[-1 + i]])

        # 先学习一些经验，再学习
        print("cur_step: ", self.cur_step)
        # if self.cur_step >= 0 and len(self.state_all) > 2:
        if self.cur_step > 800:
            # 截取最后10000条记录
            new_state = np.array(self.state_all, dtype=np.float32)[-self.replay_memory_size:-1]
            new_action = np.array(self.action_all, dtype=np.float32)[-self.replay_memory_size:-1]
            new_reward = np.array(self.reward_all, dtype=np.float32)[-self.replay_memory_size:-1]
            self.DRL.store_memory(new_state, new_action, new_reward)
            self.DRL.step = self.cur_step
            loss = self.DRL.learn()
            print_log(f"step: {self.cur_step}, loss: {loss}")
        self.cur_step += 1
