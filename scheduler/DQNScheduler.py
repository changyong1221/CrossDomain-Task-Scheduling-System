import math
import numpy as np
from scheduler.scheduler import Scheduler
from model.dqn.dqn import DQN
from predictor.TaskTypePredictor import TaskTypePredictor
from utils.state_representation import get_state, get_machine_mips_weight, get_machine_bandwidth_weight
from utils.log import print_log
import torch


class RewardPool():
    def __init__(self, reward_pool_size, action_num):
        self.action_num = action_num
        self.capacity = reward_pool_size
        self.pool = [[] for i in range(self.action_num)]
        self.reward_weight_list = [(1 / self.action_num) for i in range(self.action_num)]
        self.reward_sum_list = [0 for i in range(self.action_num)]
        self.cur_id_list = [0 for i in range(self.action_num)]
    
    def add(self, action_id, reward):
        if len(self.pool[action_id]) < self.capacity:
            self.reward_sum_list[action_id] += reward
            self.pool[action_id].append(reward)
        else:
            self.reward_sum_list[action_id] += (reward - self.pool[action_id][self.cur_id_list[action_id]])
            self.pool[action_id][self.cur_id_list[action_id]] = reward
        self.cur_id_list[action_id] = (self.cur_id_list[action_id] + 1) % self.capacity
    
    def get_reward_weight_list(self):
        total_reward = 0
        reward_mean_list = [0 for i in range(self.action_num)]
        for action_id in range(self.action_num):
            if len(self.pool[action_id]) == 0:
                reward_mean_list[action_id] = 0
            else:
                reward_mean_list[action_id] = self.reward_sum_list[action_id] / len(self.pool[action_id])
            total_reward += reward_mean_list[action_id]
        if total_reward == 0:
            return self.reward_weight_list
        for action_id in range(self.action_num):
            self.reward_weight_list[action_id] = reward_mean_list[action_id] / total_reward
        return self.reward_weight_list


class DQNScheduler(Scheduler):
    def __init__(self, multidomain_id, machine_list, machine_num, task_batch_num, machine_kind_num_list, machine_kind_idx_range_list,
                 is_federated=False, is_test=False, epsilon_decay=0.998, prob=0.5, balance_prob=0.5):
        """Initialization

        input : a list of tasks
        output: scheduling results, which is a list of machine id
        """
        self.task_dim = 2
        self.machine_dim = 3
        self.machine_num = machine_num

        self.state_all = []  # 存储所有的状态 [None,2+2*20]
        self.action_all = []  # 存储所有的动作 [None,1]
        self.reward_all = []  # 存储所有的奖励 [None,1]
        self.machine_list = machine_list
        self.machine_remain_time_weight_list = [0.1 for i in range(self.machine_num)]
        self.machine_mips_weight_list = get_machine_mips_weight(machine_list)
        self.machine_bandwidth_weight_list = get_machine_bandwidth_weight(machine_list)
        self.cpu_reward_weight_list = [0 for i in range(self.machine_num)]
        self.io_reward_weight_list = [0 for i in range(self.machine_num)]
        self.reward_pool_size = 50
        self.cpu_reward_pool = RewardPool(self.reward_pool_size, self.machine_num)
        self.io_reward_pool = RewardPool(self.reward_pool_size, self.machine_num)
        self.simple_reward_pool = RewardPool(self.reward_pool_size, self.machine_num)
        # 实际machine weight list = [0.0026595744680851063, 0.0026595744680851063, 0.005319148936170213, 0.005319148936170213, 0.02127659574468085, 0.02127659574468085, 0.026595744680851064, 0.026595744680851064, 0.026595744680851064, 0.031914893617021274, 0.031914893617021274, 0.031914893617021274, 0.031914893617021274, 0.0425531914893617, 0.06382978723404255, 0.0851063829787234, 0.09574468085106383, 0.1276595744680851, 0.1595744680851064, 0.1595744680851064]
        # 微调machine weight list
        # self.machine_weight_list = [0.0047619,0.00952381,0.01428571,0.01904762,0.02380952,0.02857143,0.03333333,0.03809524,0.04285714,0.04761905,0.05238095,0.05714286,0.06190476,0.06666667,0.07142857,0.07619048,0.08095238,0.08571429,0.09047619,0.0952381]
        self.machine_kind_idx_range_list = machine_kind_idx_range_list
        self.machine_assigned_task_list = [(int)(self.machine_mips_weight_list[i] * 1000) for i in range(machine_num)]
        self.cur_weight_list = self.machine_mips_weight_list[:]

        self.double_dqn = True
        self.dueling_dqn = True
        self.prioritized_memory = True
        # self.double_dqn = False
        # self.dueling_dqn = False
        # self.prioritized_memory = False
        self.task_type_predictor = TaskTypePredictor()
        self.task_type_predictor.load_model()
        self.DRL = DQN(multidomain_id, self.task_dim, machine_num, self.machine_dim, machine_kind_num_list,
                       self.machine_kind_idx_range_list,
                       self.double_dqn, self.dueling_dqn, self.prioritized_memory, is_federated, is_test, epsilon_decay, prob, balance_prob, self.machine_mips_weight_list, self.machine_bandwidth_weight_list)
        self.DRL.max_step = task_batch_num
        self.cur_step = 0
        self.replay_memory_size = 10000
        self.alpha = 0.5
        self.beta = 0.5
        self.C = 10
        self.last_p = 0
        self.next_learn_id = 0
        print_log("DQN网络初始化成功！")

    def schedule(self, task_instance_batch):
        task_num = len(task_instance_batch)
        
        is_cpu_task = True
        if task_instance_batch[0].size > 10:
            is_cpu_task = False
        
        states = get_state(task_instance_batch, self.machine_list, self.machine_remain_time_weight_list)
        
        # 预测任务类型
        x_predict = []
        for task in task_instance_batch:
            x_predict.append(task.get_array_data())
        task_type_id = self.task_type_predictor.predict(x_predict, load_model=False)
        task_type_str = ['simple', 'cpu-intensive', 'io-intensive']
        print(f"predict task type: {task_type_str[task_type_id]}")
        # self.state_all += states
        # self.state_all.append(states) # 这里不能用append，必须用+=，+=只append内部元素
        # cpu_reward_weight_list = [0.004416384965503231, 0.011205055838231724, 0.022944260528961258, 0.03094614478736021, 0.03206255497602222, 0.04550863765254782, 0.04552471746041207, 0.0465702433157, 0.05266250222110345, 0.050054206457905155, 0.0541331738665265, 0.05546446520437984, 0.05526258113186102, 0.060718125901592666, 0.05457588457585535, 0.07073167672159798, 0.07305444747536434, 0.06116463535197114, 0.06854170140614693, 0.10445860016095705]
        # io_reward_weight_list = [0.07250139073423022, 0.04742400740441555, 0.04927823522923519, 0.05264191635322647, 0.052789309755142656, 0.049672577300584025, 0.051430940779620855, 0.05329953600281507, 0.0476428903778909, 0.05117058129299792, 0.052825563702029434, 0.052626547709445484, 0.05138387501724827, 0.05594304932799998, 0.015405429918538396, 0.0541610164115563, 0.06504531124317864, 0.05587481136226383, 0.05406466828578392, 0.014818341791796672]
        # simple_reward_weight_list = [0.040587391738359256, 0.04379513930842527, 0.045952947191251724, 0.04840158545448957, 0.04807017101297641, 0.049168278970038136, 0.05075631946827112, 0.05058496284082964, 0.049017203594015304, 0.05124697730476538, 0.05256388475499857, 0.0527873356082551, 0.05165892860963608, 0.05286543300760081, 0.045966028796852416, 0.05519944092659215, 0.05627065035923804, 0.05532901814095673, 0.05569651720505319, 0.044081785707395016]
        cpu_reward_weight_list = self.cpu_reward_pool.get_reward_weight_list()
        io_reward_weight_list = self.io_reward_pool.get_reward_weight_list()
        simple_reward_weight_list = self.simple_reward_pool.get_reward_weight_list()
        # reward_weight_path = f"backup/test-0529/D3QN-OPT/test6/reward_weight.txt"
        # with open(reward_weight_path, 'a+') as f:
        #     f.write(f"cpu_reward_weight_list:{cpu_reward_weight_list}\nio_reward_weight_list:{io_reward_weight_list}\nsimple_reward_weight_list:{simple_reward_weight_list}\n")
        
        machines_id, is_net = self.DRL.choose_action(np.array(states), task_type_str[task_type_id], cpu_reward_weight_list, io_reward_weight_list, simple_reward_weight_list)  # 通过调度算法得到分配 id
        machines_id = machines_id.astype(int).tolist()
        return machines_id, is_net
        # if (step == 1): print_log("machines_id: " + str(machines_id))

    def learn(self, task_instance_batch, machines_id, makespan, is_net):
        # print(self.machine_mips_weight_list)
        # print(self.machine_bandwidth_weight_list)
        # exit()
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
        
        
        reward_save_path = f"backup/test-0529/D3QN-OPT/test6/reward.txt"
        action_save_path = f"backup/test-0529/D3QN-OPT/test6/action.txt"
        info_save_path = f"backup/test-0529/D3QN-OPT/test6/info.txt"
        data_save_path = f"backup/test-0529/D3QN-OPT/test6/task_type_data.txt"
        
        states = get_state(task_instance_batch, self.machine_list, self.cur_weight_list)

        batch_reward_list = []
        batch_remain_num = len(task_instance_batch) if is_net == False else 5
        batch_record_num = 5
        
        # 一些函数要在for循环外计算，提高性能
        # machine working time remained
        commit_time = task_instance_batch[0].commit_time
        total_time = 0
        for machine in self.machine_list:
            total_time += max(machine.get_execution_finish_time() - commit_time, 0)
        for i, machine in enumerate(self.machine_list):
            self.machine_remain_time_weight_list[i] = max(machine.get_execution_finish_time() - commit_time, 0) / total_time
        
        for idx, task in enumerate(task_instance_batch):  # 便历新提交的一批任务，记录动作和奖励
            machine_id = machines_id[idx]
            machine = self.machine_list[machine_id]
            
            # self.machine_assigned_task_list[machine_id] += 1

            # reward = self.C / (self.alpha * math.log(task_item, 10) +
            #                   self.beta * math.log(makespan_item, 10))
            # GoCJ数据集，w=10合适
            # Alibaba数据集，w=1000合适，否则会出现除0错误
            # w = 1000
            w1 = 1000
            w2 = 100000
            # reward = math.log(w) / (math.log(task.get_task_processing_time() * w, 10))
            # reward = math.log(task.get_task_mi() * w) / (self.alpha * math.log(task.get_task_processing_time() * w, 10) +
            #                   self.beta * math.log(m_std * w, 10))
            # reward = math.log(task.get_task_mi() * w) / (self.alpha * math.log(task.get_task_processing_time() * w, 10) +
            #                   self.beta * math.log(makespan * w, 10))
            
            # total_task_num = np.sum(self.machine_assigned_task_list)
            # cur_weight = self.machine_assigned_task_list[machine_id] / total_task_num
            # self.cur_weight_list[machine_id] = cur_weight
            # weight_ratio = cur_weight / self.machine_mips_weight_list[machine_id]
            
            
            # m_wait_time = task.get_task_commit_time() - self.machine_list[machine_id].get_finish_time()
            # m_wait_elem = 0 if m_wait_time <= 10 else math.log(m_wait_time, 10)
            
            reward_scale = 1
            
            # multiple type tasks processing
            # 1. cpu-intensive tasks
            type_id = 0
            if task.get_task_type() == "cpu-intensive":
                type_id = 1
                self.alpha = 1
                # large cpu-intensive tasks
                if task.get_task_executing_time() > 10 * task.get_task_transfer_time():
                    reward_scale = self.machine_mips_weight_list[machine_id] / (1 / self.machine_num)
            # 2. io-intensive tasks
            elif task.get_task_type() == "io-intensive":
                type_id = 2
                self.alpha = 0
                # large io-intensive tasks
                if task.get_task_transfer_time() > 10 * task.get_task_executing_time():
                    reward_scale = self.machine_bandwidth_weight_list[machine_id] / (1 / self.machine_num)
            # 3. simple tasks
            else:
                type_id = 0
                self.alpha = 0.5
                reward_scale = 1
                
            # load balance strategy
            # 两个思路：
            # 1. 减少主力机奖励 
            # 2. 增大非主力机奖励
            # 3. 根据时间进行调整
            # 机器剩余工作时间
            reward_incre_ratio = 1
            # if self.machine_remain_time_weight_list[machine_id] < (1 / self.machine_num):
            #     reward_incre_ratio = min((1 / self.machine_num) / self.machine_remain_time_weight_list[machine_id], 5)
            # reward_scale = reward_scale * reward_incre_ratio
            
            # reward_decre_ratio = 1
            
            # 机器已空闲时间
            # reward_incre_ratio = 1 + (math.log(max(task.get_task_commit_time() - machine.get_execution_finish_time(), 1), 10)) / 10
            # reward_scale = reward_scale * reward_incre_ratio
            
            self.beta = 1 - self.alpha
            
            # reward = (1 / self.machine_num) / self.machine_remain_time_weight_list[machine_id]
            reward = reward_scale * (self.alpha * (math.log((task.get_task_mi() / task.get_task_cpu_utilization()) * w1, 10) / math.log(task.get_task_executing_time() * w1, 10)) + self.beta * (
                math.log(task.get_task_size() * w2, 10) / math.log(task.get_task_transfer_time() * w1, 10)))
            
            # reward = reward_scale * (math.log(task.get_task_mi() * w, 10) / (self.beta * math.log(weight_ratio * w, 10)))
            # reward = reward_scale * (math.log(task.get_task_mi() * w, 10) / (self.alpha * math.log(task.get_task_executing_time() * w, 10)))
            # reward = reward_scale * (math.log(task.get_task_mi() * w, 10) / (self.alpha * math.log(task.get_task_executing_time() * w, 10) +
            #                   self.beta * math.log(weight_ratio * w, 10)))

            batch_reward_list.append(reward)
            
            # 加入replay memory
            if is_net and batch_remain_num > 0:
                if task.get_task_type() == "cpu-intensive":
                    self.cpu_reward_pool.add(machine_id, reward)
                elif task.get_task_type() == "io-intensive":
                    self.io_reward_pool.add(machine_id, reward)
                else:
                    self.simple_reward_pool.add(machine_id, reward)
                self.state_all.append(states[idx])
                self.action_all.append([machine_id])
                self.reward_all.append([reward])
                batch_remain_num -= 1
            else:
                if task.get_task_type() == "cpu-intensive":
                    self.cpu_reward_pool.add(machine_id, reward)
                elif task.get_task_type() == "io-intensive":
                    self.io_reward_pool.add(machine_id, reward)
                else:
                    self.simple_reward_pool.add(machine_id, reward)
                self.state_all.append(states[idx])
                self.action_all.append([machine_id])
                self.reward_all.append([reward])

            # self.state_all.append(states[idx])
            # self.action_all.append([machine_id])
            # self.reward_all.append([reward])
                
            if batch_record_num > 0:
                # with open(action_save_path, 'a+') as f:
                #     f.write(f"{machine_id}\n")
                # with open(info_save_path, 'a+') as f:
                #     f.write(f"{round(reward,3)}\t{self.cur_step}\t{task.get_task_type()}\t{task.get_task_mi() / task.get_task_cpu_utilization()}\t{task.get_task_executing_time()}\t{task.get_task_size()}\t{task.get_task_transfer_time()}\t{reward_scale}\t{self.machine_remain_time_weight_list[machine_id]}\t{reward_incre_ratio}\t{machine.get_execution_finish_time() - task.get_task_commit_time()}\t{machine_id}\n")
                # with open(data_save_path, 'a+') as f:
                #     f.write(f"{task.get_task_mi()}\t{task.get_task_cpu_utilization()}\t{task.get_task_size()}\t{type_id}\n")
                batch_record_num -= 1
        
        # reward_mean = np.mean(np.array(batch_reward_list))
        # with open(reward_save_path, 'a+') as f:
        #     f.write(str(round(reward_mean, 3)) + "\n")
        
        # 减少存储数据量
        if len(self.state_all) > 2 * self.replay_memory_size:
            self.state_all = self.state_all[-self.replay_memory_size:]
            self.action_all = self.action_all[-self.replay_memory_size:]
            self.reward_all = self.reward_all[-self.replay_memory_size:]

        batch_learn_num = len(task_instance_batch) if is_net == False else 1
        if self.prioritized_memory:
            for i in range(batch_learn_num):
                self.DRL.append_sample([self.state_all[-1 - i]], [self.action_all[-1 - i]],
                                  [self.reward_all[-1 - i]], [self.state_all[-1 - i]])
                # with open(mdp_memory_path, 'a+') as f:
                #         f.write(f"second_state:{[self.state_all[i]]}\naction:{[self.action_all[i]]}\nreward:{[self.reward_all[i]]}\nstate_:{[self.state_all[i]]}\n")
        
        # if self.prioritized_memory:
        #     for i in range(len(task_instance_batch)):
        #         self.DRL.append_sample([self.state_all[-2 + i]], [self.action_all[-1 + i]],
        #                           [self.reward_all[-1 + i]], [self.state_all[-1 + i]])

            
        # 先学习一些经验，再学习
        print("cur_step: ", self.cur_step)
        # if self.cur_step >= 0 and len(self.state_all) > 2:
        if self.cur_step > 50:
            # 截取最后10000条记录
            new_state = np.array(self.state_all, dtype=np.float32)[-self.replay_memory_size:-1]
            new_action = np.array(self.action_all, dtype=np.float32)[-self.replay_memory_size:-1]
            new_reward = np.array(self.reward_all, dtype=np.float32)[-self.replay_memory_size:-1]
            self.DRL.store_memory(new_state, new_action, new_reward)
            self.DRL.step = self.cur_step
            loss = self.DRL.learn()
            print_log(f"step: {self.cur_step}, loss: {loss}")
        self.cur_step += 1
