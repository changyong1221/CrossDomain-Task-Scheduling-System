import os
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from model.dqn.prioritized_memory import Memory
from utils.file_check import check_and_build_dir
from utils.log import print_log
from utils.state_representation import balancing, get_machine_kind_idx, task_adapting
import globals.global_var as glo


class DQN(object):
    # 每次把一个任务分配给一个虚拟机
    def __init__(self, multidomain_id, task_dim, vms, vm_dim, machine_kind_num_list, machine_kind_idx_range_list,
                 double_dqn=False, dueling_dqn=False, use_prioritized_memory=False, is_federated=False, is_test=False, epsilon_decay=0.998, prob=0.5, balance_prob=0.5, machine_mips_weight_list=None, machine_bandwidth_weight_list=None):
        self.multidomain_id = multidomain_id
        self.task_dim = task_dim  # 任务维度
        self.vms = vms  # 虚拟机数量
        self.vms_num = self.vms
        self.vm_dim = vm_dim  # 虚拟机维度

        self.s_task_dim = self.task_dim  # 任务状态维度
        self.s_vm_dim = self.vms * self.vm_dim  # 虚拟机状态维度
        self.a_dim = self.vms  # 动作空间：虚拟机的个数

        self.double_dqn = double_dqn  # 是否使用double dqn，默认为False
        self.dueling_dqn = dueling_dqn  # 是否使用dueling dqn，默认为False
        self.use_prioritized_memory = use_prioritized_memory  # 是否使用prioritized memory，默认为False

        # prioritized memory replay专用参数
        self.memory_size = 10000  # 经验池大小
        self.memory = Memory(self.memory_size)

        self.lr = 0.003  # learning rate
        self.batch_size = 32  # 128
        self.epsilon = 0.95   # epsilon初始值
        # self.epsilon = 0.1
        # self.epsilon = 1.0
        # self.epsilon = 0
        # self.epsilon_decay = 0.998  # epsilon退化率
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.1      # epsilon最小值
        self.step = 0
        self.max_step = 0
        self.target_prob = 1.0      # 学习目标算法的概率，若目标算法为fine grain，则有50%的几率在随机选择动作时以fine grain策略选择动作
        self.prob = prob
        self.is_test = is_test
        self.GAMMA = 0.3  # reward discount，惩罚项
        

        # 均衡策略
        # self.machine_weight_list = machine_weight_list
        self.machine_mips_weight_list = machine_mips_weight_list
        self.machine_bandwidth_weight_list = machine_bandwidth_weight_list
        self.machine_weight_list = [((self.machine_mips_weight_list[i] + self.machine_bandwidth_weight_list[i]) * 0.5) for i in range(len(self.machine_mips_weight_list))]
        self.accu_machine_weight_list = [np.sum(self.machine_weight_list[:i+1]) for i in range(len(self.machine_weight_list))]
        self.accu_machine_mips_weight_list = [np.sum(self.machine_mips_weight_list[:i+1]) for i in range(len(self.machine_mips_weight_list))]
        self.accu_machine_bandwidth_weight_list = [np.sum(self.machine_bandwidth_weight_list[:i+1]) for i in range(len(self.machine_bandwidth_weight_list))]
        self.machine_kind_num_list = machine_kind_num_list                  # 记录每类机器的数目
        self.machine_kind_idx_range_list = machine_kind_idx_range_list      # 记录每类机器的索引范围
        self.num_of_machine_kind = len(self.machine_kind_num_list)          # 按性能分类的机器种类数目
        self.machine_task_map = [0 for i in range(self.a_dim)]              # 记录每台机器上分配的任务数
        self.machine_kind_task_map = [0 for i in range(self.num_of_machine_kind)]   # 记录每类机器上分配的任务数
        self.machine_kind_avg_task_map = [0 for i in range(self.num_of_machine_kind)]  # 记录每类机器上的平均分配任务数
        self.balance_factor_max = 0.9       # 阶梯平衡因子，前一级阶梯与后一级阶梯的比值应介于平衡因子[min, max]之间
        self.balance_factor_min = 0.85
        self.task_affinity_factor = 0.9    # 任务亲和度，针对大任务优化，大任务对高性能机器具有更高的亲和度，该优化的优先级高于负载因子
        self.balance_prob = balance_prob     # 小于balance_prob使用任务亲和度优化，大于balance_prob使用负载均衡优化

        if self.dueling_dqn:
            self.TARGET_REPLACE_ITER = 5  # target update frequency，每过多少轮更新TargetNet
            self.eval_net = Dueling_DQN(self.s_task_dim, self.s_vm_dim, self.a_dim)

            # 打印网络结构参数
            # device = torch.device('cpu')
            # net_graph = self.eval_net.to(device)
            # summary(net_graph)

            self.eval_net.apply(self.weights_init)

            model_file_path = f"save/client-0/eval.pth"
            if os.path.exists(model_file_path):
                weights = torch.load(model_file_path)
                self.eval_net.load_state_dict(weights, strict=True)
                print("load model finished.")
            
            # model_file_path = f"save/global/global.pth"
            # if is_federated and os.path.exists(model_file_path):
            #     weights = torch.load(model_file_path)
            #     self.eval_net.load_state_dict(weights, strict=True)
            #     print("load model finished.")

            self.target_net = Dueling_DQN(self.s_task_dim, self.s_vm_dim, self.a_dim)
            self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

            self.hard_update(self.target_net, self.eval_net)  # 初始化为相同权重

            self.loss_f = nn.MSELoss()
        else:
            self.TARGET_REPLACE_ITER = 1  # target update frequency，每过多少轮更新TargetNet
            self.eval_net = QNet_v1(self.s_task_dim, self.s_vm_dim, self.a_dim)

            # 打印网络结构参数
            # device = torch.device('cpu')
            # net_graph = self.eval_net.to(device)
            # summary(net_graph)

            self.eval_net.apply(self.weights_init)
            
            model_file_path = f"save/client-0/eval.pth"
            if os.path.exists(model_file_path):
                weights = torch.load(model_file_path)
                self.eval_net.load_state_dict(weights, strict=True)
                print("load model finished.")

            # model_file_path = f"save/global/global.pth"
            # if is_federated and os.path.exists(model_file_path):
            #     weights = torch.load(model_file_path)
            #     self.eval_net.load_state_dict(weights, strict=True)

            self.target_net = QNet_v1(self.s_task_dim, self.s_vm_dim, self.a_dim)
            self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

            self.hard_update(self.target_net, self.eval_net)  # 初始化为相同权重

            self.loss_f = nn.MSELoss()

        # try:
        #     shutil.rmtree('DQN/logs/')  # 递归删除文件夹
        # except:
        #     print_log("没有发现logs文件目录")
        # self.writer = SummaryWriter("DQN/logs/")
        # dummy_input = Variable(torch.rand(5, self.s_task_dim+self.s_vm_dim))
        # with SummaryWriter(logdir="DQN/logs/graph", comment="Q_net") as w:
        #     w.add_graph(self.eval_net, (dummy_input))

    # 保存一个初始模型
    def save_initial_model(self, output_path):
        torch.save(self.eval_net.state_dict(), output_path)

    # 多个状态传入，给每个状态选择一个动作
    def choose_action(self, s_list, task_type, cpu_reward_weight_list, io_reward_weight_list, simple_reward_weight_list):
        accu_cpu_reward_weight_list = [np.sum(cpu_reward_weight_list[:i+1]) for i in range(self.vms_num)]
        accu_io_reward_weight_list = [np.sum(io_reward_weight_list[:i+1]) for i in range(self.vms_num)]
        accu_simple_reward_weight_list = [np.sum(simple_reward_weight_list[:i+1]) for i in range(self.vms_num)]
        
        is_net = False
        if self.epsilon > self.epsilon_min:  # epsilon最小值
            self.epsilon *= self.epsilon_decay
        print("epsilon: ", self.epsilon)
        action_prob_path = "backup/test-0529/D3QN-OPT/test6/action_prob.txt"
        if np.random.uniform() > self.epsilon:  # np.random.uniform()输出0到1之间的一个随机数
            is_net = True
            self.eval_net.eval()    # 进入evaluate模式，区别于train模式，在eval模式，框架会自动把BatchNormalize和Dropout固定住，不会取平均，而是用训练好的值
            actions_value = self.eval_net(torch.from_numpy(s_list).float())
            # print(f"actions_value:{actions_value}")
            
            # 原始方式，直接根据最大值选择动作
            actions = torch.max(actions_value, 1)[1].data.numpy()
            # with open(action_prob_path, 'a+') as f:
            #     f.write(f"actions_value:{actions_value}\nactions:{actions}\n")
            # print("net choose!")
        else:
            # 加权负载均衡方法
            # action_list = [0 for i in range(len(s_list))]
            # for i in range(len(s_list)):
            #     rand_prob = np.random.uniform()
            #     for j, prob in enumerate(self.accu_machine_weight_list):
            #         if rand_prob <= prob:
            #             action_list[i] = j
            #             break
            
            # 阶梯平衡因子负载均衡方法
            # action_list = [0 for i in range(len(s_list))]
            # for i, action in enumerate(action_list):
            #     action = balancing(self.machine_kind_avg_task_map, self.num_of_machine_kind,
            #                       self.machine_kind_idx_range_list, self.balance_factor_min,
            #                       self.balance_factor_max)
            #     action_list[i] = action
            #     self.machine_task_map[action] += 1
            #     kind_idx = get_machine_kind_idx(action, self.num_of_machine_kind, self.machine_kind_idx_range_list)
            #     self.machine_kind_task_map[kind_idx] += 1
            #     self.machine_kind_avg_task_map[kind_idx] = self.machine_kind_task_map[kind_idx] / \
            #                                               self.machine_kind_num_list[kind_idx]
            
            # 任务亲和度和阶梯平衡因子结合的方法，因为任务亲和度需要事先指定任务大小，已被废弃
            # action_list = [0 for i in range(len(s_list))]
            # for i, action in enumerate(action_list):
            #     if np.random.random() < self.balance_prob:
            #         if np.random.random() < self.task_affinity_factor:
            #             action = task_adapting(s_list[i], self.num_of_machine_kind, self.machine_kind_idx_range_list)
            #         else:
            #             action = np.random.randint(self.vms_num)
            #     else:
            #         action = balancing(self.machine_kind_avg_task_map, self.num_of_machine_kind,
            #                           self.machine_kind_idx_range_list, self.balance_factor_min,
            #                           self.balance_factor_max)
            #     action_list[i] = action
            #     self.machine_task_map[action] += 1
            #     kind_idx = get_machine_kind_idx(action, self.num_of_machine_kind, self.machine_kind_idx_range_list)
            #     self.machine_kind_task_map[kind_idx] += 1
            #     self.machine_kind_avg_task_map[kind_idx] = self.machine_kind_task_map[kind_idx] / \
            #                                               self.machine_kind_num_list[kind_idx]
            # print("balance choose!")
            
            # 随机分配法
            action_list = [0 for i in range(len(s_list))]
            for i, action in enumerate(action_list):
                action_list[i] = np.random.randint(self.vms_num)
            # print("random choose!")
            
            # 随机分配和阶梯平衡因子结合的方法
            # if self.step < 300:
            #     action_list = [0 for i in range(len(s_list))]
            #     for i, action in enumerate(action_list):
            #         action_list[i] = np.random.randint(self.vms_num)
            #     print("random choose!")
            # else:
            #     if np.random.random() > self.balance_prob:
            #         action_list = [0 for i in range(len(s_list))]
            #         for i, action in enumerate(action_list):
            #             action_list[i] = np.random.randint(self.vms_num)
            #         print("random choose!")
            #     else:
            #         action_list = [0 for i in range(len(s_list))]
            #         for i, action in enumerate(action_list):
            #             action = balancing(self.machine_kind_avg_task_map, self.num_of_machine_kind,
            #                               self.machine_kind_idx_range_list, self.balance_factor_min,
            #                               self.balance_factor_max)
            #             action_list[i] = action
            #             self.machine_task_map[action] += 1
            #             kind_idx = get_machine_kind_idx(action, self.num_of_machine_kind, self.machine_kind_idx_range_list)
            #             self.machine_kind_task_map[kind_idx] += 1
            #             self.machine_kind_avg_task_map[kind_idx] = self.machine_kind_task_map[kind_idx] / \
            #                                                       self.machine_kind_num_list[kind_idx]
            #         print("balance choose!")

            # if self.step < 1000:
            #     action_list = [0 for i in range(len(s_list))]
            #     for i, action in enumerate(action_list):
            #         action_list[i] = np.random.randint(self.vms_num)
            #     print("random choose!")
            # else:
            #     action_list = [0 for i in range(len(s_list))]
            #     for i, action in enumerate(action_list):
            #         action = balancing(self.machine_kind_avg_task_map, self.num_of_machine_kind,
            #                           self.machine_kind_idx_range_list, self.balance_factor_min,
            #                           self.balance_factor_max)
            #         action_list[i] = action
            #         self.machine_task_map[action] += 1
            #         kind_idx = get_machine_kind_idx(action, self.num_of_machine_kind, self.machine_kind_idx_range_list)
            #         self.machine_kind_task_map[kind_idx] += 1
            #         self.machine_kind_avg_task_map[kind_idx] = self.machine_kind_task_map[kind_idx] / \
            #                                                   self.machine_kind_num_list[kind_idx]
            #     print("balance choose!")
            actions = np.array(action_list)
        
        
        # 对动作进行负载均衡
        if self.is_test or (np.random.uniform() > self.epsilon):
            adict = {}  # 标记每个VM出现次数，实现负载均衡
            s_task_num = len(s_list)
            # print(f"s_task_num: {s_task_num}")
    
            # 后面的代码增加分配VM的负载均衡，也是对动作的探索，融入了加权随机算法
            # print(f"before: {actions}")
            for i in range(s_task_num):
                if actions[i] not in adict:
                    adict[actions[i]] = 1
                else:
                    adict[actions[i]] += 1
            
            # 按照预测任务类型进行动作的负载均衡
            if task_type == 'simple':
                for i in range(s_task_num):
                    # 按后面的概率随机给任务分配VM
                    if adict[actions[i]] > int(s_task_num * simple_reward_weight_list[actions[i]]):
                        adict[actions[i]] -= 1
                        rand_prob = np.random.uniform()
                        for j, prob in enumerate(accu_simple_reward_weight_list):
                            if rand_prob <= prob:
                                actions[i] = j
                                break
            elif task_type == 'cpu-intensive':
                for i in range(s_task_num):
                    # 按后面的概率随机给任务分配VM
                    if adict[actions[i]] > int(s_task_num * cpu_reward_weight_list[actions[i]]):
                        adict[actions[i]] -= 1
                        rand_prob = np.random.uniform()
                        for j, prob in enumerate(accu_cpu_reward_weight_list):
                            if rand_prob <= prob:
                                actions[i] = j
                                break
            else:
                for i in range(s_task_num):
                    # 按后面的概率随机给任务分配VM
                    if adict[actions[i]] > int(s_task_num * io_reward_weight_list[actions[i]]):
                        adict[actions[i]] -= 1
                        rand_prob = np.random.uniform()
                        for j, prob in enumerate(accu_io_reward_weight_list):
                            if rand_prob <= prob:
                                actions[i] = j
                                break
        print("actions: ", actions)
        return actions, is_net


    # Prioritized DQN专用函数
    # save sample (error,<s,a,r,s'>) to the replay memory
    def append_sample(self, state, action, reward, next_state):
        # self.eval_net.train()
        # target = self.eval_net(torch.from_numpy(np.array(state)).float()).detach()
        # old_val = target[0][torch.LongTensor(action)]
        # self.target_net.eval()
        # target_val = self.target_net(torch.from_numpy(np.array(next_state)).float()).detach()
        # new_val = torch.FloatTensor(reward)[0] + GAMMA * torch.max(target_val)
        # error = abs(old_val - new_val)
        
        # print(state)
        # print(action)
        # print(reward)
        # print(next_state)
        # exit()
        self.memory.add(reward, (state, action, reward, next_state))


    def learn(self):
        # 更新 Target Net
        if self.step % self.TARGET_REPLACE_ITER == 0:
            self.hard_update(self.target_net, self.eval_net)

        # Prioritized DQN单独处理
        if self.use_prioritized_memory:
            mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)

            mini_batch = np.array(mini_batch, dtype=object).transpose()
            states = mini_batch[0][0].tolist()
            next_states = mini_batch[0][3].tolist()
            actions = mini_batch[0][1].tolist()
            rewards = mini_batch[0][2].tolist()

            self.bstate = torch.from_numpy(np.array(states, dtype=np.float32)).float()
            self.bstate_ = torch.from_numpy(np.array(next_states, dtype=np.float32)).float()
            self.baction = torch.LongTensor(np.array(actions, dtype=np.float32))
            self.breward = torch.FloatTensor(np.array(rewards, dtype=np.float32))  # 奖励值值越大越好
            # print(f"bstate:{self.bstate}")
            # print(f"bstate_:{self.bstate_}")
            # print(f"baction:{self.baction}")
            # print(f"breward:{self.breward}")
            # mdp_save_path = "backup/test-0529/D3QN-OPT/test6/mdp.txt"
            # with open(mdp_save_path, 'a+') as f:
            #     f.write(f"bstate:{self.bstate}\nbstate_:{self.bstate_}\nbaction:{self.baction}\nbreward:{self.breward}\n")
            # exit()

            # 训练Q网络
            self.eval_net.train()
            # Q预测值
            q_eval = self.eval_net(self.bstate).gather(1, self.baction)  # shape (batch, 1), gather表示获取每个维度action为下标的Q值
            # print_log("q_eval: ", q_eval)

            self.target_net.train()
            q_next = self.target_net(self.bstate_).detach()  # 设置 Target Net 不需要梯度

            # 先用Q_eval即最新的神经网络估计Q_next即Q现实中的Q(S',a')中的最大动作值所对应的索引
            self.eval_net.eval()
            q_eval_next = self.eval_net(self.bstate_).detach()
            q_eval_action = q_eval_next.max(1)[1].view(self.batch_size, 1)

            # 然后用这个被Q_eval估计出来的动作来选择Q现实中的Q(s')
            q_target_prime = q_next.gather(1, q_eval_action)
            q_target = self.breward + self.GAMMA * q_target_prime

            # update priority，不需要更新priority
            # errors = torch.abs(q_eval - q_target).data.numpy()
            # for i in range(self.batch_size):
            #     idx = idxs[i]
            #     self.memory.update(idx, errors[i])

            loss = (torch.FloatTensor(is_weights) * F.mse_loss(q_eval, q_target)).mean()

            # 将梯度初始化为零
            self.optimizer.zero_grad()
            # 反向传播求梯度
            loss.backward()
            # 更新所有参数
            self.optimizer.step()
            
            # 将Q值保存到文件中
            # q_value_save_path = f"backup/test-0529/D3QN-OPT/test6/q_target_prime.txt"
            # with open(q_value_save_path, 'a+') as f:
            #     q_target_list = q_target_prime.tolist()
            #     q_value = np.mean(q_target_list)
            #     f.write(str(round(q_value, 3)) + "\n")
                
            # 将breward值保存到文件中
            # breward_save_path = f"backup/test-0529/D3QN-OPT/test6/breward.txt"
            # with open(breward_save_path, 'a+') as f:
            #     breward_list = self.breward.tolist()
            #     breward = np.mean(breward_list)
            #     f.write(str(round(breward, 3)) + "\n")
            
            # 将Q值保存到文件中
            # q_value_save_path = f"backup/test-0529/D3QN-OPT/test6/q_value.txt"
            # with open(q_value_save_path, 'a+') as f:
            #     q_target_list = q_target.tolist()
            #     q_value = np.mean(q_target_list)
            #     f.write(str(round(q_value, 3)) + "\n")

            # 将loss保存到文件中
            # loss_save_path = f"backup/test-0529/D3QN-OPT/test6/loss.txt"
            # with open(loss_save_path, 'a+') as f:
            #     f.write(str(round(loss.item(), 3)) + "\n")
        else:
            # 训练Q网络
            self.eval_net.train()
            # Q预测值
            q_eval = self.eval_net(self.bstate).gather(1, self.baction)  # shape (batch, 1), gather表示获取每个维度action为下标的Q值
            # print_log("self.eval_net(self.bstate): ", self.eval_net(self.bstate))
            # print_log("self.eval_net(self.bstate) dim: ", self.eval_net(self.bstate).size())
            # idx = self.baction[0].tolist()[0]
            # bs_list = self.eval_net(self.bstate).tolist()
            # val = bs_list[0][idx]

            self.target_net.train()
            q_next = self.target_net(self.bstate_).detach()  # 设置 Target Net 不需要梯度
            # print_log("q_next: ", q_next)

            # Q现实值
            # Tensor.view()返回的新tensor与原先的tensor共用一个内存,只是将原tensor中数据按照view(M,N)中，M行N列显示出来
            # Tensor.max(1)表示返回每一行中最大值的那个元素，且返回其索引
            # Tensor.max(1)[0]表示只返回每一行中最大值的那个元素
            # Tensor.unsqueeze(dim)用来扩展维度，在指定位置加上维数为1的维度，dim可以取0,1,...或者负数，这里的维度和pandas的维度是一致的，0代表行扩展，1代表列扩展
            # Tensor.squeeze()则用来对维度进行压缩，去掉所有维数为1（比如1行或1列这种）的维度，不为1的维度不受影响
            # Tensor.expand()将单个维度扩展成更大的维度，返回一个新的tensor

            # 将Double DQN和Vanilla DQN两种方式的Q值保存到文件中，然后画图比较
            # output_path = "result/q_value/" + ("double_dqn_q_value.txt" if self.double_dqn else "dqn_q_value.txt")

            # 如果采用Double DQN的方式
            if self.double_dqn:
                # 先用Q_eval即最新的神经网络估计Q_next即Q现实中的Q(S',a')中的最大动作值所对应的索引
                self.eval_net.eval()
                q_eval_next = self.eval_net(self.bstate_).detach()
                q_eval_action = q_eval_next.max(1)[1].view(self.batch_size, 1)

                # 然后用这个被Q_eval估计出来的动作来选择Q现实中的Q(s')
                q_target_prime = q_next.gather(1, q_eval_action)
                q_target = self.breward + self.GAMMA * q_target_prime
                loss = self.loss_f(q_eval, q_target)

            else:
                # q_next.max(1)[0].view(self.batch_size, 1)是一个32行1列的向量
                q_target = self.breward + self.GAMMA * q_next.max(1)[0].view(self.batch_size, 1)  # shape (batch, 1)

                loss = self.loss_f(q_eval, q_target)

            # 将梯度初始化为零
            self.optimizer.zero_grad()
            # 反向传播求梯度
            loss.backward()
            # 更新所有参数
            self.optimizer.step()
            
            # 将Q值保存到文件中
            # q_value_save_path = f"backup/test-0529/D3QN-OPT/test6/q_target_prime.txt"
            # with open(q_value_save_path, 'a+') as f:
            #     q_target_list = q_next.max(1)[0].view(self.batch_size, 1).tolist()
            #     q_value = np.mean(q_target_list)
            #     f.write(str(round(q_value, 3)) + "\n")
                
            # 将breward值保存到文件中
            # breward_save_path = f"backup/test-0529/D3QN-OPT/test6/breward.txt"
            # with open(breward_save_path, 'a+') as f:
            #     breward_list = self.breward.tolist()
            #     breward = np.mean(breward_list)
            #     f.write(str(round(breward, 3)) + "\n")
            
            # 将Q值保存到文件中
            # q_value_save_path = f"backup/test-0529/D3QN-OPT/test6/q_value.txt"
            # with open(q_value_save_path, 'a+') as f:
            #     q_target_list = q_target.tolist()
            #     q_value = np.mean(q_target_list)
            #     f.write(str(round(q_value, 3)) + "\n")

            # 将loss保存到文件中
            # loss_save_path = f"backup/test-0529/D3QN-OPT/test6/loss.txt"
            # with open(loss_save_path, 'a+') as f:
            #     f.write(str(round(loss.item(), 3)) + "\n")

        # 保存模型参数
        if self.step == self.max_step - 1:
            model_save_dir = f"save/client-{self.multidomain_id}"
            check_and_build_dir(model_save_dir)
            model_save_path = f"save/client-{self.multidomain_id}/eval.pth"
            torch.save(self.eval_net.state_dict(), model_save_path)
            # torch.save(self.target_net.state_dict(), "save/target.pth")
            print_log("parameters of evaluate net have been saved.")

        # 画图
        # if self.step % 10 == 0:
        #     self.writer.add_scalar('Q-value', q_eval.detach().numpy()[0], self.step)
        #     self.writer.add_scalar('Loss', loss.detach().numpy(), self.step)

        return loss.detach().numpy()

    def store_memory(self, state_all, action_all, reward_all):
        indexs = np.random.choice(len(state_all[:-1]), size=self.batch_size)

        self.bstate = torch.from_numpy(state_all[indexs, :]).float()
        self.bstate_ = torch.from_numpy(state_all[indexs + 1, :]).float()
        self.baction = torch.LongTensor(action_all[indexs, :])
        self.breward = torch.from_numpy(reward_all[indexs, :]).float()  # 奖励值值越大越好

    # 全部更新
    def hard_update(self, target_net, eval_net):
        target_net.load_state_dict(eval_net.state_dict())

    # 初始化网络参数
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):  # 批归一化层初始化
            nn.init.uniform_(m.bias)  # 初始化为U(0,1)
            nn.init.constant_(m.bias, 0)


class QNet_v1(nn.Module):  # 通过 s 预测出 a
    def __init__(self, s_task_dim, s_vm_dim, a_dim):
        super(QNet_v1, self).__init__()
        self.s_task_dim = s_task_dim
        self.s_vm_dim = s_vm_dim
        self.layer1_task = nn.Sequential(  # 处理任务状态
            nn.Linear(self.s_task_dim, 16),     # 全连接层，相当于tf.layers.dense，输入维度为3，输出维度为16，即16列
            torch.nn.Dropout(0.2),              # Dropout层
            nn.BatchNorm1d(16),                 # 归一化层，参数为维度
            nn.LeakyReLU(),                     # 激活函数
        )
        self.layer1_1vm = nn.Sequential(  # 处理虚拟机状态
            nn.Linear(self.s_vm_dim, 32),
            torch.nn.Dropout(0.2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )
        self.layer1_2vm = nn.Sequential(
            nn.Linear(32, 16),
            torch.nn.Dropout(0.2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(  # 融合处理结果
            nn.Linear(32, 16),
            torch.nn.Dropout(0.2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, a_dim)        # 输出为动作的维度
        )

    def forward(self, x):
        # print_log("type of x: ", type(x))
        # print_log("dim of x: ", x.size())
        # print_log("x[:1]: ", x[:1])
        x1 = self.layer1_task(x[:, :self.s_task_dim])  # 任务
        x2 = self.layer1_1vm(x[:, self.s_task_dim:])  # 虚拟机
        x2 = self.layer1_2vm(x2)
        x = torch.cat((x1, x2), dim=1)      # x1和x2以dim=1拼接，即横着拼，两个16列变成32列
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Dueling_DQN(nn.Module):
    def __init__(self, s_task_dim, s_vm_dim, a_dim):
        super(Dueling_DQN, self).__init__()
        self.s_task_dim = s_task_dim
        self.action_dim = a_dim
        self.s_vm_dim = s_vm_dim
        self.layer1_task = nn.Sequential(  # 处理任务状态
            nn.Linear(self.s_task_dim, 32),  # 全连接层，相当于tf.layers.dense，输入维度为3，输出维度为16，即16列
            torch.nn.Dropout(0.2),  # Dropout层
            nn.BatchNorm1d(32),  # 归一化层，参数为维度
            # nn.Tanh(),
            nn.LeakyReLU(),  # 激活函数
        )
        self.layer1_1vm = nn.Sequential(  # 处理虚拟机状态
            nn.Linear(self.s_vm_dim, 64),
            torch.nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            # nn.Tanh(),
            nn.LeakyReLU(),
        )
        self.layer1_2vm = nn.Sequential(
            nn.Linear(64, 32),
            torch.nn.Dropout(0.2),
            nn.BatchNorm1d(32),
            # nn.Tanh(),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(  # 融合处理结果
            nn.Linear(64, 32),
            torch.nn.Dropout(0.2),
            nn.BatchNorm1d(32),
            # nn.Tanh(),
            nn.LeakyReLU(),
        )
        # Dueling DQN中不使用layer3
        # self.layer3 = nn.Sequential(
        #     nn.Linear(32, a_dim)        # 输出为动作的维度
        # )
        # Dueling DQN中 Q = V + A
        self.fc_advantage = nn.Sequential(
            nn.Linear(32, a_dim)  # 输出为动作的维度
        )
        self.fc_value = nn.Sequential(
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # print("type of x: ", type(x))
        # print("dim of x: ", x.size())
        x1 = self.layer1_task(x[:, :self.s_task_dim])  # 任务
        x2 = self.layer1_1vm(x[:, self.s_task_dim:])  # 虚拟机
        x2 = self.layer1_2vm(x2)
        x = torch.cat((x1, x2), dim=1)      # x1和x2以dim=1拼接，即横着拼，两个16列变成32列
        x = self.layer2(x)
        # x = self.layer3(x)
        advantage = self.fc_advantage(x)
        value = self.fc_value(x).expand(x.size(0), self.action_dim)

        x = advantage + value - advantage.mean(1).unsqueeze(1).expand(x.size(0), self.action_dim)
        return x


#
# class Dueling_DQN(nn.Module):
#     def __init__(self, s_dim, a_dim):
#         super(Dueling_DQN, self).__init__()
#         self.state_dim = s_dim
#         self.action_dim = a_dim
#
#         self.layer1 = nn.Sequential(
#             nn.Linear(s_dim, 64),
#             torch.nn.Dropout(0.2),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(),
#         )
#         self.layer2 = nn.Sequential(
#             nn.Linear(64, 128),
#             torch.nn.Dropout(0.2),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(),
#         )
#         # Dueling DQN中 Q = V + A
#         self.fc_advantage = nn.Sequential(
#             nn.Linear(128, self.action_dim)
#         )
#         self.fc_value = nn.Sequential(
#             nn.Linear(128, 1)
#         )
#
#     def forward(self, x):
#         x = torch.Tensor(x)
#         x_dim = len(x.size())
#         if x_dim == 1:
#             x = x.reshape(1, self.state_dim)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         advantage = self.fc_advantage(x)
#         value = self.fc_value(x).expand(x.size(0), self.action_dim)
#
#         x = advantage + value - advantage.mean(1).unsqueeze(1).expand(x.size(0), self.action_dim)
#         return x
