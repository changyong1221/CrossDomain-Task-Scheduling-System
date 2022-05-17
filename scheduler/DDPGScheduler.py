import numpy as np
from scheduler.scheduler import Scheduler
from model.ddpg.ddpg import DDPG
from utils.state_representation import get_ddpg_state
from utils.log import print_log


class DDPGScheduler(Scheduler):
    def __init__(self, machine_num, task_batch_num):
        """Initialization
        """
        self.task_dim = 3
        self.machine_dim = 2

        self.state_all = []  # 存储所有的状态 [None,2+2*20]
        self.action_all = []  # 存储所有的动作 [None,1]
        self.reward_all = []  # 存储所有的奖励 [None,1]

        self.DRL = DDPG(task_batch_num, self.task_dim, machine_num, self.machine_dim)
        self.DRL.max_step = task_batch_num
        self.cur_step = 0
        print_log("DDPG网络初始化成功！")

    def schedule(self, task_instance_batch, machine_list):
        states = get_ddpg_state(task_instance_batch, machine_list)
        self.state_all += states
        # self.state_all.append(states)
        machines_id = self.DRL.choose_action(np.array(states))  # 通过调度算法得到分配 id
        machines_id = machines_id.astype(int).tolist()
        return machines_id

    def learn(self, task_instance_batch, machines_id):
        reward_save_path = f"backup/test-0508/DDPG/test/reward.txt"
        action_save_path = f"backup/test-0508/DDPG/test/action.txt"
        for idx, task in enumerate(task_instance_batch):  # 便历新提交的一批任务，记录动作和奖励
            self.action_all.append([machines_id[idx]])
            with open(action_save_path, 'a+') as f:
                f.write(f"{task.get_task_mi()}\t{machines_id[idx]}\n")
            
            # reward = task.get_task_mi() / task.get_task_processing_time() / 100
            reward = task.get_task_mi() / task.get_task_processing_time() / 100
            self.reward_all.append([reward])  # 计算奖励
            with open(reward_save_path, 'a+') as f:
                f.write(str(round(reward, 3)) + "\n")
        
        # 减少存储数据量
        if len(self.state_all) > 20000:
            self.state_all = self.state_all[-10000:]
            self.action_all = self.action_all[-10000:]
            self.reward_all = self.reward_all[-10000:]

        # 先学习一些经验，再学习
        print("cur_step: ", self.cur_step)
        if self.cur_step > 200:
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
            # 将loss保存到文件中
            loss_save_path = f"backup/test-0508/DDPG/test/error_ddpg.txt"
            with open(loss_save_path, 'a+') as f:
                f.write(str(np.round(loss, 3)) + "\n")
            print_log(f"step: {self.cur_step}, loss: {loss}")
        self.cur_step += 1
