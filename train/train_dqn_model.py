from scheduler.DQNScheduler import get_state


def train(taskDim, vmDim, cluster, filepath_input, filepath_output):
    # 虚拟机的数目
    vmsNum = len(cluster.machines)
    print("vmsNum: %d" % vmsNum)

    # 从输入文件路径读取所有的任务, all_batch_tasks的长度为
    all_batch_tasks = FileIo(filepath_input).readAllBatchLines()
    all_batch_tasks = all_batch_tasks[:1000]
    print("tasksNum: %d" % len(all_batch_tasks))
    print("环境创建成功！")

    state_all = []  # 存储所有的状态 [None,2+2*20]
    action_all = []  # 存储所有的动作 [None,1]
    reward_all = []  # 存储所有的奖励 [None,1]

    double_dqn = False
    dueling_dqn = False
    optimized_dqn = False
    prioritized_memory = True
    DRL = DQN(taskDim, vmsNum, vmDim, double_dqn, dueling_dqn, optimized_dqn, prioritized_memory)
    DRL.max_step = len(all_batch_tasks)
    print("网络初始化成功！")

    for step, batch_tasks in enumerate(all_batch_tasks):
        tasks_list = []
        for task in batch_tasks:
            tasks_list.append(Task(task[0], task[1], task[2], task[3]))  # 构建任务

        states = get_state(tasks_list, cluster.machines)
        state_all += states
        machines_id = DRL.choose_action(np.array(states))  # 通过调度算法得到分配 id
        machines_id = machines_id.astype(int).tolist()
        # if (step == 1): print("machines_id: " + str(machines_id))

        cluster.submit_tasks(tasks_list, machines_id)  # 提交任务到集群，并调度到虚拟机进行计算

        for i, task in enumerate(cluster.finished_tasks[-len(tasks_list):]):  # 便历新提交的一批任务，记录动作和奖励
            action_all.append([task.task_machine_id])
            reward = task.mi / task.task_response_time / 100
            reward_all.append([reward])  # 计算奖励

        # 减少存储数据量
        if len(state_all) > 20000:
            state_all = state_all[-10000:]
            action_all = action_all[-10000:]
            reward_all = reward_all[-10000:]

        # 如果使用prioritized memory
        if prioritized_memory:
            for i in range(len(states)):
                DRL.append_sample([state_all[-2 + i]], [action_all[-1 + i]], [reward_all[-1 + i]], [state_all[-1 + i]])

        # 先学习一些经验，再学习
        if step > 100:
            # 截取最后10000条记录
            new_state = np.array(state_all, dtype=np.float32)[-10000:-1]
            new_action = np.array(action_all, dtype=np.float32)[-10000:-1]
            new_reward = np.array(reward_all, dtype=np.float32)[-10000:-1]
            DRL.store_memory(new_state, new_action, new_reward)
            DRL.step = step
            loss = DRL.learn()
            print("step:", step, ", loss:", loss)

    finished_tasks = []
    for task in cluster.finished_tasks:
        finished_tasks.append(task.feature)
    FileIo(filepath_output).twoListToFile(finished_tasks, "w")


if __name__ == '__main__':
    pass