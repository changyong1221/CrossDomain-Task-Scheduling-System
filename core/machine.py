

class Machine(object):
    def __init__(self, machine_id, mips, memory, bandwidth):
        """Initialization
        """
        self.machine_id = machine_id
        self.ip_address = None      # 跟 flask客户端对应
        self.port = None            # 跟 flask客户端对应
        self.mips = mips
        self.memory = memory
        self.bandwidth = bandwidth
        self.realtime_cpu_utilization = 0
        self.realtime_memory_utilization = 0
        self.realtime_bandwidth_utilization = 0
        self.finish_time = 0
        self.work_time = 0
        self.task_waiting_queue = []

    def add_task(self, task):
        """Add task run instance to task_waiting_queue
        """
        self.task_waiting_queue.append(task)
        print(f"task({task.task_id}) ---> machine({self.machine_id})")

    def execute_tasks(self):
        """Execute tasks in the task_waiting_queue
        """
        for task in self.task_waiting_queue:
            task.run_on_machine(self)
            self.work_time += task.get_task_processing_time()
            self.realtime_cpu_utilization = task.get_task_cpu_utilization()
            self.realtime_memory_utilization = task.get_task_size() / self.memory
            self.realtime_bandwidth_utilization = 1
        self.task_waiting_queue.clear()

    def reset(self):
        """Reset machine to initial state
        """
        self.realtime_cpu_utilization = 0
        self.realtime_memory_utilization = 0
        self.realtime_bandwidth_utilization = 0
        self.finish_time = 0
        self.task_waiting_queue.clear()

    def get_machine_id(self):
        """Return machine_id
        """
        return self.machine_id

    def get_mips(self):
        """Return mips
        """
        return self.mips

    def get_memory(self):
        """Return memory
        """
        return self.memory

    def get_bandwidth(self):
        """Return bandwidth
        """
        return self.bandwidth

    def get_finish_time(self):
        """Return finish_time
        """
        return self.finish_time

    def set_finish_time(self, new_finish_time):
        """Set a new finish time
        """
        self.finish_time = new_finish_time
