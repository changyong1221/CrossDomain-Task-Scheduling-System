from scheduler.scheduler import Scheduler


class RandomScheduler(Scheduler):
    def __init__(self, machine_num):
        """Initialization
        """
        self.machine_num = machine_num
        self.cur_machine_id = 0

    def schedule(self, task_num):
        """Schedule tasks in random way

        @:return scheduling results, which is a list of machine id
        """
        pass
