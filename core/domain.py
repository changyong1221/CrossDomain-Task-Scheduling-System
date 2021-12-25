from scheduler.RoundRobinScheduler import RoundRobinScheduler
from utils.get_position import get_position_by_name
import globals.global_var as glo


class Domain(object):
    def __init__(self, domain_id, location, longitude=None, latitude=None, auto_locate=True):
        """Initialization
        """
        self.domain_id = domain_id
        self.location = location        # city name, for example "北京市" "莫斯科"
        self.longitude = longitude      # longitude of geographical position
        self.latitude = latitude        # latitude of geographical position
        self.machine_list = []          # all the machines in this domain
        self.idle_machine_list = []     # idle machine list
        self.cluster_list = []          # node clusters in this domain
        self.scheduler = None

        if (self.longitude is None or self.latitude is None) and auto_locate:
            self.longitude, self.latitude = get_position_by_name(location)
        print(f"domain({self.domain_id}) is created.")

    def add_machine(self, machine):
        """Add machine to domain
        """
        self.machine_list.append(machine)
        print(f"machine({machine.machine_id}) is added to domain({self.domain_id})")

    def delete_machine(self, machine):
        """Delete machine from domain
        """
        for node in self.machine_list:
            if node is machine:
                self.machine_list.remove(node)
                break

    def set_scheduler(self, scheduler):
        """Set task scheduling strategy
        """
        self.scheduler = scheduler

    def commit_tasks(self, task_list):
        """Commit tasks to cluster
        """
        if self.scheduler is None:
            print(f"scheduler of domain({self.cluster_id}) is not set, use RoundRobinScheduler on default.")
            self.set_scheduler(RoundRobinScheduler(len(self.machine_list)))
        if self.scheduler.__class__.__name__ == "RoundRobinScheduler":
            schedule_ret = self.scheduler.schedule(len(task_list))
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.run_tasks()
        elif self.scheduler.__class__.__name__ == "DQNScheduler":
            schedule_ret = self.scheduler.schedule(task_list, self.machine_list)
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.run_tasks()

            self.scheduler.learn(task_list, schedule_ret)

    def run_tasks(self):
        """Run the committed tasks
        """
        for machine in self.machine_list:
            machine.execute_tasks()

    def reset(self):
        """Reset cluster
        """
        for machine in self.machine_list:
            machine.reset()

    def clustering_machines(self, cluster_num):
        """Clustering all the machines into several clusters
        """
        pass

    def get_cluster_list(self):
        """Return cluster list
        """
        return self.cluster_list


class MultiDomain(object):
    def __init__(self, multidomain_id, location):
        """Initialization
        """
        self.multidomain_id = multidomain_id
        self.location = location
        self.longitude = None
        self.latitude = None
        self.domain_list = []       # all the domains in the multi-domain system
        self.cluster_list = []      # all the clusters in the multi-domain system
        self.machine_list = []      # all the machines in the multi-domain system
        self.idle_machine_list = []  # idle machine list
        self.scheduler = None
        self.is_using_clustering_optimization = False
        self.version = "v1.0"
        self.print_version()
        print("multi-domain scheduling system is created.")

    def auto_locate(self):
        """Auto locate the longitude and latitude
        """
        self.longitude, self.latitude = get_position_by_name(self.location)

    def add_domain(self, domain):
        """Add domain to multi-domain system
        """
        self.domain_list.append(domain)
        for cluster in domain.cluster_list:
            self.cluster_list.append(cluster)
        for machine in domain.machine_list:
            self.machine_list.append(machine)
        print(f"domain({domain.domain_id}) is add to multi-domain scheduling system.")

    def delete_domain(self, domain):
        """Delete domain from multi-domain system
        """
        for domain_ in self.domain_list:
            if domain_ is domain:
                self.domain_list.remove(domain_)
                break
        for cluster in domain.cluster_list:
            for cluster_ in self.cluster_list:
                if cluster is cluster_:
                    self.cluster_list.remove(cluster_)
        for machine in domain.machine_list:
            for machine_ in self.machine_list:
                if machine is machine_:
                    self.machine_list.remove(machine_)

    def set_scheduler(self, scheduler):
        """Set task scheduling strategy
        """
        self.scheduler = scheduler
        print(f"{scheduler.__class__.__name__} task scheduler is set for multi-domain scheduling system.")

    def commit_tasks(self, task_list):
        """Commit tasks to multi-domain system
        """
        if self.scheduler is None:
            print(f"scheduler of multidomain is not set, use RoundRobinScheduler on default.")
            self.set_scheduler(RoundRobinScheduler(len(self.machine_list)))
        if self.scheduler.__class__.__name__ == "RoundRobinScheduler":
            schedule_ret = self.scheduler.schedule(len(task_list))
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.run_tasks()
        elif self.scheduler.__class__.__name__ == "DQNScheduler":
            schedule_ret = self.scheduler.schedule(task_list, self.machine_list)
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.run_tasks()

            self.scheduler.learn(task_list, schedule_ret)

    def run_tasks(self):
        """Run the committed tasks
        """
        print("committed tasks is running...")
        for machine in self.machine_list:
            machine.execute_tasks()

    def reset(self):
        """Reset cluster
        """
        for machine in self.machine_list:
            machine.reset()
        print("multi-domain scheduling system is reset.")

    def print_version(self):
        """Print version at initialization
        """
        print("--------------------------------------------------------")
        print("|                                                      |")
        print("|       Cross-domain task scheduling system v1.0       |")
        print("|                                                      |")
        print("--------------------------------------------------------")


def create_domain(domain_id, location_name):
    """Create one domain with domain_id 0, default location is "北京市"
    """
    return Domain(domain_id, location_name)


def create_domains(location_list):
    """Create multiple domains with given location_list
    """
    domain_list = []
    for i, location in enumerate(location_list):
        domain_list.append(Domain(i, location))
    return domain_list


def create_multi_domain(multidomain_id, location):
    """Create multi-domain system using singleton pattern
    """
    multi_domain = MultiDomain(multidomain_id, location)
    multi_domain.auto_locate()
    glo.location_longitude = multi_domain.longitude
    glo.location_latitude = multi_domain.latitude
    return multi_domain
