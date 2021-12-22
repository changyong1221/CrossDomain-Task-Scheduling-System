from utils.get_position import get_position_by_name


class Domain(object):
    def __init__(self, domain_id, location, longitude=None, latitude=None):
        """Initialization
        """
        self.domain_id = domain_id
        self.location = location        # city name, for example "北京市" "莫斯科"
        self.longitude = longitude      # longitude of geographical position
        self.latitude = latitude        # latitude of geographical position
        self.machine_list = []          # all the machines in this domain
        self.cluster_list = []          # node clusters in this domain

        if self.longitude is None or self.latitude is None:
            self.longitude, self.latitude = get_position_by_name(location)

    def add_machine(self, machine):
        """Add machine to domain
        """
        self.machine_list.append(machine)

    def clustering_machines(self, cluster_num):
        """Clustering all the machines into several clusters
        """
        pass

    def get_cluster_list(self):
        """Return cluster list
        """
        return self.cluster_list


def create_one_domain():
    """Create one domain with domain_id 0, default location is "北京市"
    """
    return Domain(0, "北京市")


def create_domains(location_list):
    """Create multiple domains with given location_list
    """
    domain_list = []
    for i, location in enumerate(location_list):
        domain_list.append(Domain(i, location))
    return domain_list
