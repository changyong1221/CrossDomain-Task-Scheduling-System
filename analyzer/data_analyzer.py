import numpy as np

from utils.read_data import read_line_elems_from_file
from utils.plt_config import PltConfig
from utils.create_pic import save_to_histogram_from_list


def analyze_gocj_dataset(file_path):
    """Analyze the data distribution of given GoCJ dataset
    """
    # 1. compute range_set
    range_list = [525000, 150000, 101000, 59000, 15000]
    range_name_list = ["Huge", "Extra-Large", "Large", "Medium", "Small"]
    range_set = {
        "Huge": 0,
        "Extra-Large": 0,
        "Large": 0,
        "Medium": 0,
        "Small": 0
    }

    def find_range(lst, val):
        lst_len = len(lst)
        for i in range(lst_len):
            if val >= lst[i]:
                return i

    data_vector = read_line_elems_from_file(file_path, delimiter='\t')
    data_size = len(data_vector)
    for line in data_vector:
        idx = find_range(range_list, int(line[2]))
        range_set[range_name_list[idx]] += 1
    print(range_set)

    # 2. plot data
    x_axis_data = ['Small', 'Medium', 'Large', 'Extra-Large', 'Huge']
    y_axis_data = []
    for elem in x_axis_data:
        y_axis_data.append(range_set[elem])
    y_axis_data = np.array(y_axis_data)
    y_axis_data = y_axis_data / y_axis_data.sum()
    y_axis_data = y_axis_data * 100
    print(y_axis_data)
    plt_config = PltConfig()
    plt_config.title = f"GoCJ dataset distribution (size:{data_size})"
    plt_config.xlabel = "task length"
    plt_config.ylabel = "number of records"
    plt_config.x_axis_data = x_axis_data
    # dest_dir = f"../pic/GoCJ_dataset_distribution_size_{data_size}.png"
    dest_dir = f"../pic/GoCJ_dataset_distribution_size_{data_size}_client_5.png"
    save_to_histogram_from_list(y_axis_data, dest_dir, plt_config, show=True)


def analyze_gocj_test_dataset(file_path):
    """Analyze the data distribution of given GoCJ dataset
    """
    # 1. compute range_set
    range_list = [525000, 150000, 101000, 59000, 15000]
    range_name_list = ["Huge", "Extra-Large", "Large", "Medium", "Small"]
    range_set = {
        "Huge": 0,
        "Extra-Large": 0,
        "Large": 0,
        "Medium": 0,
        "Small": 0
    }

    def find_range(lst, val):
        lst_len = len(lst)
        for i in range(lst_len):
            if val >= lst[i]:
                return i

    data_vector = read_line_elems_from_file(file_path, delimiter='\t')
    print(data_vector[:10])
    print(type(data_vector))
    data_size = len(data_vector)
    for line in data_vector:
        idx = find_range(range_list, int(line[2] / line[3]))
        range_set[range_name_list[idx]] += 1
    print(range_set)

    # 2. plot data
    x_axis_data = ['Small', 'Medium', 'Large', 'Extra-Large', 'Huge']
    y_axis_data = []
    for elem in x_axis_data:
        y_axis_data.append(range_set[elem])
    y_axis_data = np.array(y_axis_data)
    y_axis_data = y_axis_data / y_axis_data.sum()
    y_axis_data = y_axis_data * 100
    print(y_axis_data)
    plt_config = PltConfig()
    plt_config.title = f"GoCJ dataset distribution (size:{data_size})"
    plt_config.xlabel = "task length"
    plt_config.ylabel = "number of records"
    plt_config.x_axis_data = x_axis_data
    # dest_dir = f"../pic/GoCJ_dataset_distribution_size_{data_size}.png"
    dest_dir = f"../pic/GoCJ_dataset_distribution_size_{data_size}_real_task_mi.png"
    save_to_histogram_from_list(y_axis_data, dest_dir, plt_config, show=True)


if __name__ == '__main__':
    filepath = "../dataset/GoCJ/GoCJ_Dataset_20000_train.txt"
    # filepath = "../dataset/GoCJ/client/GoCJ_Dataset_2000_client_5.txt"
    analyze_gocj_dataset(filepath)
    # analyze_gocj_test_dataset(filepath)
