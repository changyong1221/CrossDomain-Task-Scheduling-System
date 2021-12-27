import numpy as np
import pandas as pd
from utils.read_data import read_line_elem_from_file
from utils.write_file import write_vector_to_file


def create_whole_gocj(records_num, output_path):
    """Create gocj dataset of records_num size

    Attributes: task_id, task_commit_time, task_mi, task_cpu_utilization, task_size

    """
    file_path = "../dataset/GoCJ/Original_DataSet.txt"
    data_categories_list = read_line_elem_from_file(file_path)
    data_categories_num = len(data_categories_list)
    data_vector = []
    tasks_concurrency = 5

    cpu_uti_range_list = [1.0, 0.8, 0.6, 0.4, 0.2]
    mi_range_list = [525000, 150000, 101000, 59000, 15000]

    def find_range(lst, val):
        lst_len = len(lst)
        for i in range(lst_len):
            if val >= lst[i]:
                return i

    for i in range(records_num):
        data_category = np.random.randint(data_categories_num)
        task_mi = data_categories_list[data_category]
        range_idx = find_range(mi_range_list, task_mi)
        data_vector.append([i, i // tasks_concurrency, task_mi, cpu_uti_range_list[range_idx], task_mi // 10])

    write_vector_to_file(data_vector, output_path, mode='w', delimiter='\t')


def create_non_iid_data_gocj(clients_num):
    """Create non-iid data using GoCJ
    """
    # data_vector = pd.read_csv(input_path, header=None, delimiter='\t')
    # cols = ["task_id", "commit_time", 'mi', 'cpu_uti', 'size']
    # data_vector.columns = cols
    # print(len(data_vector))

    file_path = "../dataset/GoCJ/Original_DataSet.txt"
    data_categories_list = read_line_elem_from_file(file_path)
    data_categories_num = len(data_categories_list)
    tasks_concurrency = 5

    cpu_uti_range_list = [1.0, 0.8, 0.6, 0.4, 0.2]
    mi_range_list = [525000, 150000, 101000, 59000, 15000]

    def find_range(lst, val):
        lst_len = len(lst)
        for i in range(lst_len):
            if val >= lst[i]:
                return i

    client_slice_num = data_categories_num // clients_num
    client_records_num = 2000
    for client_idx in range(clients_num):
        client_data_vector = []
        for i in range(client_records_num):
            category_id = np.random.randint(client_idx * client_slice_num, (client_idx + 1) * client_slice_num)
            task_mi = data_categories_list[category_id]
            range_idx = find_range(mi_range_list, task_mi)
            client_data_vector.append([i, i // tasks_concurrency, task_mi, cpu_uti_range_list[range_idx],
                                       task_mi // 10])
        output_path = f"../dataset/GoCJ/client/GoCJ_Dataset_{client_records_num}_client_{client_idx}.txt"
        write_vector_to_file(client_data_vector, output_path)


def create_gocj():
    # test dataset
    n_records = 2000
    output_path = f"../dataset/GoCJ/GoCJ_Dataset_{n_records}_test.txt"
    create_whole_gocj(n_records, output_path)

    # train dataset
    # n_records = 12000
    # output_path = f"../dataset/GoCJ/GoCJ_Dataset_{n_records}_train.txt"
    # create_whole_gocj(n_records, output_path)


if __name__ == '__main__':
    # n_clients = 10
    # create_non_iid_data_gocj(n_clients)

    create_gocj()

