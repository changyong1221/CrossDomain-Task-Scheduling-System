

def write_list_to_file(data_list, file_path, mode='a+'):
    """Write a list object to specified file
    """
    with open(file_path, mode) as f:
        for elem in data_list[:-1]:
            f.write(str(elem) + '\t')
        f.write(str(data_list[-1]))
        f.write('\n')
        f.close()
