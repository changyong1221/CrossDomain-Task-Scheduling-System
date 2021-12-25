import os


def clear_results(dir_path):
    """Clear generated results
    """
    ls = os.listdir(dir_path)
    for file in ls:
        file_path = os.path.join(dir_path, file)
        os.remove(file_path)
    print("All results cleared.")


if __name__ == "__main__":
    dir_path_name = "task_run_results"
    clear_results(dir_path_name)
