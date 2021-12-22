import pandas as pd


# 创建指定大小的数据集
def create_dataset(filePath):
    df = pd.read_csv(filePath)
    # 修改列名
    df.columns = ['instance_id', 'instance_name', 'task_name', 'job_name', 'task_type', 'status', 'start_time',
                  'end_time', 'machine_id', 'seq_no', 'total_seq_no', 'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max']
    df['length'] = (df['end_time'] - df['start_time']) * 1000
    df['cpu_avg'] /= 100
    df['size'] = df['length'] / 10
    df = df.drop(df[df['status'] == 'Running'].index)
    df = df[['length', 'cpu_avg', 'size']]
    df = df.drop(df[(df['length'] == 0) | (df['cpu_avg'] == 0) | (df['cpu_avg'] > 1)].index)
    # 设置每时刻并发数为10
    df['commit_time'] = df.index // 10 + 1
    # 调整列顺序
    order = ['commit_time', 'length', 'cpu_avg', 'size']
    df = df[order]

    # print(df[:100])
    test_records_num = 2000
    train_records_num = 5000
    # tmp_df = df[0: test_records_num]
    # fileName = 'Alibaba-Cluster-trace-' + str(records_num) + '-test.txt'
    tmp_df = df[test_records_num: test_records_num + train_records_num]
    fileName = 'Alibaba-Cluster-trace-' + str(train_records_num) + '-train.txt'
    tmp_df.to_csv(fileName, header=False)

    # idx = 0
    # for i in range(100, 1100, 100):
    #     tmp_df = df[idx : idx+i]
    #     fileName = 'Alibaba-Cluster-trace-' + str(i) + '.txt'
    #     tmp_df.to_csv(fileName, header=False)
    #     idx += i


# 创建指定大小的数据集
def create_client_dataset(filePath, client_num):
    df = pd.read_csv(filePath)
    # 修改列名
    df.columns = ['instance_id', 'instance_name', 'task_name', 'job_name', 'task_type', 'status', 'start_time',
                  'end_time', 'machine_id', 'seq_no', 'total_seq_no', 'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max']
    df['length'] = (df['end_time'] - df['start_time']) * 1000
    df['cpu_avg'] /= 100
    df['size'] = df['length'] / 10
    df = df.drop(df[df['status'] == 'Running'].index)
    df = df[['length', 'cpu_avg', 'size']]
    df = df.drop(df[(df['length'] == 0) | (df['cpu_avg'] == 0) | (df['cpu_avg'] > 1)].index)
    # 设置每时刻并发数为10
    df['commit_time'] = df.index // 10 + 1
    # 调整列顺序
    order = ['commit_time', 'length', 'cpu_avg', 'size']
    df = df[order]

    # print(df[:100])
    test_records_num = 2000
    train_records_num = 5000
    start_idx = test_records_num

    for i in range(client_num):
        tmp_df = df[start_idx + train_records_num*i: start_idx + train_records_num*(i+1)]
        fileName = f'client/Alibaba-Cluster-trace-{train_records_num}-client-{i + 1}.txt'
        tmp_df.to_csv(fileName, header=False)


if __name__ == '__main__':
    # 解决控制台输出省略号的问题
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)

    filePath = "Alibaba-Cluster-trace-v2018-2.csv"
    # create_dataset(filePath)
    create_client_dataset(filePath, client_num=10)
