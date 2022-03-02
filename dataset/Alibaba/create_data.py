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
    df = df[['start_time', 'length', 'cpu_avg', 'size']]
    df = df.drop(df[(df['length'] == 0) | (df['cpu_avg'] == 0) | (df['cpu_avg'] > 1) | (df['length'] < 0)
                    ].index)
    # 丢弃空值
    df = df.dropna()
    # 设置每时刻并发数为10
    # df['commit_time'] = df.index // 10 + 1
    # 调整列顺序
    # order = ['start_time', 'length', 'cpu_avg', 'size']
    # df = df[order]
    df.sort_values('start_time', inplace=True)
    # print(len(df))
    # print(len(df['start_time'][:100000].unique()))
    # exit()

    test_records_num = 1000000
    train_records_num = 1000000     # 总的训练集数量

    # 计算最大最小batch_size
    # tmp_df = df[0: test_records_num]
    # batch_times = tmp_df['start_time'].unique()
    # batch_num = len(batch_times)
    # print("batch_num: ", batch_num)
    # max_len = 0
    # min_len = 100
    # for i in range(batch_num):
    #     i_len = len(tmp_df[tmp_df['start_time'] == batch_times[i]])
    #     max_len = i_len if i_len > max_len else max_len
    #     min_len = i_len if i_len < min_len else min_len
    #
    # print("max_len: ", max_len)
    # print("min_len: ", min_len)
    # exit()

    fileName = 'Alibaba-Cluster-trace-' + str(train_records_num) + '-test.txt'
    tmp_df = df[:test_records_num]
    tmp_df.to_csv(fileName, header=False, sep='\t')
    # fileName = 'Alibaba-Cluster-trace-' + str(test_records_num) + '-test.txt'
    # tmp_df = df[test_records_num: test_records_num + train_records_num]


    # 生成测试用的batch数据集
    # idx = 0
    # for i in range(1000, 11000, 1000):
    #     tmp_df = df[idx : idx+i]
    #     fileName = 'batch/Alibaba-Cluster-trace-' + str(i) + '-batch-test.txt'
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
    df = df[['start_time', 'length', 'cpu_avg', 'size']]
    df = df.drop(df[(df['length'] == 0) | (df['cpu_avg'] == 0) | (df['cpu_avg'] > 1) | (df['length'] < 0)
                    ].index)
    # 丢弃空值
    df = df.dropna()
    # 设置每时刻并发数为10
    # df['commit_time'] = df.index // 10 + 1
    # 调整列顺序
    # order = ['start_time', 'length', 'cpu_avg', 'size']
    # df = df[order]
    df.sort_values('start_time', inplace=True)
    # print(len(df[df['start_time'] == 87131]))
    # print(df[:1000])
    # exit()

    test_records_num = 100000
    train_records_num = 100000      # 单个客户端的数据集数量
    start_idx = test_records_num

    # fileName = 'Alibaba-Cluster-trace-' + str(train_records_num) + '-test.txt'
    # tmp_df = df[:test_records_num]
    # tmp_df.to_csv(fileName, header=False, sep='\t')
    for i in range(client_num):
        tmp_df = df[start_idx + train_records_num*i: start_idx + train_records_num*(i+1)]
        fileName = f'client/Alibaba-Cluster-trace-{train_records_num}-client-{i}.txt'
        tmp_df.to_csv(fileName, header=False, sep='\t')


if __name__ == '__main__':
    # 解决控制台输出省略号的问题
    pd.set_option('display.max_columns', 1000)      # 显示所有列
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.max_rows', None)         # 显示所有行

    filePath = "Alibaba-Cluster-trace-v2018-10million.csv"
    # create_dataset(filePath)
    create_client_dataset(filePath, client_num=10)
