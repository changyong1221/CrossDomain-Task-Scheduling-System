import pandas as pd

if __name__ == "__main__":
    path = f"GoCJ_Dataset_5000batches_40concurrency_train.txt"
    df = pd.read_csv(path, header=None, delimiter='\t')
    print(df[1000:2000])