import pandas as pd
import os
import numpy as np
import copy
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D


"""预定义变量"""

# 标签类别数
n_classes = 3
# 特征数量
n_features = 3
# 特征名称
features = ['task_mi', 'task_cpu_uti', 'task_size', 'label']
# 数据集路径
file_path = 'task_type_data.txt'

"""超参数"""
optimizer = 'adam'
loss_func = 'categorical_crossentropy'
batch_size = 16
epochs = 10
validation_ratio = 0.3
# dataset = 'GoCJ'
dataset = 'Alibaba'


class DataSet(object):
    def __init__(self, data_path, train_ratio=0.7):
        # whole data
        self.train_data_size = train_ratio
        self.test_data_size = 1 - self.train_data_size

        # 加载任务类型数据集
        data = pd.read_csv(data_path, header=None, delimiter='\t')
        data.columns = features
        # 数据标准化，标准化参数需要保存
        self.task_mi_mean = data['task_mi'].mean()
        self.task_mi_std = data['task_mi'].std()
        self.task_cpu_uti_mean = data['task_cpu_uti'].mean()
        self.task_cpu_uti_std = data['task_cpu_uti'].std()
        self.task_size_mean = data['task_size'].mean()
        self.task_size_std = data['task_size'].std()
        if dataset == 'GoCJ':
            self.task_cpu_uti_mean = 0
            self.task_cpu_uti_std = 1
        data_stand_para_save_path = "data_stand_paras.txt"
        with open(data_stand_para_save_path, 'a+') as f:
            f.write(f"{self.task_mi_mean}\n{self.task_mi_std}\n{self.task_cpu_uti_mean}\n{self.task_cpu_uti_std}\n{self.task_size_mean}\n{self.task_size_std}\n")

        data['task_mi'] = (data['task_mi'] - self.task_mi_mean) / self.task_mi_std
        data['task_cpu_uti'] = (data['task_cpu_uti'] - self.task_cpu_uti_mean) / self.task_cpu_uti_std
        data['task_size'] = (data['task_size'] - self.task_size_mean) / self.task_size_std

        split_idx = (int)(len(data) * self.train_data_size)
        df_train = data[:split_idx]
        df_train_data = df_train[['task_mi', 'task_cpu_uti', 'task_size']]
        df_train_label = df_train['label']
        df_test = data[split_idx:]
        df_test_data = df_test[['task_mi', 'task_cpu_uti', 'task_size']]
        df_test_label = df_test['label']

        self.train_data = df_train_data.values.reshape(-1, 1, n_features, 1, 1)
        self.train_label = df_train_label.values

        self.test_data = df_test_data.values.reshape(-1, 1, n_features, 1, 1)
        self.test_label = df_test_label.values
        
        # 数据类型转换float32
        self.train_data = self.train_data.astype('float32')
        self.test_data = self.test_data.astype('float32')
        # one-hot编码
        self.train_label = np_utils.to_categorical(self.train_label, n_classes)
        self.test_label = np_utils.to_categorical(self.test_label, n_classes)

    def get_test_dataset(self):
        return self.test_data, self.test_label

    def get_train_dataset(self):
        return self.train_data, self.train_label
    
    def get_data_standardization_paras(self):
        return self.task_mi_mean, self.task_mi_std, self.task_cpu_uti_mean, self.task_cpu_uti_std, self.task_size_mean, self.task_size_std


class TaskTypePredictor():
    def __init__(self):
        self.model = Sequential()
        
        # Feature Extraction()
        input_shape = (1, n_features, 1)
        # 第一层：卷积，32个1*3卷积核，激活函数使用RELU
        self.model.add(Conv2D(filters=32, kernel_size=(1, 1), activation='relu',
                         input_shape=input_shape))
        # 第二层：卷积，64个1*3卷积核，激活函数使用RELU
        self.model.add(Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
        # 最大池化层，池化窗口1*2
        self.model.add(MaxPooling2D(pool_size=(1, 1)))
        # Dropout 25%的输入神经元
        self.model.add(Dropout(0.25))
        # 将Pooled feature map 摊平输入全连接层
        self.model.add(Flatten())
        
        # Classification 全连接层
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        # 使用softmax激活函数做多分类
        self.model.add(Dense(n_classes, activation='softmax'))
        
        # 编译模型
        self.model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])
        
        self.task_mi_mean = None
        self.task_mi_std = None
        self.task_cpu_uti_mean = None
        self.task_cpu_uti_std = None
        self.task_size_mean = None
        self.task_size_std = None
        # self.save_path = f"save/{dataset}/predict_net.h5"
        # self.data_stand_paras_save_path = f"data_stand/{dataset}/data_stand_paras.txt"
        self.save_path = f"predictor/save/{dataset}/predict_net.h5"
        self.data_stand_paras_save_path = f"predictor/data_stand/{dataset}/data_stand_paras.txt"
    
    def train(self, data_path):
        dataset = DataSet(data_path, train_ratio=0.7)
        x_train, y_train = dataset.get_train_dataset()
        self.task_mi_mean, self.task_mi_std, self.task_cpu_uti_mean, self.task_cpu_uti_std, self.task_size_mean, self.task_size_std = dataset.get_data_standardization_paras()

        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, validation_split=validation_ratio)
        df = pd.DataFrame(history.history)
        output_path = "history.csv"
        df.to_csv(output_path)
        self.model.save_weights(self.save_path)

    def evaluate(self, data_path, load_model=False):
        if load_model:
            self.model.load_weights(self.save_path)
        
        dataset = DataSet(data_path, train_ratio=0)
        x_test, y_test = dataset.get_test_dataset()
        
        loss, acc = self.model.evaluate(x_test, y_test, batch_size=batch_size)
        print(f"test loss:{loss}, test acc: {acc}")
    
    def predict(self, x_predict, load_model=False):
        if load_model:
            self.model.load_weights(self.save_path)
        
        x_predict = np.array(x_predict, dtype=float)
        # 对输入数据预处理
        if self.task_mi_mean is None:
            df = pd.read_csv(self.data_stand_paras_save_path, header=None)
            df.columns = ['para']
            self.task_mi_mean = df['para'][0]
            self.task_mi_std = df['para'][1]
            self.task_cpu_uti_mean = df['para'][2]
            self.task_cpu_uti_std = df['para'][3]
            self.task_size_mean = df['para'][4]
            self.task_size_std = df['para'][5]
        x_predict[:, 0] = (x_predict[:, 0] - self.task_mi_mean) / self.task_mi_std
        x_predict[:, 1] = (x_predict[:, 1] - self.task_cpu_uti_mean) / self.task_cpu_uti_std
        x_predict[:, 2] = (x_predict[:, 2] - self.task_size_mean) / self.task_size_std
        x_predict = x_predict.reshape(-1, 1, n_features, 1, 1)
        
        predictions = self.model.predict(x_predict, batch_size=batch_size, verbose=1)
        y_test_pred = np.argmax(predictions, axis=1)
        print(f"{y_test_pred}")
        # 返回众数
        final_label = np.argmax(np.bincount(y_test_pred))
        return final_label
    
    def load_model(self):
        self.model.load_weights(self.save_path)

if __name__ == '__main__':
    predict_net = TaskTypePredictor()
    data_path = "dataset/GoCJ/task_type_data.txt"
    # data_path = "task_type_data.txt"
    # data_path = "predict.txt"
    predict_net.train(data_path)
    # predict_net.evaluate(data_path, load_model=True)
    
    # data = pd.read_csv(data_path, header=None, delimiter='\t')
    # data.columns = ['task_mi', 'task_cpu_uti', 'task_size', 'label']
    # print(data['label'].values)
    # data = data[['task_mi', 'task_cpu_uti', 'task_size']]
    # data = data.values
    # final_label = predict_net.predict(data, load_model=True)
    # print(final_label)
    
