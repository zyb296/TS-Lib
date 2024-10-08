from data_provider.uea import collate_fn
import os
import gc
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from utils.augmentation import jitter

import sys
sys.path.append("..")


class spO2_HerarRate_Dataset(Dataset):
    def __init__(self, X, y=None, mode="train"):
        self.mode = mode
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.mode == "train":
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx], torch.Tensor([0.])


def data_normal(data):
    # mean = data.mean(axis=(0, 2), keepdims=True)
    # std = data.std(axis=(0, 2), keepdims=True) + 1e-5
    mean = data.mean(axis=(0, 1), keepdims=True)
    std = data.std(axis=(0, 1), keepdims=True) + 1e-5
    return (data - mean) / std

def add_dim(data):
    div = (data[:, :, 0] / data[:, :, 1])[..., np.newaxis]
    # 对最后一个元素进行复制,用于填充
    padding = data[:, -1:, :]
    # 对每个序列执行一阶段差分
    diff_data = np.concatenate([data[:, 1:, :], padding], axis=1) - data
    print(f"diff.shape: ", diff_data.shape)
    data = np.concatenate([data, div, diff_data], axis=-1)
    print(data.shape)
    return data

class MyDataLoader:
    def __init__(self, args) -> None:
        self.args = args
        add_cal_dim = False
        # print(os.getcwd())
        # data_path = os.path.join(os.getcwd(), 'dataset')
        data_path = "./dataset/custom_dataset/"

        self.train_x = np.load(os.path.join(data_path, "训练集/train_x.npy"))
        self.train_y = np.load(os.path.join(data_path, "训练集/train_y.npy"))
        # TODO 0样本过多，应该筛选部分出来做训练和验证
        num_zero = 4600
        zero_index = list(np.where(self.train_y == 0)[0])

        total_index = zero_index[:num_zero] + \
            list(np.where(self.train_y != 0)[0])
        np.random.shuffle(total_index)
        self.train_x = self.train_x[total_index]
        # self.train_x = self.train_x.reshape(-1, 180, 2) 
        self.train_x = np.transpose(self.train_x, (0, 2, 1))
        # self.train_x = jitter(self.train_x, sigma=0.2)  # 增加噪声
        if add_cal_dim:
            self.train_x = add_dim(self.train_x)  # 增加维度
        self.train_y = self.train_y[total_index]

        self.test_x = np.load(os.path.join(data_path, "测试集A/test_x_A.npy"))
        # self.test_x = self.test_x.reshape(-1, 180, 2)
        self.test_x = np.transpose(self.test_x, (0, 2, 1))
        # self.test_x = jitter(self.test_x, sigma=0.2)  # 增加噪声
        if add_cal_dim:
            self.test_x = add_dim(self.test_x)  # 增加维度
        
        self.submission = pd.read_csv(os.path.join(
            data_path, "测试集A/submit_example_A.csv"))

        self.num_works = 4

    def get_loader(self, data_idx=None, return_val=True, mode="train"):
        """
        data_idx: 用于提取数据的idx
        return_val: 是否用train的20%作为验证集
        """
        drop_last = True
        instance_norm = True
        if mode == 'train':
            data = self.train_x[data_idx]
            label = self.train_y[data_idx]
            if return_val:
                n = int(len(data) * 0.8)  # 训练集数量
                x_train = data[:n]
                x_val = data[n:]
                if instance_norm:
                    x_train = data_normal(x_train)
                    x_val = data_normal(x_val)
                train_dataset = spO2_HerarRate_Dataset(x_train, label[:n])
                val_dataset = spO2_HerarRate_Dataset(x_val, label[n:])
                train_loader = DataLoader(train_dataset,
                                          batch_size=self.args.batch_size,
                                          shuffle=True,
                                          num_workers=self.num_works,
                                          drop_last=drop_last,
                                          #   collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len),
                                          )
                val_loader = DataLoader(val_dataset,
                                        batch_size=self.args.batch_size,
                                        shuffle=False,
                                        num_workers=self.num_works,
                                        drop_last=drop_last,
                                        # collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len),
                                        )
                return train_loader, val_loader
            else:
                data = data_normal(data)
                train_dataset = spO2_HerarRate_Dataset(data, label)
                train_loader = DataLoader(train_dataset,
                                          batch_size=self.args.batch_size,
                                          shuffle=True,
                                          num_workers=self.num_works,
                                          drop_last=drop_last,
                                          #   collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len),
                                          )
                return train_loader
        elif mode == 'test':
            data = self.train_x[data_idx]
            label = self.train_y[data_idx]
            data = data_normal(data)
            test_dataset = spO2_HerarRate_Dataset(data, label)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.args.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_works,
                                     # collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len),
                                     )
            return test_loader
        elif mode == 'predict':
            data = self.test_x
            data = data_normal(data)
            predict_data = spO2_HerarRate_Dataset(data, y=None, mode='predict')
            predict_loader = DataLoader(predict_data,
                                        batch_size=self.args.batch_size,
                                        shuffle=False,
                                        num_workers=self.num_works,
                                        # collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len),
                                        )
            return predict_loader
    
    
class PretrainLoader:
    def __init__(self, args):
        self.args = args
        
        data_path = "./dataset/custom_dataset/"

        self.train_x = np.load(os.path.join(data_path, "训练集/train_x.npy")).transpose((0, 2, 1))
        self.test_x = np.load(os.path.join(data_path, "测试集A/test_x_A.npy")).transpose((0, 2, 1))
        self.data = np.concatenate((self.train_x, self.test_x), axis=0)
        self.y = np.zeros(self.data.shape[0])
        # 全局做归一化
        # self.data = data_normal(self.data)
        self.num_works = 8
        
        
    def get_loader(self, data_idx, return_val=False, mode='train'):
        # if mode == 'train':
        data = self.data[data_idx]
        drop_last = True
        if return_val:
            n = int(len(data_idx) * 0.8)
            x_train = data[:n]
            x_val = data[n:]
            x_train = data_normal(x_train)
            x_val = data_normal(x_val)
            train_dataset = spO2_HerarRate_Dataset(x_train, y=None, mode='test')
            val_dataset = spO2_HerarRate_Dataset(x_val, y=None, mode='test')
            train_loader = DataLoader(train_dataset,
                                        batch_size=self.args.batch_size,
                                        shuffle=True,
                                        num_workers=self.num_works,
                                        drop_last=drop_last,
                                        )
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.args.batch_size,
                                    shuffle=False,
                                    num_workers=self.num_works,
                                    drop_last=drop_last,
                                    )
            return train_loader, val_loader
            
        data = data_normal(data)
        dataset = spO2_HerarRate_Dataset(data, y=None, mode='test')
        data_loader = DataLoader(dataset,
                                batch_size=self.args.batch_size,
                                shuffle=True,
                                num_workers=self.num_works,
        )
        return data_loader
            