import os

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
from utils.my_logger import Logger

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, pth_path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, pth_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, pth_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, pth_path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), pth_path)
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def create_version_folder(log_dir):
    # 检查日志文件夹是否存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 获取所有文件夹名称
    folders = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]
    
    # 过滤出版本文件夹
    version_folders = [f for f in folders if f.startswith('version')]
    
    if not version_folders:
        # 如果没有版本文件夹，创建version1
        new_version = 'version1'
    else:
        # 获取所有版本号
        version_numbers = [int(f.replace('version', '')) for f in version_folders]
        
        # 找到最大版本号
        max_version = max(version_numbers)
        
        # 创建新的版本文件夹
        new_version = f'version{max_version + 1}'
    
    # 创建新的版本文件夹路径
    new_version_path = os.path.join(log_dir, new_version)
    os.makedirs(new_version_path, exist_ok=True)
    
    return new_version, new_version_path

def _set_logger(args):
    """logger, 记录各种日志, 打印训练参数"""
    os.makedirs("./log", exist_ok=True)
    
    log_file = os.path.join('./log', f'{args.model}-{args.task_name}.log')
    log = Logger(log_file, level='debug')
    logger = log.get_logger()
    
    # 打印参数
    v = args.version_path.split("/")[-1]
    logger.info('\n')
    logger.info('\n')
    logger.info(f"#################################### {v} ####################################")
    logger.info("\t" + "Basic Config" + "\t")
    logger.info(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    logger.info(f'  {"Sequence Length:":<20}{args.seq_len:<20}{"Input dimension:":<20}{args.enc_in:<20}')
    logger.info('\n')

    logger.info("\t" + "Data Loader" + "\t")
    logger.info(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    logger.info(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    logger.info('\n')
    
    logger.info("\t" + "Model Parameters" + "\t")
    logger.info(f'  {"encoder layers:":<20}{args.e_layers:<20}{"d_model:":<20}{args.d_model:<20}')
    logger.info(f'  {"d_ff:":<20}{args.d_ff:<20}')
    logger.info(f'  {"p_hidden_dims:":<20}"[{args.p_hidden_dims[0]} {args.p_hidden_dims[1]:<20}]{"p_hitten_layers:":<20}{args.p_hidden_layers:<20}')
    logger.info('\n')
    
    logger.info("\t" + "Training" + "\t")
    logger.info(f'  {"Batch Size:":<20}{args.batch_size:<20}{"Learning rate:":<20}{args.learning_rate:<20}')
    logger.info('\n')
    
    return logger

def seed_everything(seed=42):
    """
    Set seed for reproducibility.
    
    Parameters:
    seed (int): Seed value to be set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False