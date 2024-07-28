import os
import torch
import random
import argparse
import numpy as np
import pandas as pd

# from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# from exp.exp_imputation import Exp_Imputation
# from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
# from exp.exp_anomaly_detection import Exp_Anomaly_Detection
# from exp.exp_classification import Exp_Classification
from exp.exp_pretrain import Exp_Pretrain
from utils.print_args import print_args

from sklearn.model_selection import StratifiedKFold
from data_provider.cus_dataloader import PretrainLoader
from utils.my_logger import Logger
from utils.tools import create_version_folder, _set_logger, seed_everything


def cross_validation(args):

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    # dataloader
    dataloader = PretrainLoader(args)

    y = dataloader.y
    
    version, version_path = create_version_folder(args.log_dir)
    # version = 'version12'
    # version_path = './log/pretrain/version12'
    
    args.version = version
    args.version_path = version_path
    logger = _set_logger(args)
    args.logger = logger

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        train_loader, val_loader = dataloader.get_loader(train_idx, mode='train', return_val=True)  # 20%用于val
        test_loader = dataloader.get_loader(test_idx, mode='test')
        
        args.fold = fold
        logger.info(f"=================== fold {fold} ===================")

        # backbone

        # model
        Exp = Exp_Pretrain
        model = Exp(args)  # set experiments

        # train val test
        logger.info(f'>>>>>>> start training <<<<<<<<<<<<')
        model.train(train_loader, val_loader)

        logger.info(f'>>>>>>> start testing <<<<<<<<<<<<')
        preds, trues = model.test(test_loader)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    seed_everything(seed=42)

    parser = argparse.ArgumentParser(description='NSTransformer')

    # basic config
    parser.add_argument("--log_dir", type=str, default='./log', help="日志记录路径")
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    # data loader
    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('--data', type=str, required=True,
                        default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str,
                        default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str,
                        default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str,
                        default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96,
                        help='input sequence length')
    parser.add_argument('--label_len', type=int,
                        default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str,
                        default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--enc_in', type=int, default=7,
                        help='encoder input size')
    parser.add_argument('--c_out', type=int, default=7,
                        help='自编码输出维度')
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str,
                        default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int,
                        default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int,
                        default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3,
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='adjust learning rate')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='loss function')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str,
                        default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int,
                        default=2, help='number of hidden layers in projector')

    # random seed
    parser.add_argument('--seed', type=int, default=2,
                        help="Randomization seed")

    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    # print_args(args)

    cross_validation(args)

    # if args.task_name == 'long_term_forecast':
    #     Exp = Exp_Long_Term_Forecast
    # elif args.task_name == 'short_term_forecast':
    #     Exp = Exp_Short_Term_Forecast
    # elif args.task_name == 'imputation':
    #     Exp = Exp_Imputation
    # elif args.task_name == 'anomaly_detection':
    #     Exp = Exp_Anomaly_Detection
    # elif args.task_name == 'classification':
    #     Exp = Exp_Classification
    # else:
    #     Exp = Exp_Long_Term_Forecast
    # #================================================================
    # exp = Exp(args)  # set experiments
    # setting = 'v1'

    # print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    # exp.train(setting)

    # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.test(setting)
    # torch.cuda.empty_cache()
