import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.dtw_metric import dtw, accelerated_dtw
from torch.utils.tensorboard import SummaryWriter
from exp.exp_basic import Exp_Basic


logger = logging.getLogger(__name__)

class Exp_Long_Term_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecasting, self).__init__(args)
        self.criterion = nn.MSELoss()
        
    def training_step(self, batch_data):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # logger.info(f"input_shape:  {batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape}")
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        if self.args.use_amp:  # 是否使用混合精度训练
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = self.criterion(outputs, batch_y)
            
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            loss = self.criterion(outputs, batch_y)
        return loss

    def validation_step(self, batch_data):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        pred = outputs.detach().cpu()
        true = batch_y.detach().cpu()

        loss = self.criterion(pred, true)
        return loss

    def test_step(self, batch_data):
        batch_x, label = batch_data
        batch_x = batch_x.float().to(self.device)
        # padding_mask = padding_mask.float().to(self.device)
        label = label.to(self.device)
        outputs = self.model(batch_x, None, None, None)
        return label, outputs

    def prediction_step(self, batch_data):
        return self.test_step(batch_data)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.args.learning_rate)
        return optimizer

    def test(self, test_loader: DataLoader):
        # TODO 要不要指定加载哪个模型
        # check_point_path = os.path.join(self.args.checkpoints, self.args.version)
        # pth_path = os.path.join(check_point_path, f"checkpoint_fold{self.args.fold}.pth")
        loaded_pth = self.pth_path
        logger.info(f"Loading model from {loaded_pth}")
        self.model.load_state_dict(torch.load(loaded_pth))
        test_data = test_loader.dataset
        
        # result save
        setting = f"{self.args.data}_{self.args.model}_{str(self.args.version)}"
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'not calculated'
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return