import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from utils.tools import EarlyStopping, adjust_learning_rate
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
        batch_x, label = batch_data
        batch_x = batch_x.float().to(self.device)
        # padding_mask = padding_mask.float().to(self.device)
        label = label.to(self.device)

        outputs = self.model(batch_x, None, None, None)

        pred = outputs.detach().cpu()
        loss = self.loss_func(pred, label.long().squeeze().cpu())
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