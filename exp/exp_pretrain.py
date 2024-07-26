import os
import time
import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import warnings
warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.loss_func = F.mse_loss()
        
        
    def training_step(self, batch_data):
        batch_x, label = batch_data
        batch_x = batch_x.float().to(self.device)

        # reconstruction
        outputs = self.model(batch_x, None, None, None)
        loss = self.loss_func(outputs, label.long().squeeze(-1))
        return loss
    
    def validation_step(self, batch_data):
        pass
    
    def configure_optimizers(self):
        pass
    