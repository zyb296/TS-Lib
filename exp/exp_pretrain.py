import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import warnings
warnings.filterwarnings('ignore')


class Exp_Pretrain(Exp_Basic):
    def __init__(self, args):
        super(Exp_Pretrain, self).__init__(args)
        self.loss_func = nn.MSELoss()
        
        
    def training_step(self, batch_data):
        batch_x, label = batch_data
        batch_x = batch_x.float().to(self.device)

        # reconstruction
        outputs = self.model(batch_x, None, None, None)
        # print(f"outputs: {outputs.shape}")
        loss = self.loss_func(outputs, batch_x)
        return loss
    
    def validation_step(self, batch_data):
        batch_x, label = batch_data
        batch_x = batch_x.float().to(self.device)
        # print(f"batch_x: {batch_x.shape}")

        # reconstruction
        outputs = self.model(batch_x, None, None, None)
        # print(f"outputs: {outputs.shape}")
        loss = self.loss_func(outputs, batch_x)
        return loss
    
    def test(self, test_loader):
        # 加载模型
        # check_point_path = os.path.join(self.args.checkpoints, self.args.version)
        # pth_path = os.path.join(check_point_path, f"checkpoint_fold{self.args.fold}.pth")
        self.model.load_state_dict(torch.load(self.pth_path))
        
        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                # ------- one batch ------------
                label, outputs = self.test_step(batch_data)
                # -----------------------------
                preds.append(outputs.detach())
                trues.append(label)
        if self.args.task_name == 'pretrain':
            # TODO 随机打印一个重建结果
            one_sample_pre = preds[0][0].cpu().numpy()
            one_sample_true = trues[0][0].cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(6, 4))
            _ = ax.plot(one_sample_pre[:, 0], label='pre')
            _ = ax.plot(one_sample_true[:, 0], label='true')
            plt.legend()
            # 将 matplotlib 图像转换为 numpy 数组
            plt.tight_layout()
            canvas = plt.gcf().canvas
            canvas.draw()

            # 获取图像大小
            width, height = canvas.get_width_height()

            # 将 RGBA 转换为 numpy 数组
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)

            # 转换为 PyTorch 张量
            image = torch.from_numpy(image).permute(2, 0, 1)
            # 记录图像数据到 TensorBoard
            self.writer.add_image('figs/reconstruct', image, global_step=0)
            
            return preds, trues
        
            
    def test_step(self, batch_data):
        batch_x, label = batch_data
        batch_x = batch_x.float().to(self.device)
        # print(f"batch_x: {batch_x.shape}")

        # reconstruction
        outputs = self.model(batch_x, None, None, None)
        return batch_x, outputs
    
    # def configure_optimizers(self):
    #     pass
    