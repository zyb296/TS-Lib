import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from torch.utils.tensorboard import SummaryWriter

from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple


logger = logging.getLogger(__name__)

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            # 'Mamba': Mamba,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = self.configure_optimizers()
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        self.writer = None
        # logger.info(f"version path: {self.args.version_path}")
        # self._tensorboard_logger()
        # self._fold_checkpoint_path()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model
    
    def _tensorboard_logger(self):
        """初始化tensorboard的writer"""
        # log_path = os.path.join(self.args.log_dir, f"{self.args.setting}/fold{self.args.fold}")
        
        # log_path = os.path.join(self.args.version_path, f"fold{self.args.fold}")  # 本机路径
        if self.args.task_name == 'pretrain':
            log_path = os.path.join("/root/tf-logs", "pretrain", self.args.version_path, f"fold{self.args.fold}")
        else:
            log_path = os.path.join("/root/tf-logs", self.args.version_path, f"fold{self.args.fold}")  # autodl必须放在/root/tf-logs/路径下
        
        # logger.info(f"tensorboard path: {log_path}")
        os.makedirs(log_path, exist_ok=True)
        self.writer = SummaryWriter(log_path)
        # os.system(f"cp -r {self.args.version_path} /root/tf-logs/")

    def _acquire_device(self):

        if torch.cuda.is_available():
            device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # if self.args.use_gpu:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(
        #         self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
        #     device = torch.device('cuda:{}'.format(self.args.gpu))
        #     print('Use GPU: cuda:{}'.format(self.args.gpu))
        # else:
        #     device = torch.device('cpu')
        #     print('Use CPU')
        return device
    
    def _fold_checkpoint_path(self):
        if self.args.task_name == 'pretrain':
            check_point_path = os.path.join(self.args.checkpoints, "pretrain", self.args.version)
        elif self.args.task_name == 'finetune':
            check_point_path = os.path.join(self.args.checkpoints, "finetune", self.args.version)
        else:
            # if self.args.use_pretrain:
            #     check_point_path = os.path.join(self.args.checkpoints, "finetune", self.args.version)
            # else:
            check_point_path = os.path.join(self.args.checkpoints, self.args.version)
            
        os.makedirs(check_point_path, exist_ok=True)
        self.pth_path = os.path.join(check_point_path, f"checkpoint_fold{self.args.fold}.pth")

    def validation(self, val_laoder: DataLoader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(val_laoder):
                # ------- one batch ------------
                loss = self.validation_step(batch_data)
                # -----------------------------
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss

    def train(self, train_loader: DataLoader, val_loader: DataLoader=None) -> None:
        # 模型权重路径
        # check_point_path = os.path.join(self.args.checkpoints, self.args.version)
        # pth_path = os.path.join(check_point_path, f"checkpoint_fold{self.args.fold}.pth")
        # if not os.path.exists(check_point_path):
        #     os.makedirs(check_point_path)

        time_now = time.time()

        train_steps = len(train_loader)
        global_step = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()
            for i, batch_data in enumerate(train_loader):
                iter_count += 1
                self.optimizer.zero_grad()

                # --------- 一个batch ---------
                loss = self.training_step(batch_data)
                # -----------------------------

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                self.optimizer.step()
                # 记录 train loss
                global_step += 1
                if global_step % 20 == 0 and self.writer is not None:  # 每20个step记录一次train loss
                    self.writer.add_scalar("Loss/train", loss.item(), global_step)
                
            logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            # train and val loss
            train_loss = np.average(train_loss)
            val_loss = self.validation(val_loader)
            logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.3f} Vali Loss: {val_loss:.3f}")
            if self.writer is not None:
                self.writer.add_scalar("Loss/val", val_loss, global_step)
            
            # early-stopping and asjust learning rate
            self.early_stopping(val_loss, self.model, self.pth_path)
            if self.early_stopping.early_stop:
                logger.info("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(self.optimizer, epoch + 1, self.args)

        # 加载最优模型
        self.model.load_state_dict(torch.load(self.pth_path))

    def test(self, test_loader: DataLoader):
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
        
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)
        
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        logger.info('accuracy:{}'.format(accuracy))
        # file_name='result_classification.txt'
        # f = open(os.path.join(folder_path,file_name), 'a')
        # f.write(f"{setting}  fold {self.args.fold}" + "  \n")
        # f.write('accuracy:{}'.format(accuracy))
        # f.write('\n')
        # f.write('\n')
        # f.close()
        return accuracy

    def prediction(self, prediction_loader:DataLoader, version:str, fold:int):
        # 加载模型
        # best_model_path = os.path.join(self.args.checkpoints, version, f"checkpoint_fold{fold}.pth")
        self.model.load_state_dict(torch.load(self.pth_path))
        
        # 预测
        preds = []
        
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(prediction_loader):
                # ------- one batch ------------
                label, outputs = self.prediction_step(batch_data)
                # -----------------------------
                preds.append(outputs.detach())
        
        preds = torch.cat(preds, 0)
        print('prediction shape:', preds.shape)
        
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        return predictions

    def training_step(self, batch_data):
        batch_x, label = batch_data
        batch_x = batch_x.float().to(self.device)
        # padding_mask = padding_mask.float().to(self.device)
        label = label.to(self.device)
        # print(batch_x.shape)

        outputs = self.model(batch_x, None, None, None)
        loss = self.loss_func(outputs, label.long().squeeze(-1))
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
