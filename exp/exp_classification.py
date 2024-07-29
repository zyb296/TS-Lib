import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from torch.utils.tensorboard import SummaryWriter
from exp.exp_basic import Exp_Basic
import torch.nn.functional as F

from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple


class FineTuneModel(nn.Module):
    def __init__(self, args, pretrain_model):
        super(FineTuneModel, self).__init__()
        self.args = args
        l = list(pretrain_model.children())
        print("pretrain model: ", l[-3])
        self.feature_extractor = nn.Sequential(*l)
        self.act = F.gelu
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.args.d_model * self.args.seq_len, self.args.num_class)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.feature_extractor(x_enc)
        
        # Output
        out = self.act(out)
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        return out
        

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        if self.args.use_pretrain:
            self.update_head()
        
    def update_head(self):
        """替换预训练网络的头部 model.projection"""
        check_point_path = os.path.join(f"{self.args.checkpoints}/pretrain", 
                                        f"version{self.args.pretrain_version}", 
                                        f"checkpoint_fold{self.args.pretrain_fold}.pth")
        print("check_point_path: ", check_point_path)
        print("pth_path: ", self.pth_path)
        self.model.load_state_dict(torch.load(check_point_path))
        self.model = FineTuneModel(self.args, self.model).to(self.device)
        
    def validation(self, val_laoder):
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

    def train(self, train_loader, val_loader=None):
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
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                self.optimizer.step()
                # 记录 train loss
                global_step += 1
                if global_step % 20 == 0:  # 每20个step记录一次train loss
                    self.writer.add_scalar("Loss/train", loss.item(), global_step)
                
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            # train and val loss
            train_loss = np.average(train_loss)
            val_loss = self.validation(val_loader)
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.3f} Vali Loss: {val_loss:.3f}")
            self.writer.add_scalar("Loss/val", val_loss, global_step)
            
            # early-stopping and asjust learning rate
            self.early_stopping(val_loss, self.model, self.pth_path)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(self.optimizer, epoch + 1, self.args)

        # 加载最优模型
        self.model.load_state_dict(torch.load(self.pth_path))

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
        
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)
        
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.logger.info('accuracy:{}'.format(accuracy))
        # file_name='result_classification.txt'
        # f = open(os.path.join(folder_path,file_name), 'a')
        # f.write(f"{setting}  fold {self.args.fold}" + "  \n")
        # f.write('accuracy:{}'.format(accuracy))
        # f.write('\n')
        # f.write('\n')
        # f.close()
        return accuracy

    def prediction(self, prediction_loader, version, fold):
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
        if self.args.use_pretrain:
            pretrained_params = []
            head_params = []
            for name, param in self.model.named_parameters():
                if "linear" in name:
                    head_params.append(param)
                else:
                    pretrained_params.append(param)
            optimizer = optim.Adam([
                {"params": pretrained_params, "lr": 0.1 * self.args.learning_rate},  # 1e-4
                {"params": head_params, "lr": self.args.learning_rate},  # 1e-3
            ])
        else:
            optimizer = optim.Adam(self.model.parameters(),
                               lr=self.args.learning_rate)
        return optimizer
