# Time Series Library (TSlib)
时间序列仓库,用于单变量或多变量时间序列的**预训练**、**分类**、**回归**、**异常检测**、**预测**任务

在[Time Series Library](https://github.com/thuml/Time-Series-Library)的基础上进行了修改,将每个任务模块进行剥离，让骨干网络和头部模块化，方便做工程项目

# Function
- [ ] 多GPU支持
- [ ] tensorboard可视化
- [ ] 各类任务数据集的探索性数据分析(exploratory data analysis)
- [ ] 预训练任务
- [ ] 数据增强
 
- [x] 对训练代码进行封装
# Tasks

## 1. Pretrain
分离骨干网络,对骨干网络做预训练
- [ ] self-supervised pretrain
- [ ] masked reconstruction pretrain
- [ ] next series prediction pretrain

## 2. Classification
单变量时序分类和多变量时序分类

## 3. Regression
TODO

## 4. Prediction
TODO

## 5. Imputation
TODO

# Usage
docker 镜像
```bash
docker load -i <path_to_docker_image>
docker run -idt --name=TSlib --shm-size 16G -v /TS-Lib:/opt/TSlib  /bin/bash
```
