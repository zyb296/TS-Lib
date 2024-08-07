# Time Series Library (TSlib)
时间序列仓库,用于单变量或多变量时间序列的**预训练**、**分类**、**回归**、**异常检测**、**预测**任务

在[Time Series Library](https://github.com/thuml/Time-Series-Library)的基础上进行了修改,将每个任务模块进行剥离，让骨干网络和头部模块化，方便做工程项目

# TODO
[ ] 多GPU支持

[-] 对训练代码进行封装,使用者只关注模型的构建
# 任务类型

## 1. Pretrain
分离骨干网络,对骨干网络做预训练
- self-supervised pretrain
- masked reconstruction pretrain
- next series prediction pretrain

## 2. Classification
单变量时序分类和多变量时序分类

## 3. Regression
TODO

## 4. Prediction
TODO

## 5. Imputation
TODO

# Usage
TODO
