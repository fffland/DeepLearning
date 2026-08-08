# HW3 Image Classification (Food-11)

## 一、实验记录

| 序号 | 改动 | 结果(val best acc) | 说明 |
|---|---|---|---|
| 1 | Baseline，`Classifier`，无增强，`n_epochs=4` | 0.52505 | 接近simple baseline(0.501) |
| 2 | +数据增强(Q1)，`n_epochs=4` | 0.53926 | 提升有限，epoch太短，增强的防过拟合作用还没体现 |
| 3 | +训练更久，`n_epochs=80`，无正则化 | 0.73971（epoch69） | **严重过拟合**：epoch80时train acc=0.973，val回落到0.713，train/val差距~23-26个点 |
| 4 | +Dropout(0.3)，`n_epochs=20`（未跑完） | 0.68368（末轮仍在涨） | 差距收窄到~5个点，方向正确但未收敛 |
| 5 | Dropout→0.45 + weight_decay(5e-4) + LR Scheduler，`n_epochs=40` | 0.74522（epoch39） | 较3基本持平，差距回升到~11个点，末轮仍在刷新，训练还不充分 |
| 6 | 5 + 增强强化 + Normalize + Label Smoothing(0.1)，`n_epochs=70` | **0.78352**（epoch65） | **本次最优**，超过medium baseline(0.732)；曲线后期平稳，非持续恶化的过拟合 |
| 7 | Q2 `Residual_Network`，初版直接拍平接fc | 训练卡死，acc~0.145 | fc第一层6700万参数过大，训练信号被稀释，非代码错误 |
| 8 | 7改用Global Average Pooling，套用6的配置 | 0.68336（epoch39，触发早停） | 低于`Classifier`约10个点，推测超参数未针对新架构重调 |

**最终采用**：`Classifier` + 增强 + Normalize + Dropout(0.45) + weight_decay(5e-4) + CosineAnnealingLR + Label Smoothing(0.1)，**val best acc = 0.78352**

## 二、结论

- 数据增强需要足够训练轮数才能体现价值，短训练下可能只看到"代价"（收敛变慢）看不到"收益"
- train/val差距是否**持续扩大**，比某一轮的绝对差距数值更能反映过拟合是否仍在恶化
- Dropout、weight_decay、增强强度、Label Smoothing、Normalize都有正向贡献，但都不是单独解决过拟合的"万能解"，需要组合、且力度要匹配训练轮数
- fc层拍平前的特征图维度直接决定参数量与训练启动速度，Global Average Pooling能有效控制这个瓶颈
- 模型架构升级不保证提升——沿用别的架构调好的超参数、未重新调优，效果可能反而更差，架构和超参数要一起调