# HW3 Image Classification (Food-11)

## 一、实验记录

1. **改动**：跑通baseline，`Classifier`（sample code默认CNN，5层Conv+BN+ReLU+MaxPool，`fc`层1024→512→11），`train_tfm`不含任何随机增强（仅Resize+ToTensor），`n_epochs=4`
   **结果**：val acc 0.36021 → 0.44492 → 0.45794 → 0.52505（最终/最佳）
   **说明**：纯baseline水平，接近PPT给出的simple baseline(0.501)。train acc未记录。

2. **改动**：在1的基础上，只加`train_tfm`数据增强（RandomResizedCrop + RandomHorizontalFlip + RandomRotation + ColorJitter），`n_epochs=4`不变，对应Q1
   **结果**：val acc 0.34943 → 0.43855 → 0.50577 → 0.53926（最终/最佳）
   **说明**：相比baseline最终仅提升约1.4个点，前两轮甚至略低于baseline。原因分析：`n_epochs=4`太短，增强"防过拟合"的核心作用还没有机会体现，反而增强让每轮数据更"嘈杂"、收敛稍慢，代价先于收益显现。

3. **改动**：在2的基础上，`n_epochs`从4提高到80，其余不变（无dropout、无weight_decay调整）
   **结果**：val best acc = 0.73971（epoch69），但epoch80时 train acc = 0.97294，val acc回落到0.71270
   **说明**：出现严重过拟合，train/val差距达约23-26个百分点；val loss从epoch30左右开始持续爬升（1.0→2.1区间反复），是过拟合中后期典型信号。确认了"训练更久"必须搭配正则化手段，否则收益被过拟合抵消。

4. **改动**：在3的基础上，`fc`层加Dropout(p=0.3)，`n_epochs`改为20（未跑满，提前查看中间结果）
   **结果**：train acc = 0.73143，val acc = 0.68368（最后一轮仍在刷新最佳，未收敛）
   **说明**：虽未跑完，但train/val差距从约23个点收窄到约5个点，验证了Dropout抑制过拟合的方向是对的。因该版本未跑至收敛，最终水平上限未知，仅作为方向性验证保留。

5. **改动**：Dropout提高到0.45，weight_decay从1e-5加大到5e-4，同时加入CosineAnnealingLR学习率调度器（`scheduler.step()`每个epoch调用一次），`n_epochs=40`
   **结果**：val best acc = 0.74522（epoch39，最后一轮才刷新最佳），中途epoch31时为 train acc = 0.85167，val acc = 0.73914
   **说明**：Dropout+weight_decay力度加大后，best val acc较3（0.73971）基本持平、略有提升，说明这一步单独带来的增益有限；train/val差距相比4反而略有回升（约11个点 vs 4的约5个点），推测与`n_epochs`提高、模型有更多轮次继续拟合训练集有关。最后一轮仍在刷新最佳、尚未看到明显平台期，说明这版配置本身可能还没训练充分，是后续叠加数据增强强化/Normalize/Label Smoothing（见6）后效果显著提升（0.74522→0.78352）的原因之一——不完全是新增手段的功劳，延长的有效训练时间也有贡献，两者在6中是混合在一起的，不是严格单变量对照。

6. **改动**：在5的基础上，同时叠加：数据增强强度加大（`RandomResizedCrop(scale=(0.6,1.0))` + `RandomErasing(p=0.3)`）、`Normalize`（train_tfm与test_tfm同步添加，ImageNet统计均值方差）、`CrossEntropyLoss(label_smoothing=0.1)`，`n_epochs=70`，`patience=7`
   **结果**：train acc最终稳定在约0.925，val best acc = **0.78352**（epoch65），最后10轮train/val acc均趋于平稳（val acc在0.777~0.784区间波动，val loss稳定在1.06附近不再持续爬升）
   **说明**：这是本次HW3综合表现最好的一版，超过medium baseline(0.732)，接近strong baseline(0.819)。虽然train/val差距约14个点看起来比5更大，但结合走势判断，此时模型已收敛到稳定平衡点，而非仍在恶化的过拟合——差距不再持续扩大，属于模型容量与任务难度共同决定的"稳态差距"，而非需要继续压制的活跃问题。本次实验一次性改动多个变量，非严格单变量对照，结论仅作整体方向参考。

7. **改动**：完成Q2 `Residual_Network`（6层卷积+3处残差相加：layer2+layer1、layer4+layer3、layer6+layer5），套用6的完整正则化/增强配置训练。初版`forward`直接拍平特征图（`256*32*32=262144`维）接`fc_layer`
   **结果**：训练卡住，前3轮val acc停留在0.145附近（接近11分类随机水平0.091），loss停在ln(11)≈2.4附近不降
   **说明**：定位为`fc_layer`第一层参数量过大（约6700万参数，是`Classifier`同层的8倍），训练信号被稀释，模型"启动"过慢，非代码逻辑错误。改用Global Average Pooling（`F.adaptive_avg_pool2d`）压缩特征图后再接`fc_layer`（输入维度262144→256）后问题解决。

8. **改动**：`Residual_Network`（GAP版）+ 6的完整配置，`n_epochs=70`，`patience=7`
   **结果**：val best acc = **0.68336**（epoch39），随后触发早停（连续7轮未刷新最佳，约epoch47左右停止）
   **说明**：低于`Classifier`同配置下的0.78352，差距约10个百分点，且训练曲线波动明显大于`Classifier`。推测主因是学习率/正则化超参数是直接沿用`Classifier`调好的配置，未针对`Residual_Network`重新调参，两种架构的最优超参数本就可能不同。作为诚实的负面结果记录，未继续为该架构单独调参。

## 二、结论

- 数据增强的效果需要足够长的训练轮数才能体现；短训练下增强的"代价"（收敛变慢）可能先于"收益"（防过拟合）显现，不能仅凭短训练结果否定增强的价值
- train/val准确率差距持续扩大（尤其伴随val loss持续爬升）是判断过拟合是否仍在恶化的关键信号，比只看某一轮的绝对差距数值更可靠；差距稳定不再扩大，即使数值不小，也可能只是模型的"稳态差距"
- Dropout、weight_decay、数据增强强度、Label Smoothing、Normalize在这次任务中都对最终效果有正向贡献，但没有一个是单独能解决过拟合问题的"万能解"，需要组合使用，且力度需要与训练轮数、模型容量匹配着调整
- 全连接层的输入维度（拍平前特征图大小）对参数量影响巨大，直接决定了训练启动速度和过拟合风险；用Global Average Pooling替代直接拍平，是控制这个瓶颈的有效手段
- 模型架构升级（如加入残差连接）不保证自动带来效果提升，尤其在直接沿用其他架构调好的超参数、未针对新架构重新调优的情况下，效果可能反而更差；架构和超参数需要匹配着一起调，不是独立变量
- 最终采用配置：`Classifier` + 数据增强(RandomResizedCrop+RandomErasing等) + Normalize + Dropout(0.45) + weight_decay(5e-4) + CosineAnnealingLR + Label Smoothing(0.1)，best val acc = 0.78352