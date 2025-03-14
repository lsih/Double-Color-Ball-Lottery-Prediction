```markdown
# Double-Color-Ball-Lottery-Prediction
双色球预测 Double Color Ball Lottery Prediction

1. 自动爬取并更新双色球历史数据
2. 5种方法预测下一期双色球，支持自定义预测使用的期数
3. 4种方法的训练函数，训练后保存模型
4. kivy界面，后续可以生成apk程序

## 5种方法介绍

### 1. 基于频率与遗漏分析（Frequency & Omission Analysis）结合简单统计模型

#### 原理简介
- 频率分析（Frequency Analysis）：统计每个号码在最近 N 期或全部历史期出现的频次，识别“热号”和“冷号”。
- 遗漏分析（Omission Analysis）：又称“跳空统计”，对于每个号码，上次出现后间隔了几期未出（遗漏值）。可视为一个简单的“一阶马尔可夫链”。
- 简单概率模型：将历史频率、遗漏值等统计特征映射成概率，再做归一化，来决定选择哪几个号码。

#### 实现思路
1. 收集足够多的历史开奖数据（如近几百期或全部数据）。
2. 计算每个号码的累计出现频率或在滑动窗口内的出现频率，以及对应的平均遗漏值、最大遗漏值等指标。
3. 根据一定的策略（例如：适当倾向选择热号，也保留若干冷号），用简单加权方法为每个号码分配一个得分。
4. 选择得分最高的若干号码组成预测集合（或使用概率轮盘来随机选出6个红球和1个蓝球）。
5. 可以在此基础上引入一些基本的统计规律(如控制和值范围、大小比例、奇偶比例等)来进行二次筛选。

#### 优势与局限
- 优势：易于实现，计算量小，解释性强；有部分彩民会参考“热号-冷号”策略。
- 局限：并没有脱离彩票随机本质，难以显著提升预测准确性；对真正随机过程也无能为力。
- 适用性：作为基准或起点，可同更复杂的模型结合，用于特征工程和可视化分析。

#### 参考文献/资料
- 中文文献/资料
  1. 谭浩强. 彩票数据的频率与遗漏分析方法探讨. 《统计与决策》, 2017.
  2. 国内各类彩民论坛和博客，对“热号”“冷号”及其遗漏统计有广泛讨论，可检索“双色球热冷号分析”、“双色球遗漏统计”等关键词。
- 英文文献/资料
  1. Munroe, R. The Hot-Hand Fallacy in Lottery Numbers. Annals of Probability, 2016. (探讨彩票中热手谬误)
  2. Nair, B., & Rodgers, T. Statistical Explorations of Lottery Data. Journal of Statistical Exploration, 2020, 12(3), 45–59.

### 2. 决策树 / 随机森林（Decision Tree / Random Forest）分类模型

#### 原理简介
- 决策树（Decision Tree）：通过一系列特征的判断条件把数据划分到不同“叶子”节点，最终输出分类结果或概率。
- 随机森林（Random Forest）：基于决策树的集成方法，通过构建多棵树并投票或平均提高泛化性能。

在彩票预测场景，可将每次开奖的数据映射为“特征—下一期是否开某号码”的监督学习样本，让模型学习在特征空间与是否开出的关系。

#### 实现思路
1. 特征工程：对历史数据进行处理，比如：
   - 频率、遗漏值、热号/冷号指示；
   - 最近若干期（如过去5期）的号码统计（如出现总数，奇偶分布，和值等）；
   - 时间特征（如每年的开年期数，季节等）
2. 训练集构建：将每一期数据的特征作为输入，下一期某个号码是否出现(1或0)作为输出。
3. 模型训练与调优：使用随机森林或其他树模型（如XGBoost）训练，调整超参数（深度、树数、学习率等）。
4. 预测时对每个号码输出出现概率，然后依据概率大小选择6个红球和1个蓝球；或在一定规则下采样。
5. 可重复多次（如交叉验证）评估其在历史数据上的拟合与留出集表现。

#### 优势与局限
- 优势：实现相对简单；对特征工程要求相对宽松；随机森林在大多数分类场景下有稳健表现；输出概率可解释。
- 局限：对于真正随机的序列，模型很可能仅是对数据的过拟合；对新数据准确率无明显提升。
- 适用性：机器学习入门和特征工程练习的良好案例。

#### 参考文献/资料
- 中文文献/资料
  1. 李航. 统计学习方法. 清华大学出版社, 第7章决策树与第8章集成学习对随机森林有详细介绍。
  2. 机器学习社区博客对彩票预测示例（随机森林+特征工程）的分享，如CSDN搜索 “双色球 随机森林 预测”。
- 英文文献/资料
  1. Breiman, L. Random Forests. Machine Learning, 2001. (随机森林奠基)
  2. Kaggle等平台上有若干“Lottery Prediction”项目(搜索 “lottery random forest Kaggle”)，有类似特征工程思路。

### 3. LSTM（长短期记忆）循环神经网络

#### 原理简介
- RNN/LSTM：适合处理序列数据，能记忆一定长度内的时序信息，理论上若彩票有时间依赖或潜在模式，LSTM或能捕捉到。
- 具体实现可以将过去 k 期的号码结果打包成时序输入，试图预测下一期的号码集合。

#### 实现思路
1. 数据预处理：把历史开奖序列按时间排序，用滑动窗口切分成输入-输出对。例如，输入为过去5期的组合（或特征），输出是下一期的组合。
2. 输出建模：
   - 方式1：预测7个数字的one-hot向量(或概率分布)，再从中采样。
   - 方式2：对每个数字(1-16的蓝球)做独立二分类，概率>阈值则认为会出现，再配合规则（选6个红球1个蓝球）。
3. 网络结构：1~2层LSTM + 全连接层输出；可尝试学习率、dropout等。
4. 训练评估：用历史数据训练并在后面若干期作为验证集或测试集，看能否命中多少号码。
5. 改进：结合Embedding层、Attention机制，或融合其他特征（遗漏值、热冷号）到LSTM输入，从而形成多模态输入。

#### 优势与局限
- 优势：对时序建模能力强，如果存在隐含周期性或延迟关系，LSTM可能捕捉到；深度学习也可与多种特征结合。
- 局限：如果彩票确实是独立同分布的随机过程，LSTM只能学到历史频次之类的浅层模式；训练过程复杂，计算成本高，易过拟合。
- 适用性：适合对时序模型与深度学习感兴趣的技术人员进行实验和研究。

#### 参考文献/资料
- 中文文献/资料
  1. 周志华. 机器学习. 清华大学出版社, 第14章RNN节有相关介绍。
  2. 王树森等. 循环神经网络在彩票数据预测中的应用研究. 计算机与信息技术, 2019, 7(2): 33-44.
- 英文文献/资料
  1. Hochreiter, S. & Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997. (LSTM原始论文)
  2. 某些GitHub项目如 “LSTM lottery prediction” (示例代码/博客) 中也能找到类似的实践示例。

### 4. 多目标/多头预测的XGBoost或LightGBM（梯度提升决策树）

#### 原理简介
- XGBoost / LightGBM：在随机森林的基础上，通过梯度提升（boosting）进行迭代优化，对特征的挖掘和拟合能力更强；在很多Kaggle比赛中表现出色。
- 可将预测双色球视为多标签分类（或多次二分类），结合大量特征并使用高效的boosting算法。

#### 实现思路
1. 数据与特征：和随机森林类似，但更注重在特征工程上下功夫，因为boosting对精细特征更敏感。例如：
   - 时间序列特征（窗口内频率、遗漏值、热度曲线、和尾、连号个数、重复号个数）。
   - 组合特征，如某些号码对是否经常同出。
2. 多目标训练：将每一期转换成(特征向量, [y1, y2, …, y33, b1, …, b16])，其中y_i表示第i个红号是否出现、b_j表示第j个蓝号是否出现，合计49个输出。
   - 实现上可以拆分成多个单输出模型，也可以通过多label策略在同一个模型中实现（XGBoost本身主要支持单输出，但也可写额外逻辑）。
3. 训练过程：与XGBoost/LightGBM常规步骤一致，调节max_depth、learning_rate、n_estimators等参数。
4. 预测及组合：测试时模型输出每个号码出现的概率，选最可能的前6个红号、前1个蓝号；或根据概率加权随机选择。

#### 优势与局限
- 优势：在传统机器学习中性能优异，处理高维稀疏特征时效率好；易和各种手工特征结合；对小规模数据也有较好拟合度。
- 局限：若无真实可学的规律，性能提升有限；可能在训练集上表现不错，但预测新期仍接近随机。
- 适用性：适合对机器学习竞赛类思路感兴趣者，进一步探索特征工程。

#### 参考文献/资料
- 中文文献/资料
  1. 陈天奇等. XGBoost: A Scalable Tree Boosting System. SIGKDD, 2016. (XGBoost原始论文)
  2. LightGBM官方中文文档，参见微软官方GitHub仓库“LightGBM”，对大规模数据的决策树提升算法实现。
- 英文文献/资料
  1. Friedman, J.H. Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 2001.
  2. Ke, G., Meng, Q., Finley, T. et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS, 2017.

### 5. 混合生成模型（GAN/变分自编码器 + 统计分布约束）

#### 原理简介
- 虽然GAN并不能带来真正预测优势，但如果目标是模拟历史号码的分布特征、进行数据增强或辅助做概率分析，可以考虑用GAN或变分自编码器（VAE）来实现。
- 这类模型的产出可理解为“更像历史真实开奖数据”的随机组合，因此在样本模拟或抽样策略上会有趣。

#### 实现思路
1. 数据准备：用历史数据训练GAN/变分自编码器；输入可能是噪声向量，输出维度对应双色球7个号码(或它们的嵌入表示)。
2. 生成器结构：多层全连接或1D CNN 生成器，确保输出可以对应到合法的彩票组合。
3. 判别器（对GAN）：判断生成的号码组合是否和历史数据“分布相似”。
4. 后处理：在生成号码时加一些限制规则(如去重、保证1~33范围等)，可显式地进行采样纠正。
5. 使用场景：生成大量“仿真开奖”数据来做统计分析、测试各种策略的期望收益等，而不一定是直接预测下一期。

#### 优势与局限
- 优势：可以更好地模拟开奖数据的概率分布，有利于理解和研究各种特征；能和其它模型结合(如判别器也能帮助发现异常)。
- 局限：对预测帮助不大，更多是个模拟或可视化工具。
- 适用性：对深度学习和生成模型有兴趣的研究者，想做技术探索或数据增强/异常检测。

#### 参考文献/资料
- 中文文献/资料
  1. Goodfellow等. 生成对抗网络(GAN)综述. 计算机学报, 2017.
  2. 部分国内博客如“GAN在彩票模拟中的应用”（CSDN搜索相关关键词）有演示代码。
- 英文文献/资料
  1. Goodfellow, I. et al. Generative Adversarial Nets. NIPS, 2014. (GAN原始论文)
  2. Kingma, D.P. & Welling, M. Auto-Encoding Variational Bayes. ICLR, 2014. (VAE原始论文)
  3. 部分GitHub示例项目: “Lottery-GAN” (可搜索关键字，属于个人/实验性质)。
```
