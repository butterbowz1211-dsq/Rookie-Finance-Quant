# 双语摘要

作者：**Gilmour Jia**  
语言：简体中文（zh-CN）与 English（en）  
用途：研究档案摘要；未指定投稿期刊，按作者要求不设字数上限。

## Chinese Abstract

**背景：** 将机器学习资产定价研究迁移至中国 A 股，需要同时核查模拟设计、历史信息可得性、收益标签和评价口径。模型分数本身不足以说明复现质量或投资可执行性。

**目的：** 本研究围绕 Gu、Kelly 与 Xiu 的研究框架，整理系统模拟、A 股收益预测与代码审计的证据，建立可追溯的研究基线。

**方法：** 历史蒙特卡洛实验涵盖线性与非线性数据生成过程、50 或 100 个基础特征，以及变量选择和重要性分析。研究按偏最小二乘（PLS）、主成分回归、弹性网、随机森林、梯度提升回归树（GBRT）和神经网络六类方法组织，并保留与原文结果的偏离记录。修订实证采用 19 个行情、股本和分红相关特征，以未缩尾的下一自然月总收益为标签，按目标月划分训练、验证和测试样本。六类方法加上普通最小二乘、岭回归和 Lasso 对照，共九种估计器在相同的 212,293 个测试股票月上计分，覆盖 2022—2025 年的 48 个月；神经网络采用三隐层变体（NN3）。预测误差、逐月截面相关性和组合毛收益分别评价。

**结果：** 修订实验中，零预测基准的合并样本测试 R² 分别为 PLS 1.1424%、岭回归 1.0902% 和 NN3 0.8862%。历史 NN3 的 1.0177% 来自 152 维输入，与修订后的 19 特征实验不构成同条件修复前后对照。GBRT 在 47 个测试月内给出截面常数预测，其相关系数仅在 1/48 个月有定义，不能据此判断稳定排序表现。冻结特征构建重放得到逐字节一致的模型输入，九模型指标经另一步骤重算相符，36 项回归检查通过；这些检查验证所记录的计算关系，不认证第三方数据的完整性。

**结论：** 本研究形成了系统模拟、历史结果分层和修订实证相互对应的研究档案，贡献主要在可审计流程与复核依据，不在提出新的模型架构。当前模型比较限于已反复查看的历史测试期，未完成差异显著性检验；财报历史版本、退市覆盖及部分公司行动仍待核查。无成本组合诊断也未覆盖完整交易约束，因此不能推导可实现净收益或普遍的模型优劣。

**关键词：** 蒙特卡洛实验；样本外评价；信息时点；模型比较；截面排序；计算可复现性

## English Abstract

**Background:** Applying machine learning to Chinese A-share return prediction requires scrutiny of data timing, target construction, and evaluation as well as model performance. A reported prediction score alone establishes neither replication quality nor implementable investment returns.

**Objective:** This study develops an auditable research baseline around the framework of Gu, Kelly, and Xiu by bringing together simulation experiments, A-share applications, and a documented review of the research pipeline.

**Methods:** The historical Monte Carlo archive covers linear and nonlinear data-generating processes with 50 or 100 base characteristics, including variable-selection and importance analyses. Six method families organize the comparisons: partial least squares (PLS), principal component regression, elastic net, random forests, gradient-boosted regression trees (GBRT), and neural networks. Departures from the original simulation results are retained. The repaired empirical experiment used 19 market, share-capital, and dividend characteristics to predict unwinsorized total returns for the next calendar month, with training, validation, and test partitions defined by target month. Adding ordinary least squares, ridge, and lasso controls yielded nine estimators, including a three-hidden-layer neural network (NN3). Every estimator was evaluated on the same 212,293 stock-month observations across 48 test months in 2022–2025. Prediction errors, monthly cross-sectional correlations, and gross portfolio outcomes were assessed separately.

**Results:** Pooled test R² relative to a zero forecast was 1.1424% for PLS, 1.0902% for ridge, and 0.8862% for NN3. The archived NN3 result of 1.0177% used 152 input dimensions and is not a controlled before-and-after comparison with the repaired 19-characteristic experiment. GBRT produced constant cross-sectional forecasts in 47 months; correlations were defined in only 1 of 48 months and cannot establish persistent ranking performance. Replaying the frozen feature build reproduced the model input byte for byte, a separate aggregation reproduced all nine reported scores, and 36 regression checks passed. These checks substantiate the recorded computational links rather than the completeness of third-party source data.

**Conclusions:** The contribution is a structured simulation archive and a traceable empirical audit, not a new model architecture. Comparisons remain descriptive: the historical test period has been examined repeatedly, and statistical significance of model differences has not been established. Financial-data vintages, delisting coverage, and some corporate actions require further verification. Gross portfolio diagnostics omit full trading constraints, so the evidence supports neither claims of attainable net profits nor universal model rankings.

**Keywords:** Monte Carlo experiments; out-of-sample evaluation; information timing; model comparison; cross-sectional ranking; computational reproducibility

## Abstract Quality Report

| 核对项 | 中文 | English |
|---|---|---|
| 篇幅（不含段首标签和关键词） | 788 个非空白字符，其中汉字 620 个；混合文本按字符报告 | 369 words, counted by whitespace splitting |
| 五要素覆盖 | 背景、目的、方法、结果、结论 | Background, objective, methods, results, conclusions |
| 关键词数量 | 6 | 6 |
| 独立写作 | 按中文论述习惯独立组织 | Independently composed in English |
| 事实与数量对齐 | 六类方法、九估计器、19 特征、212,293 个测试股票月、48 个月、三项 R²、历史 152 维结果、GBRT 有效月数、36 项检查均与英文一致 | Same evidence scope and numerical claims as the Chinese abstract |
| 解释边界 | 反复查看的历史测试期、未作显著性检验、数据审计未闭合、无净收益结论 | Repeatedly examined historical test period; no significance tests; incomplete source-data audit; no net-profit claim |
| 独立成文可读性 | 缩写在本摘要首次出现时定义；无文献引注 | Acronyms defined within this abstract; no citations |

摘要依据 `evidence_inventory.md`、`repaired_results.md`、`data_availability.md` 与主 README 独立起草。历史模拟与新实证的证据状态分别保留；摘要不把文件哈希、回归检查或重放一致性表述为全部历史实验已复现或源数据已认证。
