# 实验登记表：历史聚合记录与修订后共同样本

作者：**Gilmour Jia**  
登记日期：2026-10-25

本表按已公开文件登记实验，不把文件个数等同于独立实验或统计重复。历史目录有 **47 个聚合文件：14 个 A 股 CSV 与 33 个模拟 CSV/JSON**；逐文件来源和 SHA-256 以 [历史 manifest](../results/historical/manifest.json) 为准。修订后九模型由两批运行完成，使用共同的 19 特征面板，不能记为两个独立数据实验。完整讨论见 [主报告](research_report.md)、[方法附录](methods_appendix.md)、[修订后结果](repaired_results.md)；字段定义见 [数据字典](data_dictionary.md)。

## 1. 登记规则与指标单位

| 状态 | 含义 | 不代表什么 |
|---|---|---|
| 历史聚合文件可核对 | 文件存在，可查模型和已保存统计，复制来源和哈希已登记 | 不代表当前源码已重现旧数字；不证明旧数据无前视或幸存者偏差 |
| 历史说明或日志记录 | 参数或结果仅在既有文字/日志中出现，未形成完整冻结运行包 | 不把说明补写成 CSV 原生字段，不补造种子与版本 |
| 修订后运行已完成 | 有当次指标、设置、代码和输入哈希的 manifest，另有公开月度聚合证据 | 不代表全部第三方数据已认证，或已通过净收益和显著性检验 |
| 本次未重跑/未提供 | 缺少可供本次验证的输出或运行绑定 | 不把待补工作写成已完成结果 |

A 股 CSV 中的 `IS/valid/OOS/OOS_dm/OOS_m` 与模拟 CSV 的 `is_mean/oos_mean/is_sd/oos_sd` 均以百分数数值记录 R²：`1.02` 表示 1.02%。模拟 `sd` 是重复间标准差，不是标准误。A.3 的 `six/noise` 是选择频率（0—1）；A.4 是归一化重要性（百分数），其中 `noise` 为每个噪声变量的平均重要性，不能直接与六列相加要求总和为 100。

## 2. 模拟实验的共同设置和版本边界

模拟四格为 `(a)/(b) × Pc=50/100`，源代码固定 N=200、T=180，基础特征与宏观交互形成 `P=2Pc=100/200` 列。一月标签版本按连续 60/60/60 期训练、验证、测试；用验证 MSE 选超参数，R² 相对于训练收益均值。六大家族 PLS/PCR/Elastic Net/RF/GBRT/NN 与 Ridge/Lasso、OLS、稳健损失变体一起登记；Oracle 只在已知真 DGP 的模拟中使用。来源：[dgp.py](../research/ML金融学习计划/A_蒙特卡洛/dgp.py)、[run_mc.py](../research/ML金融学习计划/A_蒙特卡洛/run_mc.py)、[models.py](../research/ML金融学习计划/A_蒙特卡洛/models.py)、[nn.py](../research/ML金融学习计划/A_蒙特卡洛/nn.py)。

| 实验组 | 已保存内容 | 重复次数证据 | 必须保留的缺失/限制 |
|---|---|---|---|
| A.1 四格线性+Oracle | 每格11行：OLS/OLS+H、Ridge/Ridge+H、Lasso/Lasso+H、ENet/ENet+H、PCR、PLS、Oracle | 聚合 CSV 本身无 `n`；旧说明称100次 | 旧运行 commit、环境、逐种子参数与全部配置未绑定；不能仅用文件名证明100个独立重复 |
| A.1 四格树 | 每格 RF、GBRT、GBRT+H | CSV 每行 `n=100` | Pc50(a) 文件缺 `is_sd`；历史 GBRT+H 可能沿用旧 alpha 含义，当前修订不能追溯覆盖旧表 |
| A.1 四格 NN | 每格 NN1—NN5 | CSV 每行 `n=100` | 未保存各重复的全部集成 seed、epoch 和最佳网络权重；不同深度不等于 A 股也已全部比较 |
| GLM 四格及 pc50(a) i400 | GLM、GLM+H | CSV 无 `n`；旧说明称主比较20次 | 两个 pc50(a) 文件数值相同，不能计为独立重复；历史权重算法已修订，旧表未重跑 |
| h=3/6/12 线性+Oracle | 三个期限各11行 | CSV 无 `n`，不能用当前默认值补填 | 历史长周期目标是未来月收益求和；历史切分未清除边界重叠目标。当前代码的 purge 修复不使旧结果自动变为无重叠实验 |
| h=12 树 | RF/GBRT/GBRT+H三行 | CSV 无 `n`；旧续跑说明记10次 | 与主四格100次不可合并解释；参数与 horizon 边界风险同上 |
| A.3 四格选择频率 | 六个关注变量及其余变量平均选择频率 | JSON 无原生重复次数字段；各模型可能不同 | 非线性(b)两格仅含 Lasso/ENet 及其 Huber 变体，不能补称 GLM 也已在文件中；重复计数需原始记录 |
| A.4 四格×树/NN | 树 MDI、NN SSD 归一化重要性 | JSON 无原生重复次数字段 | NN SSD全零网络被跳过；应另报死亡网络和有效次数，不能由归一化表推断全部网络都有效 |

旧模拟说明常记 `theta-scale=1.08`，即(a)的0.02乘为0.0216；当前不同入口默认值并不完全相同。旧说明还把因子载荷描述为逐期独立正态，而本次所读 `dgp.py` 使用前三个公司特征作载荷。由于没有逐次历史配置与代码提交的冻结绑定，本登记不消除这类版本差异，也不把为接近某个 Oracle 数值调整信号尺度称作原文逐字复现。原文与当前实现的对应细节见方法附录。

代表性历史模拟 OOS R²：Pc50(a) Ridge=3.9697%、RF=3.6488%、GBRT=3.7470%、NN3=4.0974%、Oracle=5.5409%；Pc100(b) RF=3.2331%、GBRT=3.4236%、NN3=0.4101%、Oracle=6.2120%。这些数字来自本节对应 CSV；这里只在模拟内部比较，不拿来充当美国真实股票市场结果。

## 3. 14 个 A 股历史文件的版本登记

历史基础特征面板盘点为628,113行、5,080只股票、191个形成月。旧入口按**特征月**划分：训练2010-01—2018-12（257,218行），验证2019-01—2021-12（138,507行），测试2022-01—2025-11（232,388行）。这组规模描述的是被核对的旧主面板，不能保证每个早期 CSV 的逐行样本完全相同。旧标签 `ret_fwd` 使用历史收益清洗链，存在全样本缩尾、缺月后跳至下一观测和按形成月切分的边界风险。文件本身通常不保存逐股预测、目标月份和入样明细。来源：[证据清单](evidence_inventory.md)、[主张清单](claims_inventory.json)。

| 文件 | 特征/版本 | 已保存模型 | 关键结果或作用 | 仍缺少的运行绑定 |
|---|---|---|---|---|
| [ashare_linear.csv](../results/historical/ashare/ashare_linear.csv) | 19行情公司特征 | OLS、Ridge、Lasso、ENet、PCR、PLS | Ridge OOS 0.9497564%；PLS 0.9416704% | CSV无n_feat、split、seed和旧commit；19来自说明与代码路径 |
| [ashare_trees.csv](../results/historical/ashare/ashare_trees.csv) | 19公司特征 | RF、GBRT | OOS 0.2963593%、−0.9246525% | `grid=None`，不是已记录的最佳参数 |
| [ashare_fund.csv](../results/historical/ashare/ashare_fund.csv) | 19行情+9财务=28公司特征 | 6线性/降维模型、RF、GBRT | RF 0.5859499%，GBRT −0.3175973% | 树参数缺失；财务vintage未认证；不能将变化全部归因为财务信息 |
| [ashare_fundsmoke.csv](../results/historical/ashare/ashare_fundsmoke.csv) | 28特征早期检查 | OLS、Lasso | 两行与fund文件对应结果相同 | 不是独立样本或额外重复；完整执行日志未绑定 |
| [ashare_ind28.csv](../results/historical/ashare/ashare_ind28.csv) | 28公司特征+行业项 | 6线性/降维模型 | Ridge 0.9613797% | CSV未记行业列数和词表，不能凭名称补填总维度 |
| [ashare_v2_lin71.csv](../results/historical/ashare/ashare_v2_lin71.csv) | 71公司特征+82行业列=153 | 6线性/降维模型 | ENet 1.0107934%；OLS −7.9630852% | 不同于后续152列；求解收敛/设计病态需逐运行日志 |
| [ashare_v2_lin71b.csv](../results/historical/ashare/ashare_v2_lin71b.csv) | 152列 | Ridge、OLS | 0.9945426%、−7.9857707% | seed/环境/历史commit无绑定 |
| [ashare_v2_lin71c.csv](../results/historical/ashare/ashare_v2_lin71c.csv) | 152列，增加OOS_m | Ridge、OLS | pooled与b相同；Ridge等月1.1154889% | 增加指标列不等于独立实验 |
| [ashare_v2_lin71e.csv](../results/historical/ashare/ashare_v2_lin71e.csv) | 152列 | PCR、PLS | OOS 0.8435905%、0.8299683%；成分31、3 | 不是153列PCR/PLS的同一个输出 |
| [ashare_v2_nn71.csv](../results/historical/ashare/ashare_v2_nn71.csv) | 71公司特征+81行业列=152 | NN3（32–16–8） | IS 2.6949173%；valid 1.3848199%；OOS **1.0177149183600132%**；OOS_dm 0.4137724%；L1=1e−4、lr=0.1 | 旧说明记3种子/最多60 epoch；CSV无逐seed、最佳epoch与冻结代码，不能补写为确定的历史manifest |
| [ashare_v2_repro_check.csv](../results/historical/ashare/ashare_v2_repro_check.csv) | 152列 | Ridge | OOS 0.9945424456%，与既有Ridge近似 | 局部历史数值检查不认证全部数据、NN或本次修订 |
| [ashare_v2_rf71_leaf100.csv](../results/historical/ashare/ashare_v2_rf71_leaf100.csv) | 152列；说明记叶100/树100 | RF | OOS 0.5787500%；valid 0.6459882% | `param=None`；叶规模实验还改变树数，不是单因素效应 |
| [ashare_v2_tree71.csv](../results/historical/ashare/ashare_v2_tree71.csv) | 153列 | RF、GBRT | OOS 0.5501074%、−1.4689274% | `param=None`；旧说明RF叶5/树200，须与原运行配置区分 |
| [ashare_v3_fund56.csv](../results/historical/ashare/ashare_v3_fund56.csv) | 19行情+56财务=75公司特征，另行业共156列 | 6线性/降维模型 | Lasso OOS 1.0517%、valid1.0331%、λ0.0005、49非零；ENet1.0445% | 不是“71再加56”；该文件不含NN/RF/GBRT；不同精度及数据版本不可拼成同一排行榜 |

OLS-3 的 size/bm/mom12m 比较、部分宏观交互失败诊断、Ridge历史等权十分位多空夏普约2.41，只在旧说明或日志中有记录，未列入47个冻结聚合文件。不得把这些记录伪装为本发布提供了完整预测/持仓/收益账本；历史Ridge组合也不是NN3策略结果。`R²_demeaned`应称测试样本中心化指标，既不是可实施历史均值预测，也不是截面排序相关系数。

## 4. 修订后九模型共同样本

本次新实验使用19公司特征，不含行业、财务和宏观交互，标签是未经缩尾的下一自然月总收益。特征月资格先确定，排名再进行，最后才筛选有观测标签的拟合/计分行。测试按目标月2022-01—2025-12分割，共48个月；它不是旧47个形成月测试窗的原样重复。两批运行共享输入SHA-256 `0a993bef6247fff10db46ace0798bccd1898a55562a99a6e49d881eba8ec0e06`，266,553,972字节；原始数据的权利和质量边界仍适用。

| 分组 | 配置目标月份 | 实际有效目标月份 | 股票月数 | 月数 |
|---|---|---|---:|---:|
| 训练 | 2010-01—2018-12 | 2012-01—2018-12 | 178,488 | 84 |
| 验证 | 2019-01—2021-12 | 2019-01—2021-12 | 119,719 | 36 |
| 测试 | 2022-01—2025-12 | 2022-01—2025-12 | 212,293 | 48 |
| 合计 | — | 2012-01—2025-12 | 510,500 | 168 |

构建面板为639,255行、5,210股票、192形成月，其中517,152行满足形成时资格；最终有标签建模样本为510,500行、5,070股票。不能把构建规模写成训练规模。24个月历史门槛使有效训练目标从2012-01起；配置中的2010是窗口起点，不是实际首个可训练标签。

| 运行标签 | 文件与manifest | 模型 | 状态 |
|---|---|---|---|
| price19_repaired | [指标](../results/repaired/ashare_price19_repaired.csv)、[manifest](../results/repaired/manifest_price19_repaired.json) | Ridge、ENet、PLS、RF、GBRT | 已完成；共同样本核心批 |
| price19_repaired_extra | [指标](../results/repaired/ashare_price19_repaired_extra.csv)、[manifest](../results/repaired/manifest_price19_repaired_extra.json) | OLS、Lasso、PCR、NN3 | 已完成；同一面板补充批 |

本次保留四种测试R²：pooled零预测基准、训练均值基准、测试样本中心化以及逐月R²等权平均；主表下方将逐字提取各模型已保存设置。指标之间不可互换。两批预测在本地按键合并并重新聚合后，与原CSV的pooled R²差异小于预设容差，证据见 [pooled_metrics_recomputed.csv](../results/repaired/monthly_summary/pooled_metrics_recomputed.csv)。逐股预测未公开，公开的是月度聚合；本地复核不等于第三方已独立重现。

## 5. 补充产物、环境和缺失记录

| 产物 | 证据路径 | 已完成范围与边界 |
|---|---|---|
| 冻结特征构建 | [frozen_build.json](../results/repaired/frozen_build.json)、[replay_comparison.json](../results/repaired/replay_comparison.json) | 冻结源码前后哈希一致，重建文件字节相同；不证明原始源数据无误 |
| 九模型逐月评价 | [monthly_metrics.csv](../results/repaired/monthly_summary/monthly_metrics.csv) | 逐月误差和IC/RankIC；需同时报告有效相关系数月份 |
| Ridge组合 | [ridge_portfolio](../results/repaired/ridge_portfolio/) | 48月等权十分位毛收益诊断，未含成本/借券/成交模型 |
| NN3组合 | [nn3_portfolio](../results/repaired/nn3_portfolio/) | 与Ridge同样口径；不能与旧历史夏普混淆 |
| 针对性测试 | [test_log.txt](test_log.txt) | 36项通过；包括数据、模型、评价、组合和模拟运行管理，不是36项经济假说 |

GBRT在48个测试月中有47个月为全截面常数预测，只有2025-05的IC/RankIC有定义。因此其RankIC −0.1014是 **1/48个月** 的读数，不能描述为48月稳定负排序。其pooled R²仍使用全部测试股票月，不能用该指标为正推断每月都有排序信息。

冻结特征构建环境：Python3.12.14、NumPy2.3.5、pandas3.0.1。九模型训练环境：Python3.12.14、NumPy2.5.4、pandas3.0.6、scikit-learn1.9.1。不同步骤的环境分别登记，不能合并成一个未经证实的统一环境。NN本次调参种子0/1，最终集成100/101/102，三个最佳epoch均为2，最多60 epoch；这是本次新实验的记录，不能倒填进旧1.0177%实验。

仍缺少：全部历史运行的commit/环境/原始配置与逐种子输出绑定；可信财报原始vintage；完整历史上市退市池和退市终止收益；配股及公司行动逐案审计；所有阶段的可交易约束；预先冻结的新留出期与模型差异的依赖稳健统计推断。修订后19特征与历史152/156列同时改变特征、样本、标签和切分，分数差不能直接解释为单项修复的因果效应。

## 6. 文件级索引与原生记录

下表按历史manifest逐文件建立索引。数字、列名和哈希是文件记录；缺失明确写“未记录”，不从当前默认配置反推历史运行。随后附本次九模型CSV原生指标及参数，便于追踪。


### 6.1 历史47文件索引

以下全部状态为 historical_not_rerun_after_repairs。模型行数不是重复次数；SHA-256直接取自manifest并在生成本表时逐文件核对。

| 序号 | 文件 | 格子/维度 | 模型行数与名称 | 原生n | SHA-256 |
|---:|---|---|---|---|---|
| 1 | [ashare_fund.csv](../results/historical/ashare/ashare_fund.csv) | 见第3节；CSV无n_feat | 8：Ridge, PLS, ENet, OLS, Lasso, PCR, RF, GBRT | 未记录 | 116e1d259dd961533c8c4069127bd226a1d8dcba94714b88754e41a3a7eb6a56 |
| 2 | [ashare_fundsmoke.csv](../results/historical/ashare/ashare_fundsmoke.csv) | 见第3节；CSV无n_feat | 2：OLS, Lasso | 未记录 | 71732f9d97c14ea4cb7213b636a31626eaf7f5d34b5f91fa11f7d9a0c1f96525 |
| 3 | [ashare_ind28.csv](../results/historical/ashare/ashare_ind28.csv) | 见第3节；CSV无n_feat | 6：Ridge, PLS, ENet, OLS, Lasso, PCR | 未记录 | 6a9f32a718334a9ab80f8dc32e4f746fa95c4479ad33f93efbb68a282ec026a6 |
| 4 | [ashare_linear.csv](../results/historical/ashare/ashare_linear.csv) | 见第3节；CSV无n_feat | 6：Ridge, PLS, OLS, ENet, Lasso, PCR | 未记录 | 6461fc9cb18c5816434eed40ec0e1e757b8c28010f5d975dad07238c87d08092 |
| 5 | [ashare_trees.csv](../results/historical/ashare/ashare_trees.csv) | 见第3节；CSV无n_feat | 2：RF, GBRT | 未记录 | f46534eef2587e7e573a2e8c3214da87831622a48b93d94da6ee93f47c73098f |
| 6 | [ashare_v2_lin71.csv](../results/historical/ashare/ashare_v2_lin71.csv) | 153 列 | 6：ENet, Ridge, Lasso, PCR, PLS, OLS | 未记录 | 1bc353794dcaa1fcec98a4dc0627fbaa4f9b55de1f7f18180d92496641cb40f7 |
| 7 | [ashare_v2_lin71b.csv](../results/historical/ashare/ashare_v2_lin71b.csv) | 152 列 | 2：Ridge, OLS | 未记录 | fe6e8037075dc83c36811aa1906e44b06da7c6755d0ebbf9001816e963c949c4 |
| 8 | [ashare_v2_lin71c.csv](../results/historical/ashare/ashare_v2_lin71c.csv) | 152 列 | 2：Ridge, OLS | 未记录 | 56da5bb88a921b4f08c50cf01150a83e62212babdba3be83662efe53b53cb0ce |
| 9 | [ashare_v2_lin71e.csv](../results/historical/ashare/ashare_v2_lin71e.csv) | 152 列 | 2：PCR, PLS | 未记录 | df56a64d1fc048c9336696b293a69fd139b5dc28d2107cdcddc0d242ab8f6c46 |
| 10 | [ashare_v2_nn71.csv](../results/historical/ashare/ashare_v2_nn71.csv) | 152 列 | 1：NN3 | 未记录 | 9fbee2aeb33a52d6189124b269b96e78fc92f2a788a8b1b2fbc54cd7dfb13af6 |
| 11 | [ashare_v2_repro_check.csv](../results/historical/ashare/ashare_v2_repro_check.csv) | 152 列 | 1：Ridge | 未记录 | 44a7c8dc3572f93437afc16070e2d8ff8605a964e8b5cccea3b385b763fcbb88 |
| 12 | [ashare_v2_rf71_leaf100.csv](../results/historical/ashare/ashare_v2_rf71_leaf100.csv) | 152 列 | 1：RF | 未记录 | b6b0ac3d7f22555914c9af2a03c5418ac66633acddfa44a34872e2513250362e |
| 13 | [ashare_v2_tree71.csv](../results/historical/ashare/ashare_v2_tree71.csv) | 153 列 | 2：RF, GBRT | 未记录 | 07be1df8a1e2201c73559fcd3bda06a85d71e20daadf3d8f252c1f6639818572 |
| 14 | [ashare_v3_fund56.csv](../results/historical/ashare/ashare_v3_fund56.csv) | 156 列 | 6：OLS, Ridge, Lasso, ENet, PCR, PLS | 未记录 | 2f9426ae27060517105b37ce88eb5007d1429e6b54598cdc69464c975e4b678f |
| 15 | [tableA1_A1_NN_pc100_a.csv](../results/historical/monte_carlo/tableA1_A1_NN_pc100_a.csv) | Pc=100，DGP=a，P=200 | 5：NN1, NN2, NN3, NN4, NN5 | 100 | d258aac3efe79c988209a2c6c131ecdd1a4c1a17b87fe809ace7dd8971f017a7 |
| 16 | [tableA1_A1_NN_pc100_b.csv](../results/historical/monte_carlo/tableA1_A1_NN_pc100_b.csv) | Pc=100，DGP=b，P=200 | 5：NN1, NN2, NN3, NN4, NN5 | 100 | 0692766a811108bfb6f078cd74bd2f42e76bdebac3db27107dce76b90a2a5706 |
| 17 | [tableA1_A1_NN_pc50_a.csv](../results/historical/monte_carlo/tableA1_A1_NN_pc50_a.csv) | Pc=50，DGP=a，P=100 | 5：NN1, NN2, NN3, NN4, NN5 | 100 | 00fba7f03c122d059d5c87ba48f8368bfdd225e42c5174e3bc582216001a6107 |
| 18 | [tableA1_A1_NN_pc50_b.csv](../results/historical/monte_carlo/tableA1_A1_NN_pc50_b.csv) | Pc=50，DGP=b，P=100 | 5：NN1, NN2, NN3, NN4, NN5 | 100 | 986b147017eb20e3289d5dc853f65cd8fda6c20c8ae0f7ce2d2d21353e768f53 |
| 19 | [tableA1_A1_pc100_a.csv](../results/historical/monte_carlo/tableA1_A1_pc100_a.csv) | Pc=100，DGP=a，P=200 | 11：OLS, OLS+H, Ridge, Ridge+H, Lasso, Lasso+H, ENet, ENet+H, PCR, PLS, Oracle | 未记录 | ed17603fbc542679b4ba3a3027ab75ba335770731c509ddedcfcf37dc1e6f4ed |
| 20 | [tableA1_A1_pc100_b.csv](../results/historical/monte_carlo/tableA1_A1_pc100_b.csv) | Pc=100，DGP=b，P=200 | 11：OLS, OLS+H, Ridge, Ridge+H, Lasso, Lasso+H, ENet, ENet+H, PCR, PLS, Oracle | 未记录 | 69fbde6d242aad803b331da4fbcfe92769cd53bbe4268a7da434d9dfe943b2fd |
| 21 | [tableA1_A1_pc50_a.csv](../results/historical/monte_carlo/tableA1_A1_pc50_a.csv) | Pc=50，DGP=a，P=100 | 11：OLS, OLS+H, Ridge, Ridge+H, Lasso, Lasso+H, ENet, ENet+H, PCR, PLS, Oracle | 未记录 | 24244156ae0b92e93559fcaf14467b2af68c5d61d65dcdc2686f1e5de1418e52 |
| 22 | [tableA1_A1_pc50_b.csv](../results/historical/monte_carlo/tableA1_A1_pc50_b.csv) | Pc=50，DGP=b，P=100 | 11：OLS, OLS+H, Ridge, Ridge+H, Lasso, Lasso+H, ENet, ENet+H, PCR, PLS, Oracle | 未记录 | 1110057dac1ec77fecdd06fc280d851560b2fbde0b3f124634d40f337f799d08 |
| 23 | [tableA1_A1_TREE_pc100_a.csv](../results/historical/monte_carlo/tableA1_A1_TREE_pc100_a.csv) | Pc=100，DGP=a，P=200 | 3：RF, GBRT, GBRT+H | 100 | 41a5bed20d40efeb4d67d1cd96aef9c680b6af2ff36593636a83b9044c055881 |
| 24 | [tableA1_A1_TREE_pc100_b.csv](../results/historical/monte_carlo/tableA1_A1_TREE_pc100_b.csv) | Pc=100，DGP=b，P=200 | 3：RF, GBRT, GBRT+H | 100 | 666ad284a5ae4f701715546fb031c4c260ce9a9126aff2190c6b8514475af7c1 |
| 25 | [tableA1_A1_TREE_pc50_a.csv](../results/historical/monte_carlo/tableA1_A1_TREE_pc50_a.csv) | Pc=50，DGP=a，P=100 | 3：RF, GBRT, GBRT+H | 100 | 43350d6db5e8a387ad1a43435977b09e2624c08aea7ee90ab2e91ecd50068afb |
| 26 | [tableA1_A1_TREE_pc50_b.csv](../results/historical/monte_carlo/tableA1_A1_TREE_pc50_b.csv) | Pc=50，DGP=b，P=100 | 3：RF, GBRT, GBRT+H | 100 | 37389e4318dcb0fda92bfefc679f4a888fc067b903e4f213e4906120546f0828 |
| 27 | [tableA1_GLM_pc100_a.csv](../results/historical/monte_carlo/tableA1_GLM_pc100_a.csv) | Pc=100，DGP=a，P=200 | 2：GLM, GLM+H | 未记录 | f6a0073016f23a82ce4b63d2ad3587fd767480d720604c474cfdebbbeef5b241 |
| 28 | [tableA1_GLM_pc100_b.csv](../results/historical/monte_carlo/tableA1_GLM_pc100_b.csv) | Pc=100，DGP=b，P=200 | 2：GLM, GLM+H | 未记录 | ab40f8cb742d4fdcc72c7a9e69ddd8cb0f6e9ff1a2bf6b7edb3796cabeed02f2 |
| 29 | [tableA1_GLM_pc50_a.csv](../results/historical/monte_carlo/tableA1_GLM_pc50_a.csv) | Pc=50，DGP=a，P=100 | 2：GLM, GLM+H | 未记录 | 1b6e11d7d32bbdb23af31e4e639149d883c2437257ff879ae501fa18f4ad8839 |
| 30 | [tableA1_GLM_pc50_a_i400.csv](../results/historical/monte_carlo/tableA1_GLM_pc50_a_i400.csv) | Pc=50，DGP=a，P=100 | 2：GLM, GLM+H | 未记录 | 1b6e11d7d32bbdb23af31e4e639149d883c2437257ff879ae501fa18f4ad8839 |
| 31 | [tableA1_GLM_pc50_b.csv](../results/historical/monte_carlo/tableA1_GLM_pc50_b.csv) | Pc=50，DGP=b，P=100 | 2：GLM, GLM+H | 未记录 | 2c318cfb96d7f13b4e804476b106ac2a20a3e2cc6e0ad16cc96d9abf5bcf00ba |
| 32 | [tableA1_h12_all.csv](../results/historical/monte_carlo/tableA1_h12_all.csv) | 原文件未编码 | 11：OLS, OLS+H, Ridge, Ridge+H, Lasso, Lasso+H, ENet, ENet+H, PCR, PLS, Oracle | 未记录 | 9a6065f752e6ad858b40a986e1f9efa027c9316d27a3ea0af68a4d5a4aa8a9bf |
| 33 | [tableA1_h12_tree.csv](../results/historical/monte_carlo/tableA1_h12_tree.csv) | 原文件未编码 | 3：RF, GBRT, GBRT+H | 未记录 | 3d804fc64a7c4730f34296df90023f8554521dec962e4b2fb62109e215177e2d |
| 34 | [tableA1_h3_all.csv](../results/historical/monte_carlo/tableA1_h3_all.csv) | 原文件未编码 | 11：OLS, OLS+H, Ridge, Ridge+H, Lasso, Lasso+H, ENet, ENet+H, PCR, PLS, Oracle | 未记录 | eb12056f837fc353479577cdb7cf21739bda87d7b5216f043b8c1088b53ea6d9 |
| 35 | [tableA1_h6_all.csv](../results/historical/monte_carlo/tableA1_h6_all.csv) | 原文件未编码 | 11：OLS, OLS+H, Ridge, Ridge+H, Lasso, Lasso+H, ENet, ENet+H, PCR, PLS, Oracle | 未记录 | d377a95157f0a0767d6c32c2daf338f2a75885b5fb7ae31d5d4ed253be91b635 |
| 36 | [tableA3_pc100_a.json](../results/historical/monte_carlo/tableA3_pc100_a.json) | Pc=100，DGP=a，P=200 | 6：Lasso, Lasso+H, ENet, ENet+H, GLM, GLM+H | 未记录 | 20dfe3d7f445d017b66e1271e851cd8eb7436511f08dd6639c2689a033f0332b |
| 37 | [tableA3_pc100_b.json](../results/historical/monte_carlo/tableA3_pc100_b.json) | Pc=100，DGP=b，P=200 | 4：Lasso, Lasso+H, ENet, ENet+H | 未记录 | 80c6e3fc6e6bea6b3615e50fcf758c278a77a325b1bb6231653c43afc4e669e9 |
| 38 | [tableA3_pc50_a.json](../results/historical/monte_carlo/tableA3_pc50_a.json) | Pc=50，DGP=a，P=100 | 6：Lasso, Lasso+H, ENet, ENet+H, GLM, GLM+H | 未记录 | 03c3bdc3c2e77a89e037e1f931b3fc4bb7f31d21694de4499e6198d6059eb264 |
| 39 | [tableA3_pc50_b.json](../results/historical/monte_carlo/tableA3_pc50_b.json) | Pc=50，DGP=b，P=100 | 4：Lasso, Lasso+H, ENet, ENet+H | 未记录 | 45d0079ac30ab2fd5d7c1eaf40c0c1c4ffa25346ea3045a9767c3999c44c991e |
| 40 | [tableA4_pc100_a_nn.json](../results/historical/monte_carlo/tableA4_pc100_a_nn.json) | Pc=100，DGP=a，P=200 | 5：NN1, NN2, NN3, NN4, NN5 | 未记录 | 464b4307a8f88beae9f8c4582fd093c03333910748b79222e124fa100c420958 |
| 41 | [tableA4_pc100_a_tree.json](../results/historical/monte_carlo/tableA4_pc100_a_tree.json) | Pc=100，DGP=a，P=200 | 3：RF, GBRT, GBRT+H | 未记录 | 4db4d4951f911dfb142a7ad28b4f10d7b8bdc2b1b63d9e7576fee74975577d83 |
| 42 | [tableA4_pc100_b_nn.json](../results/historical/monte_carlo/tableA4_pc100_b_nn.json) | Pc=100，DGP=b，P=200 | 5：NN1, NN2, NN3, NN4, NN5 | 未记录 | b82796334b5807949dbc5c16008ef2613fdbd065705cdefb46a5cf9c22af8656 |
| 43 | [tableA4_pc100_b_tree.json](../results/historical/monte_carlo/tableA4_pc100_b_tree.json) | Pc=100，DGP=b，P=200 | 3：RF, GBRT, GBRT+H | 未记录 | 803470a9483f475a4471dbeec9998d156483f819d5e6898b2162787d2e66f86d |
| 44 | [tableA4_pc50_a_nn.json](../results/historical/monte_carlo/tableA4_pc50_a_nn.json) | Pc=50，DGP=a，P=100 | 5：NN1, NN2, NN3, NN4, NN5 | 未记录 | 25a84d45d6e2a6be4ce6a5ceaea6b54805593e0fbb281d63e68241b0fa41fc29 |
| 45 | [tableA4_pc50_a_tree.json](../results/historical/monte_carlo/tableA4_pc50_a_tree.json) | Pc=50，DGP=a，P=100 | 3：RF, GBRT, GBRT+H | 未记录 | 81e3a70ebbd4d62930bd3c30da1a9f75590f39ee958de549ceee737fb2b21b35 |
| 46 | [tableA4_pc50_b_nn.json](../results/historical/monte_carlo/tableA4_pc50_b_nn.json) | Pc=50，DGP=b，P=100 | 5：NN1, NN2, NN3, NN4, NN5 | 未记录 | 97d440b223913933f766cc9aabd48023b0d8b7ecb8f4f5e00d39566297532aaa |
| 47 | [tableA4_pc50_b_tree.json](../results/historical/monte_carlo/tableA4_pc50_b_tree.json) | Pc=50，DGP=b，P=100 | 3：RF, GBRT, GBRT+H | 未记录 | 71b8b01df2535708fd4fb7c922996239b8e5248bfa25366f9de365d211f890f5 |

### 6.2 修订后九模型的原生指标

数值按CSV记录转写，均为百分数；参数由CSV的parameters字段转写，不从模型名称猜测。两批使用共同样本，排序沿用原文件。

| 批次 | 模型 | 输入列 | IS | valid | OOS零基准 | OOS训练均值 | OOS测试均值 | OOS等月 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| price19_repaired | ridge | 19 | 0.9571843 | 1.5023055 | 1.0901553 | 0.4554800 | 0.4431619 | 0.5091173 |
| price19_repaired | enet | 19 | 0.6092933 | 1.6005430 | 0.7382411 | 0.1013077 | 0.0889457 | -0.0632120 |
| price19_repaired | pls | 19 | 0.9450597 | 1.4005814 | 1.1423720 | 0.5080318 | 0.4957202 | 0.5956636 |
| price19_repaired | rf | 19 | 2.0432540 | 1.6424480 | 0.7253488 | 0.0883327 | 0.0759691 | -0.1547718 |
| price19_repaired | gbrt | 19 | 2.2361006 | 0.1229175 | 0.1487643 | -0.4919516 | -0.5043870 | 0.0650222 |
| price19_repaired_extra | ols | 19 | 1.0193737 | 1.3692628 | 1.0678980 | 0.4330799 | 0.4207590 | 0.5259633 |
| price19_repaired_extra | lasso | 19 | 0.7583127 | 1.5911013 | 0.8740134 | 0.2379512 | 0.2256062 | 0.1510938 |
| price19_repaired_extra | pcr | 19 | 0.6337712 | 1.5022871 | 0.7591300 | 0.1223307 | 0.1099713 | 0.0201541 |
| price19_repaired_extra | nn3 | 19 | 0.8759036 | 1.8688278 | 0.8862384 | 0.2502546 | 0.2379111 | -0.0703719 |

| 批次 | 模型 | 已保存参数JSON |
|---|---|---|
| price19_repaired | ridge | {"lam": 100000.0} |
| price19_repaired | enet | {"lam": 0.01, "nnz": 1} |
| price19_repaired | pls | {"k": 2} |
| price19_repaired | rf | {"depth": 1, "mfeat": 10, "best": {"depth": 1, "max_features": 10}, "n_trees": 100, "min_samples_leaf": 100, "seed": 0} |
| price19_repaired | gbrt | {"depth": 1, "lr": 0.01, "ntree": 100, "best": {"depth": 1, "learning_rate": 0.01, "n_trees": 100}, "min_samples_leaf": 100, "seed": 0} |
| price19_repaired_extra | ols | {} |
| price19_repaired_extra | lasso | {"lam": 0.0031622776601683794, "nnz": 3} |
| price19_repaired_extra | pcr | {"k": 1} |
| price19_repaired_extra | nn3 | {"lam1": 0.001, "lr": 0.1, "ens": 3, "tuning_seeds": [0, 1], "ensemble_seeds": [100, 101, 102], "best_epochs": [2, 2, 2], "selection": "validation MSE of two-seed mean prediction"} |

两份运行manifest分别记录各自源码与输入哈希；确切代码绑定以manifest中的code_sha256为准，不默认等于读者当前工作树。对后续修改后的源码再运行，应使用新tag与新manifest。
