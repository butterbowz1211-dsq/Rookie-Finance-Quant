# 方法与实现附录

作者：**Gilmour Jia**  
核对日期：2026-10-25  
适用对象：本仓库修订代码、修订后的 19 特征 A 股实验，以及单独保存的蒙特卡洛实现。

发布审查补充：[19特征分段缺失统计](../results/repaired/feature_missingness/feature_missingness.csv)给出最终模型样本在排名前的有限数、缺失数与中性排名填补比例。训练/验证/测试分母分别为178,488/119,719/212,293；`mom36m`缺失比例为46.99%/23.12%/8.52%。这不是排名后“零值”的比例，真实中位排名不被当作缺失；排名参考截面还含当时合格但未来标签缺失的行，统计表则只描述最终估计与计分样本。脚本和输入绑定见[缺失统计manifest](../results/repaired/feature_missingness/manifest.json)。

构建器首部说明在审查后作了文字修订。实际运行版本保存于[executed_source/build_features.py](../results/repaired/executed_source/build_features.py)，用于哈希核查，不是新的独立执行入口；[source_annotation_revision.json](../results/repaired/source_annotation_revision.json)同时记录现行版本哈希及执行语句AST一致性。冻结构建与入口审计中的旧哈希仍指实际执行版本，不被篡改为新注释的哈希。

本附录回答“文件中的变量、模型、指标究竟如何计算，以及如何重新执行”。结果讨论见[修订实验报告](repaired_results.md)，历史文件及其证据等级见[证据清单](evidence_inventory.md)，不在这里重复完整成绩表。下文用“当前实现”描述所读源码，用“本轮执行”描述 manifest 或结果文件实际保存的设置，用“历史输出”描述 `results/historical/` 中的旧实验。三者不能互相替代。代码注释中对论文的概括不等同于已经逐项验证了原文定义；实现与 GKX 完全一致的主张还需要原文、原始数据和运行记录共同支持。

## A. 变量、样本及信息时点

### A.1 观测单位与文件契约

A 股观测单位是股票—形成月 `(code, month)`。`code` 为带 `sh/sz/bj` 前缀的字符串，`month` 为 `YYYY-MM`。特征形成月记为 $t$，标签月严格为 $t+1$。收益以小数保存，例如 `0.01` 为 1%；模型汇总 CSV 的 `IS/valid/OOS` 等 R² 列乘以 100 输出。文件字段与单位的完整说明见[数据字典](data_dictionary.md)。

修订后的 `features_monthly.csv` 共 29 列：2 个主键、19 个特征，以及 `ret, ret_fwd, label_month, ndays, close, market_cap, history_months, eligible_asof`。原始月收益表至少需要 `code,month,ret`；日线需要 `date,open,high,low,close,volume`；股本表需要 `SECURITY_CODE,END_DATE,TOTAL_SHARES,LISTED_A_SHARES`；分红表需要 `SECURITY_CODE,EX_DIVIDEND_DATE,PRETAX_BONUS_RMB,BONUS_IT_RATIO`。日线日期和股票月主键必须唯一，重复时直接报错，不在模型层任意保留一条。

[标签连接函数](../research/ML金融学习计划/D_A股数据/data_contract.py) `attach_forward_returns` 先构造 `label_month=Period(month)+1`，再按 `(code,label_month)` 左连接原始月收益。若该自然月不存在，则 `ret_fwd=NaN`，不会跳到再下一个有记录月份。runner 还检查每行 `label_month` 与预期自然月完全一致。原始 `ret`、目标 `ret_fwd` 和 `market_cap` 均不受特征缩尾或排名影响。

### A.2 日收益、月收益与公司行动

[收益代码](../research/ML金融学习计划/D_A股数据/build_returns.py) 以未复权收盘价 $P_d$ 构造日总收益。普通日期为 $P_d/P_{d-1}-1$；有现金派息 $D_d$ 和送转比例 $B_d$ 的除权日为

$$
r_d=\frac{(1+B_d)P_d+D_d}{P_{d-1}}-1,
\qquad
D_d=\frac{\texttt{PRETAX\_BONUS\_RMB}}{10},\quad
B_d=\frac{\texttt{BONUS\_IT\_RATIO}}{10}.
$$

月收益为同月可用日收益的 $\prod_d(1+r_d)-1$。日收益函数按价格记录排序，以前一条价格记录为基期；首条记录没有前收盘价，收益为缺失。除权调整只有在事件日与日线日期精确匹配、且前价格为有限正值时执行。因而缺失行情期间发生的事件、配股、重复方案及供应商记录遗漏不能由这条公式自动解决。

本轮特征 builder 读取既有 `returns_monthly.csv` 作为月收益与标签源，并调用 `daily_returns` 重算风险和流动性特征所需日收益；它没有借此证明整个既有月收益源已重新核验。builder 的分红加载器先删除完全重复源行，再按股票和除权日汇总；独立 `build_returns.py` 的加载器直接按股票和除权日求和。两条路径的源数据去重规则不同，复建原始月收益前应逐案检查分红记录。`build_returns.py` 也仍使用脚本目录下 `data/` 的路径，并未读取 `GKX_DATA_DIR`；本附录后面的环境变量命令从已合法准备的原始月收益缓存起步。

标签是**总收益**，没有减去无风险利率。builder 遇到原始月收益小于 −100% 会拒绝运行；保留下来的大正收益不因此获得真实性认证。`screen_returns.py` 的修订职责是增加当期资格及源错误标志，保留全部原始 `ret`。其可选 `ret_train` 只在显式训练区间内使用训练收益分位数缩尾，验证和测试期该列为空；默认模型不使用这列，也不使用历史 `returns_clean.csv` 作为标签源。

### A.3 样本资格、入口覆盖与实际切分

[价格特征构建器](../research/ML金融学习计划/D_A股数据/build_features.py) 逐股补齐首次至末次行情月份之间的自然月网格，缺月不填虚构收益。`history_months` 是截至形成月有日线记录的月份累计数；`eligible_asof` 要求该月有观测、`ndays>=10`、`history_months>=24`。两个阈值分别来自 `--min-days 10`、`--min-months 24`。`ndays` 是源日线记录条数，`history_months` 是观测历史，并非已核验的可成交天数或真实上市龄。

这一资格规则不使用下一月交易日或后续月份是否继续存在，但入口仍有覆盖条件：文件代码必须出现在原始月收益表中，且整份日线文件至少有 60 行（`if len(px)<60: continue`）。因此不能将本实现表述为已经消除所有按全文件覆盖筛股的问题。本次只读入口核查发现 5,237 份日线文件中 5,236 只有匹配月收益，26 只因 60 行条件被跳过；它们各有 2—54 日、仅覆盖 2025 年第四季度 1—3 个月，可满足 24 月历史门槛的股票月为 0。不过，资格过滤之前的特征缩尾可能间接受这些股票是否存在影响，不能声称删除该条件后数值必然不变。当前保留实际冻结源码和已执行结果；下一轮应先移除该入口条件、明确缩尾参考股票池，再完整重建和估计。核查记录见[数据修订记录末节](data_bug_fixes.md)。

特征输出只保留当月存在日线的行；股票池仍取决于已经取得的文件，尚未完整重建历史上市、退市及终止收益。当前 19 特征实验没有实现全面 ST 剔除、最小市值 30% 排除、或涨跌停/停牌可执行性过滤。

本轮冻结面板有 639,255 行、5,210 只股票和 192 个形成月（2010-01—2025-12）；其中当期合格 517,152 行，合格且下一自然月标签存在 510,500 行、5,070 只股票。名义训练窗口与实际有样本的窗口必须区分：

| 集合 | 配置的目标收益月 | 实际有效目标收益月 | 股票月数 | 有效月数 |
|---|---|---|---:|---:|
| 训练 | 2010-01—2018-12 | **2012-01—2018-12** | 178,488 | 84 |
| 验证 | 2019-01—2021-12 | 2019-01—2021-12 | 119,719 | 36 |
| 测试 | 2022-01—2025-12 | 2022-01—2025-12 | 212,293 | 48 |

实际训练起点推迟来自 24 个历史观测月门槛。测试形成月为 2021-12—2025-11。切分使用 `label_month`，避免 2018-12 形成的 2019-01 收益进入训练集合。模型仅用训练行拟合，验证集用于选择配置和 NN 早停；选择后没有合并训练与验证数据重新估计。该固定三段留出不是年度前推重估。测试期此前已被反复查看，也不是新冻结且从未查看过的留出期。配置和样本数绑定于两份[核心运行 manifest](../results/repaired/manifest_price19_repaired.json)、[补充运行 manifest](../results/repaired/manifest_price19_repaired_extra.json)；实际有效月范围的读取结果见[实验报告 §2](repaired_results.md)。

### A.4 19 个特征的实际计算式

以下窗口均含形成月 $t$，并以自然月网格为基础。日频特征使用上述调整后日收益。股本取不晚于该月最后有行情日的最新 `END_DATE` 记录；这一“有效日期回溯匹配”不等于已证明该股本记录在当时首次公开。

| 特征 | 当前代码计算式与最少观测 |
|---|---|
| `mvel1` | `log(close × TOTAL_SHARES)`；独立保存未变换的 `market_cap`；非正股数置缺失 |
| `turn` | 月成交量合计 / 月末 `LISTED_A_SHARES`；分母须为正 |
| `std_turn` | 最近 3 个自然月 `turn` 的样本标准差，至少 2 个有效月 |
| `dolvol` | 先取 `log(月日均(close×volume))`，再取最近 3 自然月均值，至少 1 个有效月；零交易额置缺失 |
| `ill` | 当月日均 `abs(r_d)×10^9/(close×volume)`；零交易额置缺失 |
| `zerotrade` | `1−ndays/当月源日历交易日数`；日历取所读全部日线文件日期并集 |
| `baspread` | 当月日均 `(high−low)/close`，为高低价差代理 |
| `maxret` | 当月可用调整后日收益的最大值 |
| `retvol` | 当月可用调整后日收益样本标准差，`ddof=1` |
| `mom1m` | 已实现的 $r_{i,t}$，不再额外滞后一月 |
| `mom6m`、`mom12m`、`mom36m` | 含 $t$ 的最近 6、12、36 自然月 $\prod(1+r)-1$，要求对应窗口完整有效；不是跳过最近月的动量 |
| `chmom` | `mom6m(t)−mom6m(t−6)`；任一项缺失则缺失 |
| `beta` | 最近 36 自然月个股月总收益对原始面板当月等权市场收益的带截距 OLS 斜率；至少 24 对有限观测 |
| `betasq` | `beta²` |
| `idiovol` | 与 beta 同一窗口、同一有效样本的回归残差样本标准差 |
| `chcsho` | `log(total_shares(t)/total_shares(t−12))`，要求精确滞后自然月及正股本 |
| `dy` | 最近 12 自然月已实施每股现金派息之和 / 当月末价格 |

`beta` 的协方差和市场方差均使用 `ddof=0`；残差为 $r_i-(\bar r_i-\hat\beta\bar r_m)-\hat\beta r_m$，残差标准差使用 `ddof=1`。市场方差为零时 beta、idiovol 保持缺失。市场收益由原始收益源的当月横截面均值构造，不要求与当前模型合格股票集合相同。

`dy` 在全体股票×完整月份网格上滚动，网格从最早形成月往前延伸 13 个月。没有匹配到派息记录的月份在该计算中记为零，因此“没有记录”与“确无派息”仍需源数据区分；跨拆股的每股现金口径也待核验。成交量单位、源日期并集日历和高低价差代理都应按其实际含义解释，不能仅沿用 GKX 缩写就声称原文因子已经精确复建。

### A.5 缩尾、排名、缺失值和标准化的顺序

实现顺序为：原始特征计算 → 当月特征缩尾 → 形成月资格过滤 → 当月排名及排名缺失填零 → 标签有效性过滤 → 目标月切分 → 训练集标准化。这一顺序本身是实验定义。

1. builder 将 19 特征的无穷值改为 NaN，再对各特征按月的有效横截面做 1%/99% 缩尾。此时尚未过滤 `eligible_asof`，最终无日线月份行的删除也在这之后。因此缩尾截面可能包含模型不合格行及日历补齐行中仍可计算的特征。
2. runner 先保留形成月合格行。对特征 $j$ 在月 $t$ 的有限值使用平均秩处理并列；令有效数量为 $n_{j,t}$，变换为 $q_{i,j,t}=2\operatorname{rank}_{j,t}(x_{i,j,t})/(n_{j,t}+1)-1$。缺失特征的排名值填 0，随后存为 float32。每个特征有自己的有效分母，不是统一使用当月总股票数；所有值并列时其变换为 0。
3. 只有完成上述排名后，才移除 `ret_fwd` 非有限行。标签缺失的股票仍参与形成时点的特征截面，以免排名直接依赖未来收益可得性。但训练/计分样本仍受已知收益可用性约束，这不能解决退市收益遗漏。
4. `standardize_inplace` 仅从训练行估计各列均值与总体标准差（float64 归约）；标准差小于 $10^{-8}$ 时设为 1，三段矩阵使用同一变换。矩阵本身为 float32、收益为 float64。排名缺失的 0 经过这一步后未必仍等于最终输入 0；它代表排名阶段的中性插补，不代表真实原始因子为零。

实现入口为 [run_ashare.py](../research/ML金融学习计划/D_A股数据/run_ashare.py) 的 `prepare_panel/standardize_inplace` 及 [evaluation.py](../research/ML金融学习计划/A_蒙特卡洛/evaluation.py) 的 `rank_characteristics/split_masks`。

### A.6 财务、行业和宏观扩展的隔离规则

本轮共同样本只启用 19 特征。财务扩展必须显式传入 `--allow-unverified-features`，财务 builder 必须显式传入 `--allow-unverified-vintages`；产物声明 `point_in_time_certified=false`。这两个参数只允许探索性执行，不会恢复供应商未保存的原始披露版本。

[基础财务 builder](../research/ML金融学习计划/D_A股数据/build_fundamentals.py) 对相同股票和报告期保留记录公告日最晚的版本，并传播保守公告日期；[完整财务 builder](../research/ML金融学习计划/D_A股数据/build_fundamentals_v2.py) 合并报表组件后使用所记录组件的最晚公告日。财政季度滞后按精确季度键取值，缺季度保持缺失。流量 TTM 为 Q4 年度值，或 Q1—Q3 的“当期 YTD + 上年 Q4 − 上年同期 YTD”；任一数值依赖缺失则 TTM 缺失，可用日期不早于全部依赖公告日期。月末仅向后匹配已到该保守日期的报表；基础财务模块另有 400 天匹配容忍期。

基础九财务特征的实际代理包括：当月原始市值作分母的 `bm/ep/sp/cfp`，当期资产作分母的 `roa/lev`，精确四季度资产同比 `agr`，营运资本代理变化/资产 `acc`，以及盈利同比改善连串长度（上限 4 后除以 4）的 `nincr`。`acc` 对可见的应收、存货、应付缺值有填零实现；`nincr` 的连串统计仍按可用报告行遍历。它们不能概括为“所有原始缺失均保留”或“完整季度连续性已认证”。56 列扩展中的部分会计科目代理和填零规则还需逐字段经济定义审计；完整公式以源码 `FEATURES` 及 `add_*` 函数为准。`divi/divo` 已暂停为全缺失；行业调整在月末可见截面与历史月行业中计算，不回退到当前行业快照。

runner 对已知危险历史缓存项继续排除：`divi, divo, herf, indmom, sin, bm_ia, cfp_ia, chatoia, chpmia, mve_ia, pchcapx_ia`。因此 `--chars 71` 是接口名称，不能据此保证修订输入恰有 71 个安全公司特征；真正列数应读 manifest。基础财务输入会优先读取 `fundamentals_monthly_v2.csv`，不存在才读 `fundamentals_monthly.csv`；两者与 `fundamentals_v2_monthly.csv` 不是同名资产，使用者须核对路径与哈希。

显式 `--industry` 时，按股票六位代码和形成月连接历史行业表。行业词表和频数门槛只从有标签的训练行学习，默认门槛 100；未知/低频类别映射 `OTHER`，词表字典序首项作为省略基准。词表的频数单位是股票月，不是独立公司数。

[宏观 builder](../research/ML金融学习计划/D_A股数据/build_macro.py) 的 `dp,ep,bm,ntis` 分别以当月原始市值加权 `dy,ep,bm,chcsho`，每个变量的分子和分母使用同一有限且市值为正的股票集合。市场收益用精确前一自然月市值加权原始总收益；`svar` 为其最近 12 自然月样本方差，最少 6 个月。它不是 GKX 的日频已实现方差。启用宏观交互后，runner 在训练股票月行上估计宏观变量均值/标准差，缺值填训练均值，标准化后只与公司特征排名相乘，不与行业虚拟变量交互，也没有单独追加纯宏观列。训练月份会因股票数不同而有不同重复权重；最终全部设计列再做训练集标准化。

## B. 估计器、调参目标与本轮设置

### B.1 共同选择规则与线性基准

六个主要机器学习家族为 PLS、PCR、Elastic Net、RF、GBRT、NN；OLS、Ridge、Lasso 为补充比较。本轮 A 股只训练 NN3；NN1—NN5 是模拟模块可执行的架构集合，不等于真实数据全部已跑。模型函数集中在 [models.py](../research/ML金融学习计划/A_蒙特卡洛/models.py)，NN 在 [nn.py](../research/ML金融学习计划/A_蒙特卡洛/nn.py)，网格由调用入口给定。

令训练设计为 $X$、响应为 $y$、行数为 $n$。线性/降维函数先保存训练均值 $\bar X,\bar y$，令 $X_c=X-\bar X,y_c=y-\bar y$；预测恢复为 $\bar y+(Z-\bar X)\hat\beta$。所有候选以验证集原收益单位的均方误差 `mean((y_valid−prediction)²)` 选择；代码只在严格更小时替换最优值，所以相同 MSE 保留先出现候选。测试集不参与单次网格选择。

OLS 用 `np.linalg.lstsq(Xc,yc,rcond=None)`，相当于最小化 $\|y_c-X_c\beta\|_2^2$。截距由中心化和均值恢复体现，未额外加入常数特征列。Ridge 目标为

$$
\min_\beta \|y_c-X_c\beta\|_2^2+\lambda\|\beta\|_2^2,
\qquad
\hat\beta_\lambda=V\,\operatorname{diag}\!\left(\frac{s_k}{s_k^2+\lambda}\right)U^\top y_c,
$$

其中 $X_c=U\operatorname{diag}(s)V^\top$。整个路径共享一次 SVD。这里 λ 对应未除以 $n$ 的平方损失；它不与下面 Elastic Net 的 α 同尺度。

Lasso/Elastic Net 调用 `ElasticNet(alpha=λ,l1_ratio=ρ,fit_intercept=False,max_iter=5000,tol=1e−5)`，对应

$$
\min_\beta\frac{1}{2n}\|y_c-X_c\beta\|_2^2
+\lambda\rho\|\beta\|_1+\frac{\lambda(1-\rho)}2\|\beta\|_2^2.
$$

Lasso 取 $\rho=1$，Elastic Net 取 $\rho=0.5$。非零系数数目按精确 `coef != 0` 统计。选中参数并不证明优化误差、网格边界或模型容量已经充分检验；例如本轮 Ridge 的 λ 正好在候选上端。

### B.2 PCR 与 PLS

PCR 先对 $X_c$ 做 SVD，然后对前 $k$ 个训练主成分回归：

$$
\hat\beta_k=V_{1:k}\operatorname{diag}(s_{1:k}^{-1})U_{1:k}^{\top}y_c.
$$

数值秩阈值为 `eps × max(Xc.shape) × s[0]`；候选 $k$ 截至该数值秩并去重，秩为 0 时输出训练均值。它依据解释变量方差形成成分，响应仅参与成分数选择及成分回归。

PLS 调用 `PLSRegression(n_components=k,scale=False)` 在 $X_c,y_c$ 上拟合，成分同时利用解释变量和响应的协变信息；外部已经进行训练集标准化，所以没有再启用类内尺度变换。验证和预测使用相同训练中心，最终加回 $\bar y$。具体迭代求解的未显式参数由 manifest 记录的 scikit-learn 版本决定，不另宣称作者实现了自定义 PLS 迭代器。

### B.3 RF 与 GBRT

RF 是 `RandomForestRegressor`：显式设置树数、最大深度、每次分裂候选特征数、最小叶样本数、`n_jobs=1` 和 `random_state=seed`。回归森林平均各树预测，树的回归损失与重采样等未显式选项采用所记录库版本默认值。验证 MSE 选择 `(depth,max_features)`，不以测试 R² 选择树深。

GBRT 是 `GradientBoostingRegressor`：显式设置树数、深度、学习率、`subsample=1.0`、最小叶样本数和随机种子。普通版本采用 `loss="squared_error"`，逐步拟合平方误差的下降方向；模拟 Huber 版本采用 `loss="huber",alpha=0.999`。这个 `alpha` 是残差绝对值的分位数参数，不是固定残差绝对阈值。当前实现是 scikit-learn 的这一梯度提升器，不能泛称为 XGBoost、LightGBM 或任意“提升树”实现。

### B.4 神经网络的目标、训练与早停

NN 采用纯 NumPy 实现，隐藏层宽度依次为 NN1=(32)、NN2=(32,16)、NN3=(32,16,8)、NN4=(32,16,8,4)、NN5=(32,16,8,4,2)。每个隐藏层执行仿射变换 → 批归一化 → ReLU；最后为线性输出。权重以标准差 $\sqrt{2/\mathrm{fan\_in}}$ 的正态分布初始化，偏置为 0，批归一化缩放 γ 为 1、平移 β 为 0；没有 dropout。

每个单网络仅从训练收益估计 $\mu_y,\sigma_y$，零标准差改为 1，以 $\tilde y=(y-\mu_y)/\sigma_y$ 训练，推断后还原原收益单位。每个小批的梯度对应

$$
\frac1{|B|}\sum_{i\in B}(f_\theta(X_i)-\tilde y_i)^2
+\lambda_1\left(\sum_\ell\|W_\ell\|_1+\|w_{\mathrm{out}}\|_1\right).
$$

L1 只加在隐藏层和输出层权重，不加在偏置及批归一化 γ、β。Adam 的 `beta1=0.9,beta2=0.999,eps=1e−8`；批量上限 10,000；每个 epoch 由同一个种子生成器重排训练行。学习率在候选网格中选择，训练过程中没有学习率衰减。批归一化 epsilon 为 $10^{-8}$，小批运行统计动量为 0.9；每个 epoch 结束后又用整个训练集重算统计量，并用该统计量在验证集推断。

早停 `patience=5`，验证标准化收益 MSE 改善超过 $10^{-12}$ 才保存新最优状态；连续 5 个 epoch 未达到该条件则停止。保存的是最佳权重和对应批归一化统计量。`best_epoch` 从 1 起算，最大 epoch 只是训练上限，既不等于最佳 epoch，也不等于实际停止 epoch。本轮结果仅保存最佳 epoch，不把未保存的停止 epoch 当作观测记录。

`fit_nn` 对每个 `(λ1,lr)` 候选先训练两个网络，种子为 `seed×1000+0` 与 `seed×1000+1`；将这两个网络还原为收益单位后的**平均预测**用于验证 MSE，而非先平均两个单模型 MSE。选定配置后，重新训练 `ensemble` 个网络，种子为 `seed×1000+100+e`，$e=0,\ldots,E-1$，每个网络仍单独依据验证集保存最佳 checkpoint。最终预测取这些网络收益预测的算术平均。

本轮 NN3 为 λ1=0.001、lr=0.1、E=3、epoch 上限 60；调参种子 0/1，最终种子 100/101/102，三个 `best_epochs` 均为 2。没有保存逐候选验证分数、逐 epoch 损失、实际停止 epoch、训练后完整权重或逐层激活轨迹，不能从最优参数补写这些记录。早停很早可提示后续优化诊断，但不能独自证明某一种欠拟合、过拟合或死亡 ReLU 原因。

### B.5 网格：A 股本轮、A 股默认与模拟入口

`logspace(a,b,k)` 指含两端点的 $k$ 个等对数间隔值。下表给出调用方的实际候选范围，而非论文全部候选空间：

| 方法 | A 股 runner 网格 | `run_mc.py` 网格/默认 |
|---|---|---|
| Ridge | `logspace(0,5,11)` | 同左；Ridge+H 亦同 |
| Lasso、ENet | `logspace(−6,−2,9)` | `logspace(−4,−1,10)`；普通/Huber 共用 |
| PCR、PLS | $k=1\ldots\min(p,n_{train}-1,40)$；本轮 1—19 | $k=1\ldots20$ |
| RF | depth={1,3,6}；mfeat={min(5,p),min(10,p)} 去重 | depth={1,3,6}；mfeat={30} |
| GBRT | depth={1,2}；lr={0.01,0.1}；树数固定为命令参数 | 同一深度/学习率网格；另拟合 GBRT+H |
| NN | λ1={1e−5,1e−4,1e−3} × lr={0.01,0.1} | 同左；缺省架构为 NN1—NN5 |
| GLM、GLM+H | 本轮未运行 | `logspace(−4,−0.5,25)` |

A 股 runner 的 CLI 默认 RF/GBRT 均为 200 棵、最小叶样本 100；本轮核心运行显式改为 **100 棵、叶样本 100**。默认 NN 集成 10 个、epoch 上限 100；本轮显式改为 3 个、60 epoch。`--nn-depth` 默认 0 表示不自动追加 NN；若模型列表已含 `nn` 而深度为 0，则使用深度 3。默认模型列表仅 `ridge,lasso,enet`，所以直接运行空参数命令不会自动完成九模型比较。

本轮选中参数依次为：Ridge λ=100000；Lasso λ=0.0031622776601683794、3 个非零系数；ENet λ=0.01、1 个非零系数；PCR k=1；PLS k=2；RF depth=1、mfeat=10；GBRT depth=1、lr=0.01；NN3 如上。树模型 seed 均为 0。原始参数列见[核心结果 CSV](../results/repaired/ashare_price19_repaired.csv)和[补充结果 CSV](../results/repaired/ashare_price19_repaired_extra.csv)。

### B.6 Huber、GLM 和 Oracle 的模拟扩展

稳健线性变体为本项目实现选择：训练中心化后固定 $\bar X,\bar y$，用残差绝对值 99.9% 分位数作为每轮阈值 ξ，权重 $w_i=\min(1,\xi/|e_i|)$。这并非对截距和固定 ξ 共同求解的统一标准优化器。OLS+H 从 OLS 起步，最多 30 轮 IRLS，系数最大变化小于 $10^{-9}$ 停止；Ridge+H 对每个 λ 从零系数起步，最多 15 轮，解 $(X_c^\top W X_c+\lambda I)\beta=X_c^\top Wy_c$，再按验证 MSE 选 λ。

Lasso+H/ENet+H 先用普通 Elastic Net 网格选 λ，随后**固定该 λ**做最多 10 轮加权重估，系数变化阈值 $10^{-8}$；源码注释提到重解 λ 网格，但实际循环没有重新搜索 λ。`X`、`y` 同时乘 $\sqrt w$，仍交给同一 `ElasticNet`。不应把注释中的意图写成已经执行的流程。

GLM 对每个输入特征使用 `SplineTransformer(n_knots=3,degree=3,include_bias=False,extrapolation="linear")`。逐组扣除训练样条均值，再 QR 正交化并乘 $\sqrt n$。在这些标准化正交组上求解 $(2n)^{-1}\|y_c-Q\beta\|^2+\lambda\sum_g\|\beta_g\|_2$；该组惩罚不等于在任意原始样条基上使用相同 λ。实现每个候选最多 60 次组坐标扫描，最大更新小于 $10^{-9}$ 时停止；每组步长依赖该组 Gram 矩阵最大特征值。函数签名中的 `iters=400` 未传入这个内部迭代上限，不能写作实际做了 400 轮。

GLM+H 在普通组惩罚验证选 λ 后，最多做 8 轮 IRLS；加权 `X,y` 均乘 $\sqrt w$，并重算加权块 Lipschitz 常数，不再假设组 Gram 为单位阵。组是否选中按组内系数绝对值之和大于 $10^{-10}$ 判断。

Oracle 只在模拟中使用：线性 DGP 的输入为 $(c_1,c_2,c_3x)$，非线性 DGP 为 $(c_1^2,c_1c_2,\mathrm{sgn}(c_3x))$，然后用训练数据做带截距 OLS。它知道真实函数的特征形式，但仍估计系数；不是直接输出已知 $g^*$ 的无估计误差预测。多期限入口沿用形成时点这些 Oracle 特征回归期限响应，不能称为已解析求得了多期条件期望 Oracle。

## C. 蒙特卡洛设计与补充表

### C.1 DGP 方程与随机性

[dgp.py](../research/ML金融学习计划/A_蒙特卡洛/dgp.py) 固定 $N=200,T=180,P_c\in\{50,100\}$，设计矩阵有 $P=2P_c$ 列，按 $[c_1,\ldots,c_{P_c},c_1x,\ldots,c_{P_c}x]$ 排列。对每个模拟种子创建一个 `np.random.default_rng(seed)`，依次生成公司特征、宏观状态、共同因子冲击和个体冲击：

$$
\rho_j\sim U[0.9,1),\quad
\bar c_{ij,0}\sim N(0,1),\quad
\bar c_{ij,t}=\rho_j\bar c_{ij,t-1}+\sqrt{1-\rho_j^2}\,\eta_{ij,t},
\quad \eta_{ij,t}\sim N(0,1),
$$
$$
c_{ij,t}=\frac{2\,\operatorname{CSrank}(\bar c_{ij,t})}{N+1}-1,
\qquad x_0\sim N(0,1),\quad
x_t=0.95x_{t-1}+\sqrt{1-0.95^2}\,u_t,
\quad u_t\sim N(0,1).
$$

模拟排名使用双重 `argsort` 生成 1—N 秩，未使用 A 股的平均并列秩处理；连续正态变量几乎处处无并列。代码先生成 $T+1$ 期公司特征和宏观状态，收益设计只取前 $T$ 期；未设置额外 burn-in。

$$
r_{i,t+1}=g^*(c_{i,t},x_t)+\beta_{i,t}^{\top}v_{t+1}+\epsilon_{i,t+1},
\quad \beta_{i,t}=(c_{i1,t},c_{i2,t},c_{i3,t})^{\top},
$$
$$
v_{t+1}\sim N(0,0.05^2I_3),\qquad
\epsilon_{i,t+1}=0.05\,T_{5,i,t+1}.
$$

这里 0.05 是 Student-t 的 **scale**，所以 $\operatorname{Var}(\epsilon)=0.05^2\times5/3$，不是 0.05²。共同因子使同月个股误差相关。两种条件均值为

$$
g_a^*=0.02c_1+0.02c_2+0.02c_3x,
\qquad
g_b^*=0.04c_1^2+0.03c_1c_2+0.012\,\operatorname{sgn}(c_3x).
$$

`--theta-scale` 对以上三项信号系数整体相乘，不改变噪声尺度。DGP 函数也允许直接传入 `theta/v_scale/eps_scale` 覆盖值；主 CLI 只公开 theta-scale。记录不充分的历史结果不能仅按当前默认值反推其生成参数。

### C.2 期限目标、划分与边界清除

[run_mc.py](../research/ML金融学习计划/A_蒙特卡洛/run_mc.py) 的 `build_design` 在 $h=1$ 使用下一月收益；在 $h>1$ 使用

$$
y^{(h)}_{i,t}=\sum_{k=0}^{h-1}r_{i,t+k+1},\qquad T_{eff}=T-h+1.
$$

这是**月简单收益之和**，不是 $\prod_{k=1}^{h}(1+r_{i,t+k})-1$ 的复合持有收益。输入仍为形成时点 $c_{i,t},c_{i,t}x_t$，没有把未来特征放入设计。数组先按时间、再按股票展平。

令 $L=\lfloor T_{eff}/3\rfloor$。训练形成期取 `[0,L−(h−1))`，验证取 `[L,2L−(h−1))`，测试取 `[2L,T_eff)`，区间右端不包含。训练和验证各移除末尾 $h-1$ 个形成期，使相邻集合标签不共用同一收益冲击；同一集合内部的多期响应仍可重叠。该防交叠协议是修订后的研究实现，不宣称已经证明是 GKX 原始多期限代码的唯一切法。

| 期限 h | $T_{eff}$ | 训练形成期数 | 验证形成期数 | 测试形成期数 | 每条边界清除期数 |
|---:|---:|---:|---:|---:|---:|
| 1 | 180 | 60 | 60 | 60 | 0 |
| 3 | 178 | 57 | 57 | 60 | 2 |
| 6 | 175 | 53 | 53 | 59 | 5 |
| 12 | 169 | 45 | 45 | 57 | 11 |

各期数乘 200 为该集合股票期行数。模拟标准化仅使用训练均值和总体标准差，恰为零的标准差改为 1；与 A 股 runner 的 $10^{-8}$ 门槛及 float32 矩阵实现不完全相同。

### C.3 主入口与 A.3/A.4/分片入口的默认差异

| 项目 | `run_mc.py` | `table_a3.py` | `table_a4.py` | `launch_shards.py` |
|---|---|---|---|---|
| 默认 Pc / DGP | 50 / a | 50 / a | 50 / a | 50 / a |
| 默认重复数 | 10 | 50 | 20 | 总计 100，默认 12 分片 |
| 默认 theta-scale | **1.0** | **1.08** | **1.08** | 子命令固定传 **1.08** |
| 默认方法 | lin,oracle | l1 | tree | kind 必填 lin/tree/nn |
| RF 深度 | 1,3,6 | 不适用 | 1,2,3,4,5,6 | 1,2,3,4,5,6 |
| RF 候选特征数 | 30 | 不适用 | 3,5,10,20,30,50 | 3,5,10,20,30,50 |
| GBRT 学习率 | 0.01,0.1 | 不适用 | 0.001,0.01 | 0.001,0.01 |
| RF/GBRT 树数 | 300/300 | 不适用 | 300/300 | 300/300 |
| 输出目录 | `GKX_MC_OUT_DIR`，否则 `out_repaired/` | 脚本旁 `out/` | 脚本旁 `out/` | 继承子进程环境 |

模拟 RF 最小叶样本来自 `fit_rf` 默认值 5；GBRT 来自 `fit_gbrt` 默认值 1。NN 默认 10 个最终集成网络、最多 100 epoch。各直接入口种子为 `seed0+rep`，默认 seed0=0。分片脚本将 0 到总重复数−1 尽量均匀切成连续段；它调用相对 `run_mc.py`，应在蒙特卡洛目录执行。不能把主入口的 theta-scale=1、窄树网格写成所有历史表共同设置，也不能把 A.3/A.4 的 1.08 默认为论文原始参数。

### C.4 变量选择频率与重要性

[Table A.3](../research/ML金融学习计划/A_蒙特卡洛/table_a3.py) 在同一 DGP 训练/验证切分中拟合 Lasso、ENet、GLM 及各自 Huber 版本。Lasso/ENet 用系数非零，GLM 用组内绝对系数和大于 $10^{-10}$ 判定选中。记录的六个位置为 `0,1,2,Pc,Pc+1,Pc+2`，对应 $c_1,c_2,c_3,c_1x,c_2x,c_3x$；其余 $P-6$ 列的频率取所有重复和列的平均。六个关注变量不代表两种 DGP 下六个都是独立真实有效项。

[Table A.4](../research/ML金融学习计划/A_蒙特卡洛/table_a4.py) 对树模型读取 `feature_importances_` 的杂质下降重要性；每次重复将整条向量归一化为总和 100，再跨重复平均。NN 在推断模式、使用训练批归一化统计量，计算训练输入坐标上的

$$
SSD_j=\sum_{i\in train}\left(\frac{\partial f(X_i)}{\partial X_{ij}}\right)^2.
$$

每个最终网络先将自己的 SSD 归一化为 100，再平均各网络的比例；它不是先求集成预测梯度再平方，也不是原始未标准化因子的敏感度。输出收益标准化的共同尺度在单网络比例内会抵消。总 SSD 非有限或非正的网络被跳过；若一个架构所有网络均如此，该次架构没有重要性记录。因而每架构有效重复数必须核对，不能自动等同请求重复数。树重要性在全零情形没有对应显式保护，重跑时应检查有限值。

表中的 `Noise` 为剩余 $P-6$ 个变量的**单变量平均**重要性，不是噪声总份额。因此归一化核对是“六项之和 + (P−6)×Noise”，而不是表面七列简单相加。这些重要性也不是因果效应或统计显著性检验。

## D. 误差指标与截面相关

### D.1 四种 R² 的分母

令测试目标月集合为 $\mathcal T$，该月已观测标签样本为 $\mathcal I_t$，$e_{it}=y_{it}-\hat y_{it}$。定义 $SSE=\sum_{t\in\mathcal T}\sum_{i\in\mathcal I_t}e_{it}^2$，训练收益均值为 $\bar y_{tr}$，测试样本自身均值为 $\bar y_{te}$。[evaluation.py](../research/ML金融学习计划/A_蒙特卡洛/evaluation.py) 分开实现：

$$
R^2_{0,pooled}=1-\frac{SSE}{\sum_{t,i}y_{it}^{2}},
\qquad
R^2_{train\ mean}=1-\frac{SSE}{\sum_{t,i}(y_{it}-\bar y_{tr})^{2}},
$$
$$
R^2_{sample\ mean}=1-\frac{SSE}{\sum_{t,i}(y_{it}-\bar y_{te})^{2}},
\qquad
\overline{R^2_{0,month}}=\frac{1}{M_{defined}}\sum_{t:\,D_t>0}
\left(1-\frac{\sum_i e_{it}^{2}}{D_t}\right),\quad D_t=\sum_i y_{it}^{2}.
$$

零基准 pooled 指标为本轮 A 股主指标；训练、验证 R² 也采用零基准。测试样本中心化指标的基准使用了事后测试均值，只能作诊断；它不是训练历史均值预测，也不是截面排序指标。等月平均只对分母为正、R² 有定义的月份取平均，分母为零返回 NaN。pooled 指标等于以 $D_t$ 为权重的月 R² 平均，而非等月或按股票数的平均，因此其符号可与等月平均不同。

计分拒绝空向量、非匹配形状及非有限收益/预测，不在内部静默删除错误观测；平方和用 float64。R² 可为负，且没有下界。CSV 中 1.02 代表 1.02% 的 R²，不是 1.02% 年化收益、预测方向准确率或固定程度的排序能力。

模拟指标另外遵循 [run_mc.py](../research/ML金融学习计划/A_蒙特卡洛/run_mc.py) 的 `r2_relative_to_train_mean`：IS 和 OOS 分母均以训练收益均值为中心。IS 分母为训练 $\sum(y-\bar y_{tr})^2$，OOS 分母为测试 $\sum(y-\bar y_{tr})^2$。它不是 A 股的主零基准 R²；两类实验不能不注明基准就排在同一成绩榜上。

### D.2 逐月 Pearson IC、Spearman Rank IC 与有效月份

[聚合脚本](../scripts/summarize_predictions.py) 在每个**目标收益月**计算

$$
IC_t=\operatorname{Corr}_{i\in\mathcal I_t}(\hat y_{it},y_{it}),\qquad
RankIC_t=\operatorname{Corr}_i(\operatorname{rank}(\hat y_{it}),\operatorname{rank}(y_{it})).
$$

Spearman 使用双方平均并列秩后计算 Pearson 相关。少于 2 个观测，或预测/收益极差 `np.ptp` 恰为零时相关系数为 NaN。极差检查避免完全相同的小数因标准差浮点残差产生伪相关；缺失不会填零。均值按有效月份等权计算，并分别保存 `pearson_ic_valid_months` 和 `spearman_rank_ic_valid_months`。

本轮九模型共同覆盖 212,293 个测试股票月、48 个目标月；其余八模型两类 IC 均有 48 个月。**GBRT 有 47 个月给整个截面完全相同的预测，仅 2025-05 有两个预测值；两种 IC 各只有 1/48 个月有定义。** 其均值 −0.1562 和 −0.1014 只能描述那一个月，不能概括为持续负排序，亦不可与其他模型 48 月均值视作相同覆盖率比较。GBRT 的 pooled R² 仍用全部 48 个月，正值不代表持续存在截面区分度。原始逐月分子、分母、相关系数见[monthly_metrics.csv](../results/repaired/monthly_summary/monthly_metrics.csv)。

这些是描述性点估计，本轮没有计算 HAC 标准误、月份区块 bootstrap 区间、模型差异显著性或多重比较修正。股票月共享市场冲击且有时间依赖，不能把 212,293 行当成 212,293 次独立重复。未来推断必须明确月份依赖及整个研究反复使用测试期的选择问题。

### D.3 两批预测的共同样本核验

聚合器要求每份本地预测有 `code,month,label_month,ret_fwd,market_cap`。两批按三字段键进行严格一对一**外连接**，只有每行均同时存在才继续；重复模型列、重复键、缺键、标签月不一致均报错。两边 `ret_fwd` 和 `market_cap` 逐值要求相同（允许双方市值同时缺失），而非各自删行后只比较看起来相近的结果。

逐月输出包含 `n,sum_squared_error,sum_squared_return,r2_zero,pearson_ic,spearman_rank_ic`。再次汇总 SSE/Σy² 重算 pooled R²，按 `model` 与训练程序结果表一对一核对，容差为 $10^{-10}$ 个**百分点**。本轮最大差异约 $2.22\times10^{-14}$ 个百分点。这个重算支持汇总与预测一致，不构成不同原始数据来源的独立验证。

## E. 组合构造与绩效诊断

[portfolio_check.py](../research/ML金融学习计划/D_A股数据/portfolio_check.py) 从修订版预测文件读取已观测总收益，要求 manifest 的目标字符串为 `next-calendar-month unmodified total return`。按目标月分组相当于使用对应上一形成月的预测排序；重复股票—目标月、非有限预测/收益直接报错。每月至少 20 只股票；预测常数则跳过；使用 `pd.qcut(prediction,10,labels=False,duplicates="drop")`，若实际不足十组则跳过该月，不以代码顺序打破并列硬凑十分位。

最高组与最低组分别等权持有，定义

$$
R^H_t=\frac1{|H_t|}\sum_{i\in H_t}y_{it},\quad
R^L_t=\frac1{|L_t|}\sum_{i\in L_t}y_{it},\quad
R^{spread}_t=R^H_t-R^L_t.
$$

`market_ew` 是同一已观测标签样本全体股票等权收益，不是独立发布的全市场指数。样本资格与未来标签覆盖会影响这个可计分集合；没有持仓缺失后延续、退市变现或停牌资金占用模型。

对一条月收益序列 $R_1,\ldots,R_M$，[portfolio_statistics](../research/ML金融学习计划/A_蒙特卡洛/evaluation.py) 要求至少两个有限观测，且每项严格大于 −1。样本标准差 $s_R$ 采用 `ddof=1`，输出：

$$
Ratio_{rf=0}=\sqrt{12}\frac{\bar R}{s_R},\quad
W_0=1,\quad W_m=\prod_{j=1}^{m}(1+R_j),
$$
$$
CAGR=W_M^{12/M}-1,\quad
CumulativeReturn=W_M-1,\quad
MDD=\min_{0\le m\le M}\left(\frac{W_m}{\max_{0\le j\le m}W_j}-1\right).
$$

标准差为零时比率写 `null`。回撤财富曲线包含初始净值 1，第一月亏损不会被遗漏。月份不连续时 runner 把 CAGR 置 `null`；剩余累计收益与回撤仍只是可用观测序列诊断，不能解释成完整连续账户表现。CAGR 不是 $(1+\bar R)^{12}-1$。

本轮 Ridge、NN3 均形成连续 48 个月十分位，公开保存[两组组合序列及统计](../results/repaired/)；高组 CAGR 分别为 27.3109% 和 26.0039%，最大回撤分别为 −20.4003% 和 −22.2072%。这些值是毛收益诊断。年化均值/标准差显式假定 rf=0，不能称为已扣除真实无风险收益的多头 Sharpe。最低组收益不是实施做空所得损益；高减低价差的复合值也未指定资金、保证金、借券与再投资账户，所以不能视为可实现的投资者 CAGR。手续费、冲击成本、交易延迟、涨跌停成交、借券可得性、换手和容量均未建模。

## F. 可移植执行与输出约定

### F.1 环境与执行层级

以下命令从仓库根目录执行，使用当前环境的 `python`，并将运行文件放到新的本地目录。先合法取得原始缓存；`private/raw` 只是示例输入目录，须替换为实际获准路径。`requirements.txt` 是兼容范围清单，不是精确锁文件。安装依赖前应先检查自己环境；要复现某个已归档执行，优先按其 manifest 还原版本，而非默认最新依赖。

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run_smoke.py --nn

$repoRoot = (Get-Location).Path
$env:GKX_DATA_DIR = (Resolve-Path './private/raw').Path
$env:GKX_DERIVED_DIR = Join-Path $repoRoot 'local_runs/reproduce_01/features'
$env:GKX_OUT_DIR = Join-Path $repoRoot 'local_runs/reproduce_01/models'
$env:GKX_MC_OUT_DIR = Join-Path $repoRoot 'local_runs/reproduce_01/mc'
```

`GKX_DATA_DIR` 与 `GKX_DERIVED_DIR` 必须不同；builder 会检查解析后目录是否相同。runner 优先读 `GKX_DERIVED_DIR`，其次才读 `GKX_DATA_DIR`，都未设置则读取脚本旁 `data/`。`GKX_CLEAN_DIR` 仅用于显式行业扩展，指向包含历史 `stock_industry_monthly.csv` 的目录。工作目录变化不会改变这些已设为绝对值的环境变量。

合成 smoke 的输出来自虚构股票和收益，只检验管道能否执行，不能替代真实数据、全量 Monte Carlo 或经济定义验证。它运行 Ridge、RF、GBRT 及可选 NN1；`--nn` 下 NN 只有 2 个集成网络、最多 5 epoch，属于工程试运行设置。

### F.2 重建本轮 19 特征及九模型

```powershell
python 'research/ML金融学习计划/D_A股数据/build_features.py' --min-months 24 --min-days 10
python 'research/ML金融学习计划/D_A股数据/run_ashare.py' --chars 19 --models ridge,enet,pls,rf,gbrt --rf-trees 100 --rf-leaf 100 --gb-trees 100 --gb-leaf 100 --seed 0 --tag reproduce_01_core
python 'research/ML金融学习计划/D_A股数据/run_ashare.py' --chars 19 --models ols,lasso,pcr,nn --nn-depth 3 --nn-ensemble 3 --nn-epochs 60 --seed 0 --tag reproduce_01_extra
```

每个 tag 产出 `ashare_<tag>.csv`、`predictions_<tag>.csv`、`manifest_<tag>.json`。指标及预测在每个模型完成后重写当前 checkpoint；manifest 在全部模型完成后才落盘。因此只见到部分 CSV 而无 manifest 时，不能把该目录认作完整成功运行。已有指标文件时 runner 拒绝相同 tag；新执行仍应使用全新目录/tag，不能依赖这一单文件检查作为所有输出的事务保护。

下面从自己的新预测文件重算聚合，不需要也不应把逐股文件复制到公开结果目录：

```powershell
python scripts/summarize_predictions.py --input "$env:GKX_OUT_DIR/predictions_reproduce_01_core.csv" "$env:GKX_OUT_DIR/predictions_reproduce_01_extra.csv" --metrics "$env:GKX_OUT_DIR/ashare_reproduce_01_core.csv" "$env:GKX_OUT_DIR/ashare_reproduce_01_extra.csv" --output 'local_runs/reproduce_01/monthly_summary'
python 'research/ML金融学习计划/D_A股数据/portfolio_check.py' --predictions "$env:GKX_OUT_DIR/predictions_reproduce_01_core.csv" --manifest "$env:GKX_OUT_DIR/manifest_reproduce_01_core.json" --model ridge --output 'local_runs/reproduce_01/ridge_portfolio'
python 'research/ML金融学习计划/D_A股数据/portfolio_check.py' --predictions "$env:GKX_OUT_DIR/predictions_reproduce_01_extra.csv" --manifest "$env:GKX_OUT_DIR/manifest_reproduce_01_extra.json" --model nn3 --output 'local_runs/reproduce_01/nn3_portfolio'
```

聚合器拒绝已有输出目录；组合器拒绝已有结果文件。财务和宏观扩展另用新的 tag/目录，并明确其探索性状态。旧 `run_ashare2.py/run_ashare3.py` 已委托同一修订 runner；保留文件名用于追溯，不会自动恢复旧版 71/56 财务定义或默认行业集合。

### F.3 显式模拟命令

以下是新实验命令，不表示旧大规模结果已经用修订代码重跑。先用少量重复检查环境，再预先冻结完整配置；重复数、种子、theta-scale、期限及每个网格都应在执行日志中保留。

```powershell
python 'research/ML金融学习计划/A_蒙特卡洛/run_mc.py' --pc 50 --model a --models lin,oracle --reps 2 --seed0 0 --theta-scale 1.0 --horizon 1 --tag mc_check_pc50_a
python 'research/ML金融学习计划/A_蒙特卡洛/run_mc.py' --pc 100 --model b --models tree --reps 2 --seed0 0 --theta-scale 1.0 --horizon 3 --rf-trees 300 --gb-trees 300 --rf-depth 1,3,6 --rf-mfeat 30 --gb-depth 1,2 --gb-lr 0.01,0.1 --tag mc_check_pc100_b_h3
python 'research/ML金融学习计划/A_蒙特卡洛/run_mc.py' --pc 50 --model a --models nn --nndepth 3 --nnes 3 --nep 60 --reps 2 --seed0 0 --theta-scale 1.0 --horizon 1 --tag mc_check_nn3
python 'research/ML金融学习计划/A_蒙特卡洛/table_a3.py' --pc 50 --model a --models l1,glm --reps 2 --seed0 0 --theta-scale 1.08 --tag a3_check_pc50_a
python 'research/ML金融学习计划/A_蒙特卡洛/table_a4.py' --pc 50 --model a --methods tree --reps 2 --seed0 0 --theta-scale 1.08 --rf-trees 300 --gb-trees 300 --rf-depth 1,2,3,4,5,6 --rf-mfeat 3,5,10,20,30,50 --gb-lr 0.001,0.01 --tag a4_check_pc50_a
```

A.3/A.4 当前输出固定到蒙特卡洛脚本旁 `out/`，不会跟随 `GKX_MC_OUT_DIR`。主模拟 raw 文件每行保存 seed、模型、`split_protocol`、三段行数以及四位小数的百分数 IS/OOS R²；汇总 CSV 由这些已舍入数值求均值和重复间样本标准差。标准差不是均值标准误。原始 `info` 虽在内存生成，但只有首次重复的部分参数打印到标准输出，没有逐重复完整写入 raw。

主模拟 `--resume` 检查 horizon/purge 协议，并按已见 seed 整体跳过，未核对所有网格、theta、源代码或该 seed 的模型集合完整性。中断或更改模型集合后续跑须额外审查，不能把它当作任意配置可混合的通用 checkpoint。A.3 的修订续跑则把 l1/glm 家族展开为实际方法名，校验 pc、DGP、theta-scale 和相关源码 SHA-256 签名；拒绝旧无签名、重复或损坏记录，已存在 `(seed,method)` 不重复写入。A.4 拒绝向已有 tag 隐式追加；它没有提供与 A.3 相同的续跑接口。

[launch_shards.py](../research/ML金融学习计划/A_蒙特卡洛/launch_shards.py) 可在蒙特卡洛目录组织分片；[merge_raw.py](../research/ML金融学习计划/A_蒙特卡洛/merge_raw.py) 则固定查找该目录 `out/raw_<tag>*.jsonl`，不读取 `GKX_MC_OUT_DIR`。合并器对重复 `(seed,model)` 仅保留首条，不校验完整实验签名，并会跳过无法解析的行。因此使用分片工具时必须显式统一输出位置、配置和种子覆盖，检查每模型有效重复数；不能把“合并成功”当作来源一致认证。所有新 tag 均应与历史归档隔离。

## G. manifest、哈希、环境与不可恢复信息

### G.1 已保存的绑定关系

两份 A 股 manifest 保存作者、目标定义、R² 单位、按目标月的窗口、三段样本数、测试形成月与目标月、实际特征列、命令参数、预处理元数据、输入文件名/字节数/SHA-256、Python/NumPy/pandas/scikit-learn 版本，以及 `run_ashare.py/evaluation.py/models.py/nn.py` 源码哈希。本轮两批的输入及上述核心模型源哈希相同。输入面板 SHA-256 为

```text
0a993bef6247fff10db46ace0798bccd1898a55562a99a6e49d881eba8ec0e06
```

面板为 266,553,972 字节。[frozen_build.json](../results/repaired/frozen_build.json) 记录完整重建前后三个依赖源文件的哈希、开始/结束 UTC、参数与返回码；[replay_comparison.json](../results/repaired/replay_comparison.json) 记录首次模型输入与冻结重建输出逐字节一致及原月收益哈希。该重放绑定到生成时的冻结源码，不意味着以后修改源码后旧结果会自动更新，也不验证第三方输入的经济真实性。

[aggregation_manifest.json](../results/repaired/monthly_summary/aggregation_manifest.json) 另外保存两份本地预测与两份指标 CSV 的哈希/字节数、聚合脚本哈希、环境、共同键检查和 R² 容差。原始预测不随公开汇总发布；SHA-256 只能检验持有者手中某份文件是否一致，不能由哈希恢复预测，更不能恢复所用原始行情。

### G.2 各阶段环境分别报告

| 已记录阶段 | Python | NumPy | pandas | scikit-learn |
|---|---|---|---|---|
| 冻结价格特征重建 | 3.12.14 | 2.3.5 | 3.0.1 | 该记录未列出 |
| 九模型训练 | 3.12.14 | 2.5.4 | 3.0.6 | 1.9.1 |
| 月度指标重聚合 | 3.12.14 | 2.3.5 | 3.0.1 | 该步骤未记录/未重训模型 |

不能把它们写成完全相同的单一环境。`requirements.txt` 当前为宽范围约束，不能单凭它保证字节级复现。现有 manifest 也没有完整记录 BLAS 实现及线程数、全部依赖锁定、CPU 特性和编译选项；随机种子固定仍不等于任何平台下浮点位级完全一致。只有已经执行的冻结重建文件比对可以支持此次特定输入的字节一致结论。

### G.3 历史记录、缺失记录与验证的范围

[历史 manifest](../results/historical/manifest.json) 记录被复制的相对源路径、文件长度与 SHA-256，说明归档副本与取得文件一致。它不能补回旧运行提交、依赖锁定、全部候选 MSE、每个 NN 的训练轨迹或原始财报 vintage。历史 NN3 约 1.02% 是另一版 152 列输入的已有聚合记录，不能用当前 19 列结果代替，也不能由两者差值定量识别某项修复的效果。旧年度/期限/重要性表是否曾采用 1.08 信号缩放及具体网格，应以保存日志证据为准，不能从当前入口默认值倒推。

本轮已保存最优超参数、测试预测和聚合证据，但未保存所有候选分数、完整拟合模型权重、NN 逐 epoch 轨迹、实际早停时刻或完整逐股持仓与成交账本。原始股票池历史版本、供应商更正前报表、抓取时间及部分公司行动覆盖同样无法由汇总 CSV 恢复。可以补做的新实验应另建版本；无法恢复的历史事实应标注未记录，不能以当前运行结果回填。

修订测试记录见[test_log.txt](test_log.txt)：本次合并运行 36 项测试通过，包含数据契约、模型/评价、期限边界、聚合与记录续跑等有限案例。它们验证被明确覆盖的代码行为，不证明所有经济定义正确、不认证第三方原始数据，也不等于历史全量模拟已重跑。代码哈希、重放一致、预测聚合一致、统计推断和可交易性验证是不同证据，任何一项都不能代替其他项。

本附录由 Gilmour Jia 的研究代码与已保存记录整理。Codex 协助实现核对和文稿编排；作者负责最终研究判断、数据授权及发布内容。这里未将 AI 工具列为作者，也未声称进行外部独立审稿认证。


