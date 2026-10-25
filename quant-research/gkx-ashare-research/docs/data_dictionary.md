# 数据字典与时间口径

作者：**Gilmour Jia**  
版本日期：2026-10-25

本字典描述公开代码的修订版。历史结果 CSV 对应旧版数据和预处理，不能套用本字典重新解释成修订版结果。数值收益使用小数单位：`0.01` 表示 1%。模型结果表中的 `OOS` 则按百分数输出：`1.02` 表示 R²=1.02%。

## 修订价格特征面板：29 列

文件：`GKX_DERIVED_DIR/features_monthly.csv`。主键为 `(code, month)`，要求唯一。形成月 `month=t` 的月末信息预测 `label_month=t+1` 的总收益；原始数据与财报版本尚未被本次实现测试完整认证。

| 字段 | 类型／单位 | 定义与可用时点 | 缺失及处理规则 |
|---|---|---|---|
| `code` | 字符串 | 带 `sh/sz/bj` 前缀的股票代码 | 不允许主键重复 |
| `month` | `YYYY-MM` | 特征形成月 t | 不根据未来收益是否可得删除形成月 |
| `label_month` | `YYYY-MM` | 精确的下一自然月 t+1 | 月份始终可计算，不代表该月收益一定存在 |
| `ret` | 小数收益 | 原始面板中 t 月总收益，含现有现金分红和送转处理 | 缺失保留；不缩尾；源记录仍可能存在公司行动遗漏 |
| `ret_fwd` | 小数收益 | 原始面板中 `(code,t+1)` 的总收益 | 找不到精确下一月则 NaN，不能跳到下一条有记录月份；不缩尾 |
| `ndays` | 整数，CSV 可能浮点存储 | t 月日线记录数 | 这是源行情有记录的日期数，未单独核验每一天实际可成交 |
| `close` | 元／股 | t 月最后有记录日的未复权收盘价 | 缺失保留，不用于填补未来价格 |
| `market_cap` | 元 | t 月最后有记录日收盘价 × 当时股本记录的总股数 | 正值原始规模，独立于 `mvel1` 的缩尾／排名；原始股本时间戳仍待验证 |
| `history_months` | 整数 | 截至 t 月有日线记录的月份累计数 | 当前实现统计观测历史，不能当作已验证的上市龄 |
| `eligible_asof` | 布尔值 | 当月日线数量达到 `min-days`，且历史观测月数达到 `min-months` | 默认分别为 10 和 24；不使用 t+1 交易天数或未来覆盖长度 |
| `mvel1` | 自然对数 | `log(market_cap)` | 非正股本置缺失；特征层缩尾，模型层再排名 |
| `turn` | 比率 | t 月成交量合计 ÷ 月末流通 A 股股数 | 分母非正或缺失则缺失；成交量原始单位须由来源进一步核验 |
| `std_turn` | 比率标准差 | 包含 t 的最近 3 个自然月 `turn` 样本标准差，至少 2 个有效月 | 缺月进入日历网格，不能用第 3 条有记录月份冒充第 3 个自然月 |
| `dolvol` | 自然对数 | 每月日均 `close×volume` 取 log，再取包含 t 的 3 自然月均值，至少 1 个有效月 | 交易额为行情价格和成交量构造的代理，非独立核验的成交额字段 |
| `ill` | 尺度化比率 | 当月均值 `abs(adjusted_daily_return)×1e9/(close×volume)` | 零交易额置缺失，属于 Amihud 类代理 |
| `zerotrade` | 比率 | `1−ndays/该月源日历交易天数` | 日历来自所读股票日线日期并集，非独立交易所日历 |
| `baspread` | 比率 | 当月均值 `(high−low)/close` | 高低价差代理，不能称为实际报价买卖价差 |
| `maxret` | 小数收益 | 当月调整后日收益最大值 | 调整覆盖现有现金派息和送转；配股等仍待补充 |
| `retvol` | 小数收益标准差 | 当月调整后日收益样本标准差（ddof=1） | 少于 2 个有效日收益则缺失 |
| `mom1m` | 小数收益 | 特征月 t 已实现的原始月总收益 | 与旧代码额外 `shift(1)` 版本不同；该月未知则缺失 |
| `mom6m` | 累计小数收益 | 包含 t 的最近 6 自然月 `∏(1+ret)−1` | 需要 6 个完整有效月；包含最近月，不能混称为跳过最近月的动量 |
| `mom12m` | 累计小数收益 | 包含 t 的最近 12 自然月累计总收益 | 需要 12 个完整有效月；原文动量窗口仍需单独逐项对齐 |
| `mom36m` | 累计小数收益 | 包含 t 的最近 36 自然月累计总收益 | 需要 36 个完整有效月 |
| `chmom` | 收益差 | `mom6m(t)−mom6m(t−6)` | 精确自然月滞后；任一项缺失则缺失 |
| `beta` | 无量纲斜率 | 最近 36 自然月个股总收益对原始面板等权市场收益的带截距 OLS 斜率 | 至少 24 对有限观测；市场方差为零则缺失；协方差和方差均 ddof=0 |
| `betasq` | 无量纲 | `beta²` | beta 缺失则缺失 |
| `idiovol` | 小数收益标准差 | 上述回归残差的样本标准差 | 与 beta 使用同一窗口和有效观测 |
| `chcsho` | 对数变化 | `log(total_shares(t)/total_shares(t−12))` | 缺少精确 12 自然月前的股本或非正股本则缺失 |
| `dy` | 比率 | 最近 12 自然月已实施每股现金派息之和 ÷ t 月收盘价 | 使用除权日，不使用未来年度派息；除权方案覆盖与跨拆股的每股口径仍待核验 |

19 个因子在 builder 中按月对可计算特征做 1%/99% 横截面缩尾，随后模型脚本在当月合格股票中做横截面排名。缩尾不作用于 `ret`、`ret_fwd`、`market_cap`。builder 的缩尾先于资格过滤，这是一个需要在后续统一预处理实验中检验的明确选择。模型排名后的缺失值填为中性值 0；原始 CSV 保留 NaN，不把不可得原始数据写成真实零值。

CSV 包含标签缺失但形成月信息已知的记录，避免横截面预处理先根据未来收益覆盖选股。模型评估只能使用标签存在的样本，并应报告其数量；这仍不能消除原始股票池的历史缺失。

## 原始输入表所需字段

| 表 | 必要字段 | 时间／单位及用途 |
|---|---|---|
| `prices/<code>.csv` | `date, open, high, low, close, volume` | 日线；价格为未复权价格；日期须唯一；`open` 为原接口保留字段，主要特征使用 high/low/close/volume |
| `returns_monthly.csv` | `code, month, ret` | 原始月总收益；主键唯一；数据 builder 不使用历史缩尾后的 `returns_clean.csv` 作为标签来源 |
| `shares.csv` | `SECURITY_CODE, END_DATE, TOTAL_SHARES, LISTED_A_SHARES` | `END_DATE` 用于向后匹配股本有效记录；股数单位须核验，日期不等同已证明的首次公开时间 |
| `dividends.csv` | `SECURITY_CODE, EX_DIVIDEND_DATE, PRETAX_BONUS_RMB, BONUS_IT_RATIO` | 现金和送转比例按每 10 股口径除以 10；除权日用于已实施分配事件；来源方案重复／缺失仍需核验 |
| `statements.csv` | `SECURITY_CODE, REPORT_DATE, NOTICE_DATE` 以及下列财务科目 | 报告期与公告日必须区分；修订缓存不能仅凭公告日冒充原始 vintage |
| `full_bal.csv/full_inc.csv/full_cf.csv/full_main.csv` | 同上三个键／日期；完整报表科目 | 合并后采用各组件保守的最晚记录公告日；流量科目需要完整 TTM 依赖 |
| `universe_a.csv` | `code` | 原财务实验代码池；不是已验证的历史全市场上市退市清单 |
| `stock_industry_monthly.csv` | `code, month, industry_code` | 位于 `GKX_CLEAN_DIR`；每月历史行业；财务行业调整没有静态行业回退 |

基本财务面板使用的关键原始科目为：`TOTAL_ASSETS`、`TOTAL_LIABILITIES`、`TOTAL_EQUITY`、`PARENT_NETPROFIT`、`TOTAL_OPERATE_INCOME`、`NETCASH_OPERATE`、`ACCOUNTS_RECE`、`INVENTORY`、`ACCOUNTS_PAYABLE`。利润表和现金流量表的流量科目通常为年初至今累计值，季度缺项不能用当前 YTD 冒充 TTM。

完整财务模块还依赖现金、长短借款、债券、固定资产、在建工程、折旧摊销、营业成本、销售与管理费用、研发、应收应付、所得税及员工数等科目。源字段与经济定义存在代理项，具体代码计算式应连同 `data_bug_fixes.md` 的限制一并阅读，不能把“可计算”当作原文口径已经验证。

## 其他修订产物

| 文件／字段 | 内容与状态 |
|---|---|
| `returns_clean.csv` 的 `is_first_observed` | 源面板首次观测月标志，非真实 IPO 日期 |
| `returns_clean.csv` 的 `eligible_asof` | 交易日与首月规则的当期标志，保留所有原始收益行 |
| `returns_clean.csv` 的 `invalid_return` | 收益非有限值或低于 −100% 的源数据问题标志 |
| 可选 `ret_train` | 仅在明确训练区间内使用训练阈值缩尾的副列；验证／测试期为空，绝不代表实际持有收益 |
| `fundamentals_monthly.csv` | `code,month,bm,ep,sp,roa,lev,agr,acc,cfp,nincr`；基于保守公告日的探索性财务面板 |
| `fundamentals_v2_monthly.csv` | `code,month` 与代码 `FEATURES` 清单中的 56 列；`divi/divo` 暂停且全缺失；行业调整及集中度在月度可见截面计算 |
| `macro_monthly.csv` 的 `dp,ep,bm,ntis` | 分别为 dy、ep、bm、chcsho 的当月原始市值加权值；每个变量单独排除缺失分子对应权重 |
| `macro_monthly.csv` 的 `mkt_ret` | 原始月收益用前一自然月市值加权；缺前月市值则不入该月加权 |
| `macro_monthly.csv` 的 `svar` | 最近 12 自然月 `mkt_ret` 样本方差，至少 6 有效月；不是原文日频已实现方差的精确复现 |
| `features_manifest.json` | 标签定义、资格定义、行数与原始源数据限制 |
| `fundamentals_manifest.json/fundamentals_v2_manifest.json` | 显式记录探索性 opt-in 和 `point_in_time_certified=false` |

财报模块默认拒绝生成未验证 vintage 的输出。`--allow-unverified-vintages` 只允许探索性执行，不提供历史版本核验。若准备发表确定性实证结论，应补入可追溯原始版本、具体字段定义及完整财务端到端验证。

## 本地数据访问与公开范围

`GKX_DATA_DIR` 读取原始缓存，`GKX_DERIVED_DIR` 写修订产物，二者必须不同。完整逐股数据和大面板不因代码公开自动获得再分发许可。本库可公开字段说明、获取代码、合成测试、汇总结果与文件哈希；原始数据的来源条款和再分发权利仍需作者核对。
