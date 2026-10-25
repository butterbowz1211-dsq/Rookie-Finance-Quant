contract_role: domain
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: pass

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

criteria_binding_unavailable

本席位按先前盲态承诺审查资产定价与中国股票研究中的方法归属、引用含义及贡献边界。读取冻结包中的主报告、双语摘要、方法附录、修订结果、实验登记表，并补读数据字典；没有读取其他席位的报告。外部核查限定于文稿引用的三项研究及 GKX 官方附录。本卡是多角色 AI 模拟审查中的领域建议，不构成人类独立认证，也不表示整套十阶段流程已完成。

D2 的 pass 针对当前明确收窄的研究档案主张：已实现的方法、留存的历史输出、修订实验的描述性结果及其限制。它不认证原始数据、全部模型实现、模型差异的统计显著性或投资可执行性。本席位未发现触发既定 warn、block 或 fatal 标准的领域缺陷；没有变更评分计划。方法设计、统计计算及编辑判断仍由相应席位负责。

### S1: 先行研究归属与增量贡献定位准确

**Evidence Anchor**: text: docs/research_report.md §2 — “本次文献核对主要覆盖上述三篇直接相关研究及所取得的 GKX 附录”

主报告将 GKX 置于方法来源位置，同时承认中国市场已有直接先例，未把迁移到 A 股本身声称为首次贡献。吴等报告的作者、年份、样本区间、108 个特征及六类方法与[清华大学官方页面](https://thuifr.pbcsf.tsinghua.edu.cn/info/1013/2112.htm)一致；对 Leippold、Wang、Zhou 的介绍限于其摘要实际讨论的流动性、散户、期限和交易成本，未将这些研究的机制结论移植为本项目的已验证结果。该摘要可核于[出版社文章记录](https://www.sciencedirect.com/science/article/pii/S0304405X21003743)。文稿把贡献定位为可审计流程和复核依据，与现有材料的证据层级相符。

### S2: 模拟数值及评价基准的归属得到纠正

**Evidence Anchor**: text: docs/research_report.md §5 — “GKX附录中的RF/GBRT 3.35%和OLS 1.14%位于模拟Table A.1”

通过[GKX 官方论文页面](https://academic.oup.com/rfs/article/33/5/2223/5758276)提供的 Internet Appendix 核查，Table A.1 确实在模型 (a)、Pc=50 的 OOS 栏报告上述数值；附录同时注明模拟 R² 以训练样本均值为基准。文稿对数值出处的修正正确，方法附录 D.1 又区分本项目模拟的训练均值基准与 A 股主表的零预测基准，避免把模拟成绩当成美国股票实证成绩。Oracle 被解释为知道真实特征形式后估计系数的回归，也与该附录的定义相容。

### S3: 特征缩写与实际经济构念没有被混同

**Evidence Anchor**: table: docs/methods_appendix.md §A.4，19 个特征的实际计算式表及表后说明

方法附录逐项说明当前代理：高低价差不是实际报价买卖价差，动量窗口包含最近月，beta 使用原始面板等权市场收益，股本有效日期不自动等于首次公开日期。数据字典再次标明这些限制；修订结果 §2.2 明确否认 19 个代理与 GKX 原始特征逐一完全等价。这使读者能够区分字段命名、可计算公式与原文构念，未把代码可运行提升为经济定义已被精确复现。

### S4: 预测误差、排序能力与可交易收益的解释分开

**Evidence Anchor**: text: docs/repaired_results.md §6 — “最低十分位的收益也不是可实际取得的做空损益”

文稿清楚区分总收益标签与 GKX 的风险溢价语境、零预测误差改善与截面排序、rf=0 均值波动率比率与已扣真实无风险收益的 Sharpe。对 GBRT 仅 1/48 月相关系数有定义的解释保持克制，对高减低价差的复合值也没有赋予完整资金账户含义。借券、交易成本、涨跌停、停牌与成交条件被明确列为未建模内容，因此当前组合数字没有被写成可实现净收益。

### S5: 历史结果与修订证据的研究身份得到保留

**Evidence Anchor**: text: docs/abstracts.md，Chinese Abstract 结果 — “与修订后的 19 特征实验不构成同条件修复前后对照”

摘要、主报告和实验登记表一致保留历史 152 维 NN3 与修订 19 特征实验的区别；没有通过两者差值量化泄漏修复效果，也没有用新 PLS 分数认证旧 NN3 结果。对反复查看测试期、尚无差异显著性检验、历史模拟未全量重跑、退市与公司行动覆盖未闭合的披露，限制了当前主张的适用范围。保留负结果及不一致单元进一步支持其研究档案定位。

### Coverage Receipt

Polarity: Weaknesses

本席位未识别到需要登记为弱点的 D2 事项。已经明确披露的研究边界不自动构成领域表述错误；本卡也不据此替其他席位消除方法或完整性风险。

| 维度 | 检查范围与未报告该类弱点的依据 |
|---|---|
| D1: methodology_rigor | 不属于本席位评分资格；仅阅读方法定义以判断 D2 的概念和归属，不宣称该维度没有技术问题。 |
| D2: domain_accuracy | 核查三项引用及 GKX 官方模拟附录、总收益与风险溢价区分、代理变量定义、历史与新结果身份、预测与交易解释。未发现被查来源与当前核心领域主张冲突；已知差异有明确限定。 |
| D3: argumentative_coherence | 不属于本席位评分资格；未给出独立逻辑评分或替其他角色作无缺陷保证。 |
| D4: cross_disciplinary_relevance | 不属于本席位评分资格；没有评价跨学科读者覆盖或影响力。 |
| D5: writing_and_structure | 不属于本席位评分资格；没有以领域卡代替文字、版式和结构检查。 |
| D6: venue_fit_and_contribution | 不属于本席位评分资格；目标标准未绑定，未判断场所适配性；只核查了贡献陈述中的领域事实归属。 |

外部核查的实际覆盖为清华官方网页、GKX 主文与官方附录，以及 Leippold 等的出版社可检索摘要和题录。未进行广泛新文献搜索、逐项原文参数对照、原始行情审计或历史全量复算，故本卡不能扩展为这些工作的完成证明。
