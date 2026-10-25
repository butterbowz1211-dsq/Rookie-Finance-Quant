contract_role: da
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: pass

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

criteria_binding_unavailable

校准状态：NOT_CALIBRATED。本座位使用既定盲态评分计划，未发生 Scoring Plan Dissent。此次为同模型五角色模拟审查中的 da 座位，未读取其他座位输出，角色分离不表示误差独立。D3 的 pass 仅针对冻结稿件实际提出的、受明确限制的论证；不认证数据完整性、原论文精确复现、模型普遍优越或可执行收益，也不构成目标刊物或存档平台适配判断。

### 审查范围与已成立的论证

已阅读冻结包的主报告、双语摘要、方法附录、修订结果及实验登记表；另核对两份修订运行 manifest、重放记录、公开月度聚合与重算汇总，并定向检查 A 股 runner 的资格过滤、排名、标签筛选和计分路径，以及聚合器的共同样本与常数预测处理。未读取私有逐股预测或原始市场数据，未重新训练模型，也未据此声称复核了全部模型实现或历史模拟。

稿件的有效核心是建立分层研究档案与可追踪的描述性基线。它把方法实现、历史数值记录、当前重跑和经济结论区分开，避免以当前代码回填旧实验身份。Evidence Anchor: text: docs/research_report.md §1 “本文不把这些状态统称为精确复现成功。” Confidence: 5/5，直接核对研究定位与各结果章节的限定表述。

共同样本比较的计算链条有实际可核对部分：两份 manifest 的输入哈希、19 个特征、三段样本数、测试月份及四个核心源码哈希一致。本座位从 432 条公开模型月记录重新求和，每模型均得到 48 月、212,293 个股票月，PLS、Ridge、NN3 的零基准 pooled R² 分别为约 1.142372%、1.090155%、0.886238%，与报告精度相符。这只核对公开汇总内部的计算关系。Evidence Anchor: dataset: results/repaired/monthly_summary/monthly_metrics.csv 的全部 432 条模型月记录及两份 manifest_price19_repaired 运行记录。Confidence: 5/5，直接读取并另行汇总公开数值。

### 最强反论证及其适用边界

对研究最有力的挑战是：一次共同样本上的正预测分数，并不能识别新增的经济信息；重复查看测试期、存续及退市覆盖、股本记录当时可得性、未审计极端收益、固定切分和不等调参范围，都可能塑造当前排序。把平方误差汇总一致或输入字节一致解释成经济可靠性，会把计算层面的可追踪性跨越成未经支持的市场推断。历史高维实验与当前 19 特征实验又同时改变多项条件，任何差值都不能单独识别修复效果。即使较高分模型真实降低本期误差，也仍不能推出稳定的截面优势或净投资收益。

这项反论证能够阻止更强的推广结论，但没有推翻当前稿件实际保留的贡献。主报告、双语摘要与修订结果均承认这些限制，模型排序只按本次实验描述；公开重放与聚合的一致性被限定为计算关系；历史 1.02% 也只作为需继续追查的旧记录。将这些已承认、且没有被用于推出更强结论的局限再标为“核心论证崩溃”，会把尚未开展的新研究当成本稿已作出的主张。Evidence Anchor: text: docs/abstracts.md English Abstract Conclusions “the evidence supports neither claims of attainable net profits nor universal model rankings.” Confidence: 4/5，五份文本的主张边界一致；未核验私有源数据，不能排除尚未暴露的相反证据。

### 反例、选择性报告与替代解释核查

历史与修订实验不存在被用于因果归因的前后差：正文和结果文档列明输入维度、样本、标签、切分及实现同时改变，并明确拒绝将差值解释为泄漏修复的量。当前重跑也没有被用来认证旧高维 NN3 的真实性。Evidence Anchor: text: docs/repaired_results.md §8 “不能把两者之差定量解释成” “修掉了多少泄漏”。Confidence: 5/5，直接核对该处及其逐项变化清单。

“正 R² 即有效排序”的反例被稿件主动保留。GBRT 的公开月度表只有 1 个 IC/Rank IC 有效月，训练均值基准分数为负；NN3、RF 的等月 R² 也为负。文本没有把 pooled 正值或单个有效月解释成稳定排序。公开聚合可核对相关系数有效月数；47 月预测逐股完全相同这一更细事实仍依赖作者保留的预测及诊断，不能由缺失 IC 单独重建，本座位未把它升级为独立逐股认证。Evidence Anchor: table: docs/repaired_results.md §3 四种 R² 表与 §4 相关系数有效月份表。Confidence: 5/5，对已公开的数值与解释关系有直接依据。

不同模型的容量、验证网格及早停表现仍是排序的可能解释。RF/GBRT 选中浅树，NN3 三个最佳 epoch 均为 2，不能据此推断复杂模型在 A 股无效。方法附录把实现选择、网格边界和缺失训练轨迹列明，正文也没有作这种机制归因，因此该替代解释是后续研究方向，尚不构成本稿未披露的逻辑缺陷。Evidence Anchor: text: docs/methods_appendix.md §B.4 “早停很早可提示后续优化诊断，但不能独自证明某一种欠拟合、过拟合或死亡 ReLU 原因。” Confidence: 4/5，源码调用与现存参数支持判断，缺少完整候选轨迹。

未发现冻结稿件隐去已经公开的明显失败结果来构造单一冠军叙事。登记表包含负分、重复文件、缺少重复次数及未重跑状态。不过，“保留了失败实验”不证明所有曾尝试的配置都被保存；现有登记依据已公开文件，不是全部研究尝试的事前清单。稿件已经承认历史配置、种子及轨迹缺失，因此本座位既不认定存在选择性删除，也不认证不存在选择性报告。Evidence Anchor: table: docs/experiment_registry.md §2 模拟实验状态表及 §3 历史 A 股文件登记表。Confidence: 4/5，能确认本包公开范围，无法恢复未保存的历史尝试。

毛收益组合可以由不完整可交易样本、交易约束和成本遗漏解释，公开比率不足以证明投资者可获得相应回报。正文将其限定为无成本诊断，方法附录明确高减低价差未定义完整自融资、保证金与借券账户；因而没有检出从毛收益到可实现收益的推论跳跃。Evidence Anchor: text: docs/methods_appendix.md §E “不能视为可实现的投资者 CAGR”。Confidence: 5/5，直接核对组合定义与限制。

### D3 判断与覆盖边界

以上挑战未形成达到预承诺 warn、block 或 fatal 阈值的新增未处理论证缺陷，故 D3 为 pass。本座位没有以“审计”一词替代完整性认证，也没有因为角色要求而制造否定结论。源数据真实性、私有预测细节、全部历史执行以及未经查看的外部相反证据仍超出本次核查范围；未来若获得与中心主张冲突的新证据，应重新判断。当前 CRITICAL 和 MAJOR 均为零，以下表格按协议留空；这表示没有足够依据登记这两类问题，不表示研究已获全面认证。

#### CRITICAL

| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|---|---|---|---|---|---|

#### MAJOR

| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|---|---|---|---|---|---|
