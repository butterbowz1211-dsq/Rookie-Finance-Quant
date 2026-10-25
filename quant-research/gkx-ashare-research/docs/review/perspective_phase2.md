contract_role: perspective
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: pass

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

criteria_binding_unavailable

本席位依据预先登记的 D4 标准，审阅冻结包中的主报告、双语摘要、方法附录、修订后结果、实验登记表和 README，检查机器学习概念向资产定价及投资实践的解释、复现接口的可理解性和应用边界。未读取其他审稿席位输出。本次为材料阅读与解释边界审查，没有执行实证重跑，也不据此认证数据质量或技术实现正确性。既有预承诺仍适用，无需修改评分计划。

### S1: 预测误差指标有清楚的经济解释边界

**Evidence Anchor**: text: docs/research_report.md §6：“不是年化收益、方向预测准确率或排序相关系数。”

主报告先给出零预测基准的 R² 公式，再把百分数读数解释为平方预测误差的相对改善。方法附录 D.1 与修订结果 §3 进一步列明训练均值、测试样本均值及等月平均的不同分母与权重，解释 pooled 与等月平均符号可以不同。相邻领域读者能够辨识指标究竟回答什么问题，并避免把正分数读作投资收益或可实现的事前基准。

### S2: 误差改善与截面排序的差异以实际结果说明

**Evidence Anchor**: table: docs/repaired_results.md §4，九模型 Pearson IC、Spearman Rank IC 及各自有效月数表。

相关系数的定义、单位、按有效月份平均的规则及常数预测处理均有说明。GBRT 的 1/48 月覆盖与其他模型的 48/48 月并列呈现，并明确其正 pooled R² 不保证持续的股票区分能力。这个例子将机器学习损失与金融截面排序的差别具体化，也没有从退化预测直接推断某种市场机制。

### S3: 投资含义被限定在有证据的组合诊断范围

**Evidence Anchor**: text: docs/repaired_results.md §6：“最低十分位的收益也不是可实际取得的做空损益”。

文稿区分总收益与超额收益、毛收益与净收益、最低组收益与做空损益，以及高减低名义价差与完整自融资账户。年化均值/标准差比率明确标注 rf=0，价差复合值也未包装为投资者可实现 CAGR。成本、成交、借券和资金约束缺口被直接放在组合结果附近，足以支持读者理解已计算的量及其应用限制；本席位不因尚未完成实盘评价而把一份明确自限的研究档案判为跨域推论失当。

### S4: 复现入口说明了可执行层级和数据依赖

**Evidence Anchor**: text: docs/methods_appendix.md §F.2：“每个 tag 产出 `ashare_<tag>.csv`、`predictions_<tag>.csv`、`manifest_<tag>.json`。”

README 将无需真实行情的合成最小流程与使用自备授权数据的实证流程分开。方法附录 A.1 给出字段、主键和单位，F.1—F.3 交代工作目录、环境变量、运行命令、输出用途及不同入口的限制。读者能够从研究问题追踪到相应运行入口，而不会把合成运行、哈希一致或当前代码直接等同于全部历史结果已经重现。本项评价的是接口解释与导航完整性，并非端到端运行验证。

### S5: 历史、模拟和修订证据的身份对邻近领域读者透明

**Evidence Anchor**: table: docs/experiment_registry.md §1，历史聚合文件、历史说明或日志、修订后运行及未重跑状态的定义表。

状态表将文件存在、运行绑定、当前重算与仍未验证的经济结论区分开；主报告 §5 和 §7—§8 同时避免将模拟数值当作美国真实市场成绩，或把 152 维历史 NN3 与 19 特征修订 NN3 的差值当成修复因果效应。双语摘要保持相同边界。研究贡献因此可以被理解为实现与审计基础，而不依赖未经支持的模型普遍优劣或可交易利润主张。

### S6: 可解释性输出与因果或统计证据有所区分

**Evidence Anchor**: text: docs/methods_appendix.md §C.4：“这些重要性也不是因果效应或统计显著性检验。”

附录解释树重要性与 NN 梯度平方重要性的计算对象，说明 NN 归一化顺序及 Noise 为单变量平均值。结合主报告对 PLS 成分的限定，这些说明有助于资产定价读者避免把预测性重要性直接认作定价机制或因果效应。

### Coverage Receipt

覆盖极性：Weaknesses。本席位在已读材料及 D4 授权范围内未发现需要形成负面条目的缺陷；这不是对其他维度的无缺陷声明。已披露的源数据、统计推断和交易执行不足仍是真实研究边界，但没有被文稿转换为超出证据的跨域结论。

| 契约维度 | 检查范围与未形成负面条目的依据 |
|---|---|
| D1 methodology_rigor | 不属于本席位评分权限；仅阅读相关说明以理解 D4，不对技术严谨性作无缺陷认定。 |
| D2 domain_accuracy | 不属于本席位评分权限；不替代领域事实与文献核验。 |
| D3 argumentative_coherence | 不属于本席位评分权限；不替代整体论证审查或反方压力测试。 |
| D4 cross_disciplinary_relevance | 已检查概念定义与单位、误差与排序、统计结果与投资解释、历史与修订实验身份、复现输入输出及运行导航；关键说明与限制可定位，未发现触发预承诺 warn 或 block 的歧义或重大断裂。 |
| D5 writing_and_structure | 不属于本席位评分权限；不对全文语言、图表或形式规范作结论。 |
| D6 venue_fit_and_contribution | 不属于本席位评分权限，且未绑定目标标准；不作期刊适配声明。 |
