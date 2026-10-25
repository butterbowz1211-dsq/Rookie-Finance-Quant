contract_role: eic
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: warn
trigger: "局部术语、重复叙述、交叉引用或图表说明不清，但读者仍能识别核心研究问题与证据链，且可通过有限编辑修复。"

### D6: venue_fit_and_contribution
score: pass

## Review Body

本卡维持盲态预承诺，审阅范围为主报告、双语摘要、方法附录、修订结果、实验登记、README、ROADMAP及主结果图。D5 的警告对应一处入口说明与详细附录之间的局部表达不一致；主要研究问题、历史与当前结果的身份、证据限制均可辨认。D6 的通过只表示文稿所限定的复现档案与审计贡献已有明确定位和可追踪的材料指引，不表示新的资产定价结论已经成立，也不是数据、算法或结果的外部认证。本次未重跑模型、复算全部数值或核验第三方原始数据，不作所分配维度以外的评分。

criteria_binding_unavailable

### S1: 贡献定位与既有研究的关系交代清楚

**Evidence Anchor**: text: docs/research_report.md §2，第1段，“本身不能作为首次提出的贡献”；docs/abstracts.md，Chinese Abstract 结论，“贡献主要在可审计流程与复核依据”。

主报告承认机器学习在中国股票市场上的直接先例，摘要与结论把实际增量限定为研究记录整理、实现核查与修订实证的对应关系，没有把应用场景本身或一个较高分数当作新方法贡献。这一定位允许档案在尚无新模型架构、差异推断或净收益证明时仍具有明确的研究用途。D6 的判断据此成立，不依赖任何臆造的期刊门槛。

### S2: 实验身份与完成状态分层有效

**Evidence Anchor**: table: docs/experiment_registry.md §1 状态表、§3 历史 A 股版本表及§4 修订后运行标签表。

登记表把历史聚合文件、文字记录、修订后运行与未重跑状态分别处理，给出具体文件和缺失记录。主报告与摘要保留历史 152 维 NN3 和当前 19 特征实验的差异，读者能识别哪些是旧输出、哪些是新执行、哪些是计划。形成月、目标月、名义窗口和实际有效训练月也分别交代。这些组织选择直接支持档案的可读性与后续核查。

### S3: 图表将误差改善和排序信息分开表达

**Evidence Anchor**: figure: docs/research_report.md 图1，docs/figures/repaired_model_comparison.png，左右两个面板及总标题。

图示分别给出零预测基准 R² 与月度 Rank IC，坐标标出不同单位，逐模型标注有效月份，并在图内说明固定切分、总收益和无置信区间。GBRT 的 1/48 覆盖被直接显示，正文又解释常数预测与平方误差改善能够同时出现。图表因此支持已完成实验的描述，而没有用统一外观掩盖指标或覆盖率差异。

### S4: 后续计划与已有成果的边界可辨认

**Evidence Anchor**: text: ROADMAP.md，开篇，“下面是未完成或需要更强证据的工作，不是已经取得的结果”；docs/methods_appendix.md §G.3，“无法恢复的历史事实应标注未记录”。

路线图按数据、股票池、时间设计、推断和执行约束组织后续项目，并给出完成判据；方法附录则说明无法从现有记录恢复的内容。两者没有将计划作为既有成果，也没有用当前配置回填历史运行事实。公开材料与本地输入的范围、计算一致性与原始数据真实性的区别，在主报告和附录中均有交代，支持读者理解审计贡献的实际边界。

### W1: README 将单项防覆盖检查概括为全部输出保护

**Severity**: Minor
**Evidence Anchor**: text: README.md，“使用自己有权访问的数据”末段，“已有输出拒绝覆盖”；docs/methods_appendix.md §F.2，“已有指标文件时 runner 拒绝相同 tag”。
**Confidence**: 5 — 同一文稿包内可直接核对的措辞范围差异。

README 对输出保护的概括比方法附录所述实现范围更广。附录已明确：模型逐步重写指标与预测 checkpoint，最终才写 manifest，防重名检查针对已有指标文件，不能视为所有输出的事务保护。新读者若只按 README 操作，可能误以为任意残留文件都必然被拦截。这是运行说明的局部清晰度问题，不改变文稿所呈现的核心研究证据。

建议把 README 的对应句改为“已有指标文件时拒绝相同 tag；每次新实验使用新的输出目录和 tag，部分文件或缺少 manifest 不代表完整成功运行”，并链接方法附录 §F.2。该修改只需同步说明，不要求新增实验或改变历史结果。
