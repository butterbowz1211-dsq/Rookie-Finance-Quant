contract_role: methodology
## Dimension Scores

### D1: methodology_rigor

score: warn
trigger: "局部方法描述、参数记录、版本信息、敏感性说明或统计呈现存在缺口，但已有证据足以确定主要设计、核查主要结果且未显示结论依赖该缺口；通过有限补充或澄清即可解决。"

### D2: domain_accuracy

score: not_assessed

### D3: argumentative_coherence

score: warn
trigger: "非中心表述或限定条件存在局部歧义、不一致或外推过宽，但主体证据链成立，且通过局部措辞和边界说明即可恢复准确对应。"

### D4: cross_disciplinary_relevance

score: not_assessed

### D5: writing_and_structure

score: not_assessed

### D6: venue_fit_and_contribution

score: not_assessed

## Review Body

本卡评价冻结包中五份主文档，并对数据字典、可用性声明、19特征构建器、实证runner、NN实现、运行manifest和公开月度指标作针对性核查。评价对象是有边界的公开复现研究档案；没有将其改按新资产定价规律或可执行投资策略评分。保留Phase1标准，不作评分计划异议。本轮为角色分离的AI模拟审查，未读取其他席位输出；未重训模型、未获取私有逐股面板，也未完成整套十阶段外部认证。判断为advisory范围，未发现触发本席位预承诺致命条件的证据。

### S1: 共同样本指标及源码绑定可由公开材料核查

**Evidence Anchor**: dataset: results/repaired/monthly_summary/monthly_metrics.csv；results/repaired/ashare_price19_repaired.csv；results/repaired/ashare_price19_repaired_extra.csv；两份manifest_price19_repaired运行清单的code_sha256字段

本次由432条模型月记录重新累加SSE与收益平方和，九模型均为48月、212,293股票月，所得零基准R²与两份结果CSV最大绝对差约4.44×10⁻¹⁴个百分点，低于档案所设10⁻¹⁰容差。GBRT公开相关系数有效月数确为1，其余八模型为48。包内run_ashare.py、evaluation.py、models.py、nn.py的SHA-256均与两批manifest对应字段一致。这支持共同样本的计算和来源绑定；公开聚合量本身不能独立验证私有逐股输入或源收益的经济真实性。

### S2: 预处理与时间边界具有明确可执行定义

**Evidence Anchor**: text: docs/methods_appendix.md A.5 "标签缺失的股票仍参与形成时点的特征截面"

附录区分特征形成月和下一自然月标签，说明先按当期资格排名、再筛选有标签行，并仅从训练集估计标准化参数；所读runner与这些顺序一致。方法附录还明确固定切分、不合并训练验证重估及模拟多期限边界清除协议，避免把单次程序内验证与整个研究的测试期反复探索混为一谈。

### S3: 主论证对证据范围的限制与结果相容

**Evidence Anchor**: text: docs/research_report.md §8 "没有证明线性模型普遍更适合A股"

历史152维NN3和修订19特征比较被明确分开，未把它们的差额归因为修复效果；模拟结果未作为美国实证成绩；GBRT常数预测、缺少差异推断及组合交易约束也在主文和双语摘要中一致披露。因此，新留出期、依赖稳健推断、公司行动审计与可执行净收益模型是扩大研究主张前的条件，不能仅因这些尚未完成就断言当前记录性结论已被推翻。

### W1: 缺少可公开核查的分特征缺失规模

**Severity**: Minor
**Evidence Anchor**: absence: 修订19特征实验的预处理披露 — expected 19列在训练、验证、测试各段排名前的有限值数及缺失填零比例; checked docs/methods_appendix.md A.4–A.5、docs/data_dictionary.md、docs/repaired_results.md §2、两份修订运行manifest及冻结package文件清单
**Confidence**: 4 — 已核对缺失处理文字、生成报告的代码与公开文件范围

填零规则已经说明，但读者无法判断每个特征有多少行实际使用中性排名，尤其36个月窗口与24个月入样门槛的差异会产生结构性缺失。构建器能生成总体features_report.md，冻结公开包未包含该报告，且总体覆盖不能代替最终训练/验证/测试分段覆盖。建议从已冻结面板补一张不含个股信息的聚合表，分别列出各段排名前有限值数、缺失数及比例，注明分母和是否在资格过滤后统计；不要把真实中位排名产生的零计作插补。此补充不要求改动现有模型或重新选择参数，也不据此预判结果会改变。

### W2: 构建器首部的时点保证强于方法附录所能支持的范围

**Severity**: Minor
**Evidence Anchor**: text: research/ML金融学习计划/D_A股数据/build_features.py 模块说明 "所有特征只用**第 t 月及之前**的信息"；docs/methods_appendix.md A.4 "不等于已证明该股本记录在当时首次公开"
**Confidence**: 5 — 两处原文及shares_asof的END_DATE匹配实现可以直接对照

源码说明给出无条件的信息时点保证，而方法附录正确保留了股本首次公开日期、供应商回补与全文件入口条件的限制。单独阅读代码的复现者可能据此前者误认输入已获得完整point-in-time认证。建议将首部表述限定为按记录的有效日期及形成月回溯构造，并明确未认证历史公开版本和覆盖；不必据此改写已冻结输出，也不能把当前END_DATE回溯匹配说成已证实的未来信息泄漏。问题是非中心表述边界，主文的谨慎结论仍成立。

## Arithmetic Receipts

no_recomputable_statistics: 已检查research_report.md、abstracts.md、methods_appendix.md、repaired_results.md、experiment_registry.md及对应公开指标：报告的是连续收益、R²、相关系数、组合均值/标准差和模拟参数；没有以检验统计量与p值形成的推断声明、可用于样本量反演的检验自由度或满足离散量尺条件的均值/标准差。DGP中的Student-t自由度是生成参数，不是样本检验自由度。公开SSE/收益平方和的聚合一致性核查不属于p_from_test_statistic、grim、grimmer或n_from_df四项有界程序，故不编造AR程序或收据。

