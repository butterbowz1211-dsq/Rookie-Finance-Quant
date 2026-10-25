# 审查意见与修改记录

作者：Gilmour Jia  
日期：2026-10-25

本次使用五个角色分别审查同一冻结稿件：方法、领域、跨学科理解、结构与贡献、反论证。各角色先写未读取正文的标准，再读取正文与证据包，提交前不共享其他角色报告。角色来自同一模型系列，可能存在相关错误；这不是独立人类同行评审，也不是投稿期刊决定。

## 原始审查意见

| 角色 | 原始记录 | 实际发现 |
|---|---|---|
| 方法 | [报告](review/methodology_phase2.md)、[事前标准](review/methodology_phase1.md) | 两项Minor：缺少分段特征缺失统计；构建器首部时点保证过强 |
| 领域 | [报告](review/domain_phase2.md)、[事前标准](review/domain_phase1.md) | 在所查三篇文献与稿件范围内未发现需登记的领域弱点 |
| 跨学科理解 | [报告](review/perspective_phase2.md)、[事前标准](review/perspective_phase1.md) | 在指定维度未发现需登记的弱点 |
| 结构与贡献 | [报告](review/eic_phase2.md)、[事前标准](review/eic_phase1.md) | 一项Minor：README对输出防覆盖的概括过宽 |
| 反论证 | [报告](review/da_phase2.md)、[事前标准](review/da_phase1.md) | CRITICAL与MAJOR条目均为零；未将已披露的局限再次包装成推翻有限主张的反证 |

“未发现”只限该角色实际阅读和检查的范围。报告中的自报置信程度没有校准，不能被解释为正确概率。五份卡的角色、预承诺触发语句和证据锚点均通过所运行的格式校验；格式通过不验证研究事实。方法卡缺少统一的无目标期刊标记，故没有把这些卡组装成一份机械认证的编辑判决；其实际方法意见仍逐条保留和处理。

## 修改与证据

| 来源 | 处理 | 核查位置与剩余边界 |
|---|---|---|
| methodology W1 | 从冻结模型输入导出19特征×3段共57行缺失统计，给出有限值数、缺失数、分母与中性排名填补比例 | [CSV](../results/repaired/feature_missingness/feature_missingness.csv)、[manifest](../results/repaired/feature_missingness/manifest.json)、[脚本](../scripts/summarize_feature_missingness.py)。计数针对最终合格且有标签样本，真实中位排名不算插补。没有重训模型或据此建立原因解释 |
| methodology W2 | 将当前builder首部限定为按记录有效日期与形成月回溯，明确历史公开版本和覆盖未认证 | 保留[实际执行字节副本](../results/repaired/executed_source/build_features.py)；[说明修订记录](../results/repaired/source_annotation_revision.json)登记旧/新哈希，去除模块说明后的AST完全相同。旧执行哈希不被覆盖 |
| eic W1 | README改为已有指标文件时拒绝相同tag，并要求新目录、新tag，说明部分输出或缺manifest不代表成功 | [README](../README.md)与[方法附录F.2](methods_appendix.md)同步；未声称增加了事务性写入机制 |
| 术语辅助检查 | 增加[术语速查](glossary.md)，修正GitHub数学公式展示 | 供读者查阅，不作为新实证证据 |

补充缺失统计发现：训练段`mom36m`有83,878/178,488行（46.99%）在排名前缺失；验证段27,677/119,719行（23.12%），测试段18,078/212,293行（8.52%）。这些量已纳入报告和独立路线图。它们解释了需要进一步审查的覆盖条件，不证明缺失是任何模型成绩的原因。

## 完成范围与未完成事项

本轮调用学术写作、研究流程、角色审查和数据可用性技能，完成证据整理、实际代码修订、36项回归测试、九模型共同样本重跑、文稿与数据声明以及上述反馈处理。初始文稿核查另注册了60条精确字节绑定声明；登记覆盖与主张正确性仍是不同问题。

本次采用已获作者授权的连续整理与发布流程，未宣称完成技能中全部人工检查点、十二代理写作协议或完整十阶段认证。共享覆盖工具的首次尝试因依赖导入权限失败，没有据此伪造覆盖通过记录；五角色格式校验另在可访问依赖的环境中实际运行。没有把有限原创性片语抽查称为完整查重，也没有把同模型多角色称为跨模型独立验证。

尚需补充的研究工作见[ROADMAP](../ROADMAP.md)，包括信息可得性、入口条件与缩尾参考池、缺失处理、年度前推、新留出期、统计不确定性、旧高维实验复建和实际交易约束。源数据真实性与权利、完整历史运行和净收益可执行性没有被本次审查认证。

## 人与AI协作记录 / Human–AI collaboration

Gilmour Jia保持文稿与项目代码的作者署名。Codex协助资料盘点、代码审查与局部修订、测试和新实验、报告整合、图表与同模型多角色检查；DeepSeek的既有协助按作者陈述和历史记录披露，本轮没有调用DeepSeek服务。最终研究选择与发表责任属于作者。

Gilmour Jia remains the named author of the manuscript and project code. Codex assisted with evidence organization, code changes, tests, selected empirical runs, documentation, figures, and role-separated checks within the same model family. Prior DeepSeek assistance is disclosed from the author's account and historical records; no DeepSeek service was called in this run. This record does not establish independent human review, a journal decision, or a complete pipeline certification.
