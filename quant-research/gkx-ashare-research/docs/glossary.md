# 术语与指标速查

作者：Gilmour Jia

| 缩写或术语 | 本档案中的含义 | 阅读时需区分 |
|---|---|---|
| GKX | Gu、Kelly、Xiu三位作者的简称 | 引用原论文，不是本项目的新模型名称 |
| PLS | Partial Least Squares，偏最小二乘 | 成分选择使用响应信息 |
| PCR | Principal Component Regression，主成分回归 | 先依据特征协方差降维 |
| OLS | Ordinary Least Squares，普通最小二乘 | 线性基线；高维和秩亏时需检查 |
| Ridge / Lasso / ENet | 岭回归、Lasso、Elastic Net（弹性网） | 不同正则化；方法家族、估计器变体与独立实验数不可混用 |
| RF | Random Forest，随机森林 | 多棵随机化回归树的集成 |
| GBRT | Gradient Boosted Regression Trees，梯度提升回归树 | 顺序提升，与RF的随机化树集成不同 |
| NN / NN3 | Neural Network，神经网络；NN3为三隐层变体 | 本轮不是对所有NN1—NN5的A股实证重跑 |
| GLM | 本模拟代码中的Group Lasso样条模型 | 此处不是通常泛指的Generalized Linear Model；应以方法附录目标函数为准 |
| DGP | Data-Generating Process，数据生成过程 | 模拟中已知的信号与噪声机制，不是市场真实机制已获证明 |
| Oracle | 使用已知生成机制信息的模拟参照 | 实际市场中无法直接获得的参照 |
| MSE | Mean Squared Error，均方误差 | 验证集选参指标；不是方向准确率 |
| SSE | Sum of Squared Errors，误差平方和 | 合并R²计算使用平方和比值 |
| OOS | Out of Sample，样本外 | 不自动意味着测试期从未被开发者查看 |
| R² | 相对指定基准的平方预测误差改善比例 | 本库区分零预测、测试均值、训练均值和等月平均口径 |
| IC | Information Coefficient，信息系数 | 本库为月度截面Pearson相关；常数预测时未定义 |
| Rank IC | 秩信息系数 | 本库为月度截面Spearman相关；必须报告有效月份数 |
| TTM | Trailing Twelve Months，过去12个月累计 | 财务流量需有完整季度/同期依赖，不能把缺失TTM替换为当年累计值 |
| vintage | 某个时点实际可获得的数据版本 | 当前下载的历史报表可能已被事后修订 |
| ST | 中国股票市场的特别处理标记 | 本轮19特征样本尚未完成完整历史ST筛选 |
| CAGR | Compound Annual Growth Rate，复合年增长率 | 与“月均收益先年化”不同；组合资金账户解释仍需成立 |
| rf | Risk-free rate，无风险利率 | 本轮总收益比率中显式假设rf=0；未实接无风险序列 |
| SHA-256 | 文件内容校验哈希算法 | 验证字节一致性，不能认证数据真实性或研究设计 |
| CSV / JSON / JSONL | 表格、结构化对象、逐行结构化对象文件 | 文件格式本身不能证明运行次数、去重或完整性 |
| manifest | 实验文件与配置清单 | 需核对实际保存字段；未记录内容不能由清单名称推断 |
| 回归测试 | 修订软件后的自动检查 | 与计量经济学的回归模型不是同一概念 |
| AI辅助审查 | 使用人工智能工具检查代码、数据关系和文字 | 不能替代作者判断或独立人类同行评审 |

详细计算定义与实现偏离见[方法附录](methods_appendix.md)。
