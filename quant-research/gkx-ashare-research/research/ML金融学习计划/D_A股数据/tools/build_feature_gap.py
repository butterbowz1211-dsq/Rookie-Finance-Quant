# -*- coding: utf-8 -*-
"""生成 GXK2020 A股复现的 66 项特征缺口清单（含行业哑变量）。"""
import csv, os, io
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MISSING = [
(1,"absacc","Absolute accruals","应计绝对值","应计项目的绝对值 / 平均总资产","季","A","PARENT_NETPROFIT, NETCASH_OPERATE, TOTAL_ASSETS","与 acc 同源，取绝对值"),
(3,"aeavol","Abnormal earnings announcement volume","盈余公告异常成交量","公告窗口日均成交量 / 非公告窗口日均成交量 − 1","季","A","NOTICE_DATE + returns_daily 成交量","窗口取公告日前后各 1 日"),
(4,"age","Firm age","上市年龄","公司上市（首次进入样本）至当前整年数","月","A","prices 首个交易日","东财无上市日字段，用首个行情日近似，略偏高"),
(10,"bm_ia","Industry-adjusted book-to-market","行业调整账面市值比","bm − 同行业中位数","月","A","fundamentals_monthly.bm + ind_csrc","行业中位数按大类 88 个计算"),
(11,"cash","Cash holdings","现金比率","货币资金 / 总资产","季","A","MONETARYFUNDS, TOTAL_ASSETS",""),
(13,"cashpr","Cash productivity","现金生产率","(市值 + 总负债 − 账面权益) / 货币资金","月","A","shares, close, TOTAL_LIABILITIES, TOTAL_EQUITY, MONETARYFUNDS",""),
(15,"cfp_ia","Industry-adjusted cash flow-to-price","行业调整现金流价格比","cfp − 同行业中位数","月","A","cfp + ind_csrc",""),
(16,"chatoia","Industry-adjusted change in asset turnover","行业调整资产周转率变化","Δ(营业收入/总资产) − 同行业中位数","年","A","TOTAL_OPERATE_INCOME, TOTAL_ASSETS, ind_csrc",""),
(19,"chinv","Change in inventory","存货变动","Δ存货 / 平均总资产","年","A","INVENTORY, TOTAL_ASSETS",""),
(21,"chpmia","Industry-adjusted change in profit margin","行业调整销售利润率变化","Δ(净利润/营业收入) − 同行业中位数","年","A","PARENT_NETPROFIT, TOTAL_OPERATE_INCOME, ind_csrc",""),
(23,"cinvest","Corporate investment","企业投资","资本开支相对上期变化 / 平均总资产（行业中位数调整）","年","A","CONSTRUCT_LONG_ASSET, TOTAL_ASSETS","函数形式需按原文脚注核对"),
(27,"divi","Dividend initiation","首次分红哑变量","本年派现、上年未派现 → 1","年","A","dividends.csv 年度派现额","按除权除息日归年"),
(28,"divo","Dividend omission","停止分红哑变量","上年派现、本年未派现 → 1","年","A","dividends.csv 年度派现额",""),
(31,"ear","Earnings announcement return","盈余公告异常收益","公告日 [−1,+1] 累计异常收益（相对市场）","季","A","NOTICE_DATE + returns_daily + 市场收益",""),
(32,"egr","Equity growth","权益增长率","Δ净资产 / 期初净资产","年","A","TOTAL_EQUITY",""),
(34,"gma","Gross profitability","毛利率（毛利/资产）","(营业收入 − 营业成本) / 总资产","季","A","TOTAL_OPERATE_INCOME, OPERATE_COST, TOTAL_ASSETS",""),
(35,"grCAPX","Growth in capital expenditures","资本开支增长率","Δ资本开支 / 期初资本开支","年","A","CONSTRUCT_LONG_ASSET",""),
(37,"herf","Industry concentration","行业集中度","行业内各公司营收份额的 HHI","年","A","TOTAL_OPERATE_INCOME + ind_csrc",""),
(41,"indmom","Industry momentum","行业动量","行业等权组合过去 6 个月收益","月","A","returns_monthly + ind_csrc",""),
(42,"invest","Investment","投资率","(资本开支 + Δ存货) / 总资产","年","A","CONSTRUCT_LONG_ASSET, INVENTORY, TOTAL_ASSETS",""),
(52,"mve_ia","Industry-adjusted market value","行业调整市值","ln(市值) − 同行业中位数","月","A","mvel1 + ind_csrc",""),
(54,"operprof","Operating profitability","营业利润率","(营收 − 营业成本 − 销售/管理费用 − 财务费用) / 净资产","年","A","利润表多项 + TOTAL_EQUITY","财务费用作利息费用代理"),
(55,"orgcap","Organizational capital","组织资本","Σ(销售管理费用 × 0.85^t) / 总资产","年","A","SALE_EXPENSE, MANAGE_EXPENSE","需 ≥10 年历史，2013 年后样本可用"),
(56,"pchcapx_ia","Industry-adjusted change incapex","行业调整资本开支变化率","Δ资本开支率 − 同行业中位数","年","A","CONSTRUCT_LONG_ASSET + ind_csrc",""),
(59,"pchgm_pchsale","Change in gross margin minus change in sales","毛利率变化 − 营收变化","毛利率同比变化率 − 营收同比变化率","年","A","营收, 营业成本",""),
(61,"pchsale_pchinvt","Change in sales minus change in inventory","营收变化 − 存货变化","营收同比变化率 − 存货同比变化率","年","A","营收, INVENTORY",""),
(62,"pchsale_pchrect","Change in sales minus change in receivables","营收变化 − 应收变化","营收同比变化率 − 应收账款同比变化率","年","A","营收, ACCOUNTS_RECE",""),
(63,"pchsale_pchxsga","Change in sales minus change in SG&A","营收变化 − 销管费用变化","营收同比变化率 − (销售+管理费用)同比变化率","年","A","营收, SALE_EXPENSE, MANAGE_EXPENSE",""),
(64,"pchsaleinv","Change in sales-to-inventory","营收/存货 变化率","(营收/存货) 的同比变化率","年","A","营收, INVENTORY",""),
(65,"pctacc","Percent accruals","百分比应计","应计项目 / 净利润绝对值","年","A","PARENT_NETPROFIT, NETCASH_OPERATE","净利润为负时按绝对值处理"),
(66,"pricedelay","Price delay","价格延迟","1 − 含滞后市场收益回归 R² / 仅同期市场收益回归 R²","月","A","周收益 + 市场收益","用 A 股全样本市值加权组合作市场"),
(75,"roavol","ROA volatility","ROA 波动率","近 20 个季度 ROA 的标准差","季","A","PARENT_NETPROFIT, TOTAL_ASSETS","2013 年后才有完整 5 年窗口"),
(76,"roeq","Return on equity","净资产收益率","净利润 / 平均净资产","季","A","PARENT_NETPROFIT, TOTAL_EQUITY",""),
(78,"rsup","Revenue surprise","营收意外","营收相对季节随机游走预期的标准化偏离","季","A","TOTAL_OPERATE_INCOME","需 5 年季度营收"),
(79,"salecash","Sales to cash","营收/现金","营业收入 / 货币资金","年","A","TOTAL_OPERATE_INCOME, MONETARYFUNDS",""),
(80,"saleinv","Sales to inventory","营收/存货","营业收入 / 存货","年","A","TOTAL_OPERATE_INCOME, INVENTORY",""),
(81,"salerec","Sales to receivables","营收/应收账款","营业收入 / 应收账款","年","A","TOTAL_OPERATE_INCOME, ACCOUNTS_RECE",""),
(84,"sgr","Sales growth","营收增长率","营业收入同比增速","年","A","TOTAL_OPERATE_INCOME",""),
(85,"sin","Sin stock dummy","罪恶股哑变量","白酒/烟草/博彩等 → 1","年","A","ind_csrc 大类映射","仅行业级近似；个股需主营判断"),
(87,"std_dolvol","Volatility of dollar volume","成交额波动","日成交额标准差 / 均值","月","A","returns_daily 成交额",""),
(89,"stdacc","Accrual volatility","应计波动","近 5 年应计项目标准差","年","A","PARENT_NETPROFIT, NETCASH_OPERATE",""),
(90,"stdcf","Cash flow volatility","现金流波动","近 5 年 (经营现金流/总资产) 标准差","年","A","NETCASH_OPERATE, TOTAL_ASSETS",""),
(91,"tang","Asset tangibility","有形资产比率","(固定资产 + 存货) / 总资产","季","A","FIXED_ASSET, INVENTORY, TOTAL_ASSETS",""),
(12,"cashdebt","Cash flow to debt","现金流负债比","(净利润 + 折旧摊销) / 平均总负债","年","B","需补：现金流量表折旧摊销（FA_IR_DEPR 等）",""),
(22,"chtx","Change in taxes","所得税变化","Δ所得税费用 / 平均总资产","年","B","需补：利润表 INCOME_TAX_EXPENSE",""),
(25,"currat","Current ratio","流动比率","流动资产 / 流动负债","季","B","需补：TOTAL_CURRENT_ASSETS, TOTAL_CURRENT_LIAB",""),
(26,"depr","Depreciation rate","折旧率","折旧摊销 / 期初固定资产净值","年","B","需补：折旧摊销合计",""),
(36,"grltnoa","Growth in long-term net operating assets","长期净经营资产增长率","Δ(长期经营资产 − 长期经营负债) / 平均总资产","年","B","需补：无形资产、商誉、长期借款、应付债券、递延所得税",""),
(44,"lgr","Growth in long-term debt","长期负债增长率","Δ长期负债 / 期初长期负债","年","B","需补：LONG_LOAN, BOND_PAYABLE",""),
(50,"ms","Mohanram score","Mohanram 财务打分","8 信号打分：ROA、CFO、ROA 波动、营收增长波动、研发、资本开支、广告、流动比率","年","B","需补：研发费用、流动比率、广告费用","广告费用东财无字段，需代理或置 0"),
(57,"pchcurrat","Change in current ratio","流动比率变化率","流动比率同比变化率","年","B","需补：流动资产、流动负债",""),
(58,"pchdepr","Change in depreciation rate","折旧率变化率","折旧率同比变化率","年","B","需补：折旧摊销",""),
(60,"pchquick","Change in quick ratio","速动比率变化率","速动比率同比变化率","年","B","需补：流动资产、流动负债、存货",""),
(67,"ps","Piotroski F-score","Piotroski F 分数","9 信号：ROA、CFO、ΔROA、CFO>ROA、Δ杠杆、Δ流动比率、股本、Δ毛利率、Δ周转率","年","B","需补：流动比率、股本变动","其余信号现有表可算"),
(68,"quick","Quick ratio","速动比率","(流动资产 − 存货) / 流动负债","季","B","需补：流动资产、流动负债",""),
(69,"rd","R&D increase","研发强度变化","研发费用增长率（或研发/营收变化）","年","B","需补：RESEARCH_EXPENSE",""),
(70,"rd_mve","R&D to market value","研发市值比","研发费用 / 市值","年","B","需补：研发费用",""),
(71,"rd_sale","R&D to sales","研发营收比","研发费用 / 营业收入","年","B","需补：研发费用",""),
(72,"realestate","Real estate holdings","投资性房地产占比","投资性房地产 / 总资产","年","B","需补：INVEST_REALESTATE",""),
(77,"roic","Return on invested capital","投入资本回报率","EBIT / (有息负债 + 权益)","年","B","需补：利息费用、长期借款、应付债券",""),
(92,"tb","Tax-to-book income","税会差异","所得税费用 / 税前利润 − 法定税率(25%)","年","B","需补：INCOME_TAX_EXPENSE, TOTAL_PROFIT",""),
(18,"chempia","Industry-adjusted change in employees","行业调整员工人数变化","员工人数同比变化 − 同行业中位数","年","C","需新数据：员工人数（年报/东财 F10）",""),
(24,"convind","Convertible debt indicator","可转债哑变量","期末存在可转换债券 → 1","年","C","需新数据：可转债明细（债券信息/公告）","东财仅有应付债券合计，无转换条款；可先用 0 近似"),
(38,"hire","Employee growth","员工人数增长率","员工人数同比增速","年","C","需新数据：员工人数","与 chempia 同一数据源"),
(82,"secured","Secured debt ratio","抵押负债占比","抵押借款 / 总负债","年","C","需新数据：借款抵押明细（财报附注）","附注级信息，抓取成本最高"),
(83,"securedind","Secured debt indicator","抵押负债哑变量","存在抵押借款 → 1","年","C","需新数据：借款抵押明细","与 secured 同源"),
]
HAVE = [
(2,"acc","Accruals","应计项目"),(5,"agr","Asset growth","资产增长率"),(6,"baspread","Bid-ask spread","买卖价差"),
(7,"beta","Beta","贝塔"),(8,"betasq","Beta squared","贝塔平方"),(9,"bm","Book-to-market","账面市值比"),
(14,"cfp","Cash flow-to-price","现金流价格比"),(17,"chcsho","Change in shares outstanding","股本变动率"),
(20,"chmom","Change in momentum","动量变化"),(29,"dolvol","Dollar volume","成交额"),(30,"dy","Dividend yield","股息率"),
(33,"ep","Earnings-to-price","盈利价格比"),(39,"idiovol","Idiosyncratic volatility","特质波动率"),
(40,"ill","Illiquidity","Amihud 非流动性"),(43,"lev","Leverage","杠杆率"),(45,"maxret","Maximum daily return","最大日收益"),
(46,"mom12m","12-month momentum","12 月动量"),(47,"mom1m","1-month momentum","1 月动量"),
(48,"mom36m","36-month momentum","36 月动量"),(49,"mom6m","6-month momentum","6 月动量"),
(51,"mvel1","Market value","对数市值"),(53,"nincr","Number of earnings increases","盈余连续增长"),
(73,"retvol","Return volatility","收益波动率"),(74,"roaq","ROA","资产收益率"),(86,"sp","Sales-to-price","销售价格比"),
(88,"std_turn","Volatility of turnover","换手率波动"),(93,"turn","Share turnover","换手率"),
(94,"zerotrade","Zero trading days","零成交天数占比"),
]
assert len(MISSING)==66, len(MISSING)
assert len(HAVE)==28, len(HAVE)
assert len({m[0] for m in MISSING} | {h[0] for h in HAVE}) == 94
CN = {"A":"P0 现有数据可算","B":"P1 补抓报表字段","C":"P2 附注/新数据源"}
os.makedirs(DATA, exist_ok=True)
csv_path = os.path.join(DATA, "特征缺口清单_66.csv")
with io.open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["序号","论文编号","缩写","英文名","中文名","论文定义","建议更新频率","A股状态","所需数据/需补字段","备注"])
    for i, m in enumerate(MISSING, 1):
        w.writerow([i, m[0], m[1], m[2], m[3], m[4], m[5], CN[m[6]], m[7], m[8]])
A = [m for m in MISSING if m[6]=="A"]
B = [m for m in MISSING if m[6]=="B"]
C = [m for m in MISSING if m[6]=="C"]
def table(rows):
    out = ["| # | 编号 | 缩写 | 中文名 | 定义 | 频率 | A股所需数据 | 备注 |",
           "|---|------|------|--------|------|------|--------------|------|"]
    for i, m in enumerate(rows, 1):
        note = m[8] or ""
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(i, m[0], m[1], m[3], m[4], m[5], m[7], note))
    return "\n".join(out)
md = []
md.append("# GXK(2020) A股复现：66 项特征缺口清单（含行业哑变量）\n")
md.append("> 生成脚本：tools/build_feature_gap.py；机器可读版：data/特征缺口清单_66.csv。")
md.append("> 口径：论文 94 个公司特征按 Internet Appendix Table A.6；行业哑变量单独计入，不在 94 之内。\n")
md.append("## 0. 总览\n")
md.append("| 项目 | 数量 | 说明 |")
md.append("|---|---|---|")
md.append("| 论文特征总数 | 94 | 年频 61 / 季频 13 / 月频 20 |")
md.append("| 已实现 | 28 | 19 个价格类 + 9 个财报类 |")
md.append("| **缺失** | **66** | 本清单主体 |")
md.append("| ├ P0 现有数据可算 | {} | 价格、财报现有 25 列、分红、股本、行业即可构造 |".format(len(A)))
md.append("| ├ P1 补抓报表字段 | {} | 东财还有字段没抓（流动资产、所得税、研发、折旧摊销、长期借款等） |".format(len(B)))
md.append("| └ P2 附注/新数据源 | {} | 员工人数、抵押借款、可转债 |".format(len(C)))
md.append("| 行业哑变量 | 88（大类）/ 19（门类） | 论文为 74 个 sic2 哑变量 |\n")
md.append("**结论：66 项里 {} 项（{:.0f}%）不需要新数据源，把现有表加工出来就能把覆盖率从 28/94 提到 {}/94。**\n".format(len(A), len(A)/66*100, 28+len(A)))
md.append("## 1. P0：现有数据可直接构造（{} 项）\n".format(len(A)))
md.append("所需表都在 data/：statements.csv、features_monthly.csv、returns_daily/、dividends.csv、shares.csv、industries.csv。\n")
md.append(table(A))
md.append("\n## 2. P1：需补抓报表字段（{} 项）\n".format(len(B)))
md.append("东财报表接口本来就有这些科目，只是 fetch_statements.py 没抓。补一轮抓取即可解锁。\n")
md.append("| 需补字段 | 东财字段名（建议） | 解锁特征 |")
md.append("|---|---|---|")
md.append("| 流动资产 / 流动负债 | TOTAL_CURRENT_ASSETS / TOTAL_CURRENT_LIAB | currat, quick, pchcurrat, pchquick, ps, ms |")
md.append("| 所得税费用 | INCOME_TAX_EXPENSE | chtx, tb |")
md.append("| 折旧与摊销 | FA_IR_DEPR + IA_AMORTIZE + LPE_AMORTIZE | depr, pchdepr, cashdebt |")
md.append("| 长期借款 / 应付债券 | LONG_LOAN / BOND_PAYABLE | lgr, roic, grltnoa |")
md.append("| 研发费用 | RESEARCH_EXPENSE | rd, rd_mve, rd_sale, ms |")
md.append("| 投资性房地产 | INVEST_REALESTATE | realestate |")
md.append("| 无形资产 / 商誉 | INTANGIBLE_ASSET / GOODWILL | grltnoa |")
md.append("| 利息费用 | FE_INTEREST_EXPENSE | roic、operprof 精度 |")
md.append("\n")
md.append(table(B))
md.append("\n## 3. P2：需附注或新数据源（{} 项）\n".format(len(C)))
md.append("| # | 编号 | 缩写 | 中文名 | 定义 | 数据来源 | 处理建议 |")
md.append("|---|------|------|--------|------|----------|----------|")
for i, m in enumerate(C, 1):
    md.append("| {} | {} | {} | {} | {} | {} | {} |".format(i, m[0], m[1], m[3], m[4], m[7], m[8] or "—"))
md.append("\n> 这 5 项建议最后做。convind 在 A 股可先用 0 近似（可转债发行人占比很低）；secured/securedind 需要财报附注文本，抓取成本最高，可先放弃并在复现说明里注明。")
md.append("\n## 4. 行业哑变量（sic2）\n")
md.append("论文做法：SIC 前两位 → 74 个哑变量，与 94 个特征一起进模型（Figure 5 共 95 项排序）。\n")
md.append("A 股对应物：industries.csv 的 ind_csrc，面板 5,080 只 100% 覆盖。\n")
md.append("| 层级 | 取值数 | 哑变量列数（含截距时） | 建议 |")
md.append("|---|---|---|---|")
md.append("| 门类（首段，如“制造业”） | 19 | 18 | 偏粗 |")
md.append("| 大类（如“金融业-货币金融服务”） | 88 | 87 | 粒度最接近 sic2 的 74 档，**推荐主用** |")
md.append("| 东财三级（ind_em） | 258 | 257 | 过细，小行业会共线 |")
md.append("\n注意三点：")
md.append("1. **时点问题**：industries.csv 是当前快照，用它回填 2010 年会带来前视偏差（行业变更、借壳）。论文用当期 SIC。补齐方案：按年报披露行业逐年落库，或至少标注“行业分类为当前快照”。")
md.append("2. **与行业调整特征共用同一套行业码**：bm_ia、cfp_ia、chatoia、chpmia、pchcapx_ia、indmom、herf、mve_ia 的中位数/组合都按 ind_csrc 大类算，保证与哑变量口径一致。")
md.append("3. **含截距时 drop_first**：线性模型保留 87 列会完全共线，树模型可保留全部。")
md.append("\n## 5. 数据补齐路线图（暂停 ML 之后）\n")
md.append("| 阶段 | 内容 | 产出 | 工作量 |")
md.append("|---|---|---|---|")
md.append("| S1 | 先修数据质量：退市股/幸存者偏差、配股复权、涨跌停与停牌、行业时点 | 无偏面板 | 大（需补退市股行情与配股公告） |")
md.append("| S2 | 从现有表构造 43 个 P0 特征 | 71/94 | 小（纯 transform） |")
md.append("| S3 | 扩 fetch_statements.py 抓 8 类字段，做 18 个 P1 特征 | 89/94 | 中 |")
md.append("| S4 | 员工人数等 5 个 P2 特征 | 92–94/94 | 大 |")
md.append("| S5 | 重跑 GXK 表 1/表 2 与 Figure 5，再谈 ML | 论文复现 | 中 |")
md.append("\n**顺序建议：S1 与 S2 并行，S3 紧随，S4 可选。** 理由：S1 决定所有结论是否可信（现在 RF/GBRT 的 R²oos 为负，幸存者偏差和涨跌停是主要嫌疑）；S2 成本最低、立刻把特征覆盖率从 28/94 翻到 71/94。")
md.append("\n## 6. 定义口径待核清单\n")
md.append("下列 7 项在 Table A.6 只有一句话描述，做因子前必须回原文脚注 / GHZ(2019) 附录逐条核对，否则复现口径会漂：")
md.append("cinvest、invest、grltnoa、orgcap（折旧率 15%）、ms（8 信号阈值）、ps（9 信号）、pricedelay（回归设定）。")
md.append("\n## 附：已实现 28 项\n")
md.append("| 编号 | 缩写 | 中文名 |")
md.append("|------|------|--------|")
for h in HAVE:
    md.append("| {} | {} | {} |".format(h[0], h[1], h[3]))
md_path = os.path.join(BASE, "特征缺口清单_66.md")
with io.open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("A/B/C =", len(A), len(B), len(C))
print("csv:", csv_path)
print("md chars:", len("\n".join(md)))