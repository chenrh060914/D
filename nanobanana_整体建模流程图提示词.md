# NanoBanana 整体建模流程图提示词

## 2026年MCM C题：与星共舞（Dancing with the Stars）

---

## 一、提示词（中文版）

### 主提示词

```
请为我生成一张精美的学术风格流程图，展示数学建模竞赛（MCM）C类大数据分析题目的完整建模工作流程。

这是一个关于美国真人秀节目《与星共舞》（Dancing with the Stars）的投票数据分析项目，包含以下核心模块：

【整体流程架构】
采用自上而下的分层流程图设计，包含6个主要阶段，呈现清晰的逻辑递进关系：

第一层：数据输入层
- 核心数据（421条×53字段）：评委评分、选手信息、比赛结果
- 补充数据：社交媒体粉丝数据（辅助参考）

第二层：数据预处理阶段
- 缺失值处理（N/A→np.nan）
- 异常值检测（IQR方法）
- 赛季规则标记（排名法/百分比法/评委决定机制）
- 特征工程（评分计算、类别编码）

第三层：四问建模求解阶段（核心层，并列展示）
- 问题1：粉丝投票估算模型
  • 约束优化 + 贝叶斯推断双方案融合
  • 输出：淘汰预测准确率100%，置信区间±14.4%
  
- 问题2：投票合并方法对比
  • 随机森林 + SHAP分析
  • 输出：差异率28.36%，识别3个高争议案例
  
- 问题3：名人特征影响分析
  • 线性回归 + 随机森林双模型验证
  • 输出：年龄重要性75.4%，CV R²=0.13
  
- 问题4：新投票系统设计
  • 强化学习 + 动态权重机制
  • 输出：争议率降低8.7pp，公平淘汰率+17pp

第四层：结果分析与检验阶段
- 基础分析：数值解读、模型对比
- 深层分析：关联性验证、敏感性分析
- 有效性检验：交叉验证、残差分析
- 鲁棒性验证：参数扰动、泛化测试

第五层：论文撰写阶段
- 开篇三要素（题目、摘要、关键词）
- 核心基础章节（问题重述、假设、符号说明）
- 数据处理章节（预处理、参数校准）
- 模型求解章节（四问完整求解）
- 收尾章节（结论、改进、备忘录）

第六层：成果输出层
- 25页完整报告
- 1-2页备忘录（给节目制作人的建议）
- AI使用报告

【视觉设计要求】
- 整体配色：采用蓝色、绿色、橙色为主色调，符合学术论文风格
- 每个主要阶段使用不同的浅色背景区分
- 箭头使用实线表示主流程，虚线表示数据依赖关系
- 问题1-4并列展示，用连接线表示依赖关系（问题2/3依赖问题1的估算结果，问题4依赖前三问结论）
- 在关键节点标注核心指标（如准确率、R²值等）
- 添加小图标增强视觉效果（数据图标、模型图标、文档图标等）
- 整体布局采用纵向流程图，横向展开问题1-4的并列结构
- 图片分辨率高清，适合A4打印

【文字标注】
- 所有文字使用学术论文常用字体（Times New Roman或Arial）
- 中英文混排，关键术语保留英文（如Random Forest, SHAP, Bayesian Inference等）
- 核心数据使用加粗突出显示
```

---

## 二、提示词（英文版）

### Main Prompt

```
Please generate an elegant academic-style flowchart illustrating the complete modeling workflow for an MCM (Mathematical Contest in Modeling) Category C big data analysis problem.

This is a voting data analysis project for the American reality TV show "Dancing with the Stars (DWTS)". The flowchart should include the following core modules:

【Overall Architecture】
Design a top-down hierarchical flowchart with 6 main stages showing clear logical progression:

Layer 1: Data Input Layer
- Core Dataset (421 records × 53 fields): judge scores, contestant info, competition results
- Supplementary Data: social media follower data (auxiliary reference)

Layer 2: Data Preprocessing Stage
- Missing value handling (N/A → np.nan)
- Outlier detection (IQR method)
- Season rule labeling (Ranking/Percentage/Judge-Save mechanism)
- Feature engineering (score calculation, categorical encoding)

Layer 3: Four-Problem Modeling & Solving Stage (Core layer, displayed in parallel)
- Problem 1: Fan Vote Estimation Model
  • Constrained Optimization + Bayesian Inference dual-scheme fusion
  • Output: Elimination prediction accuracy 100%, CI ±14.4%
  
- Problem 2: Vote Aggregation Method Comparison
  • Random Forest + SHAP Analysis
  • Output: Difference rate 28.36%, 3 high-controversy cases identified
  
- Problem 3: Celebrity Feature Impact Analysis
  • Linear Regression + Random Forest dual-model validation
  • Output: Age importance 75.4%, CV R²=0.13
  
- Problem 4: New Voting System Design
  • Reinforcement Learning + Dynamic Weighting mechanism
  • Output: Controversy rate -8.7pp, Fair elimination rate +17pp

Layer 4: Results Analysis & Validation Stage
- Basic Analysis: numerical interpretation, model comparison
- In-depth Analysis: correlation validation, sensitivity analysis
- Validity Testing: cross-validation, residual analysis
- Robustness Testing: parameter perturbation, generalization tests

Layer 5: Paper Writing Stage
- Opening Elements (title, abstract, keywords)
- Core Foundation Chapters (problem restatement, assumptions, symbol definitions)
- Data Processing Chapter (preprocessing, parameter calibration)
- Model Solving Chapter (complete solutions for all four problems)
- Closing Chapters (conclusions, improvements, memo)

Layer 6: Output Deliverables Layer
- 25-page complete report
- 1-2 page memo (recommendations for show producers)
- AI usage report

【Visual Design Requirements】
- Color scheme: blue, green, orange as primary colors, academic paper style
- Use different light background colors to distinguish each main stage
- Solid arrows for main flow, dashed arrows for data dependencies
- Problems 1-4 displayed in parallel, with connecting lines showing dependencies
- Annotate key metrics at critical nodes (accuracy, R² values, etc.)
- Add small icons for visual enhancement (data icons, model icons, document icons)
- Overall layout: vertical flowchart with horizontal expansion for Problems 1-4
- High-resolution image suitable for A4 printing

【Text Annotations】
- Use academic fonts (Times New Roman or Arial)
- Bilingual labeling where appropriate, key terms in English
- Bold formatting for core metrics
```

---

## 三、补充提示词（细节强化版）

### 如需更详细的流程图，可使用以下补充提示词：

```
请在上述流程图基础上，添加以下细节：

【问题依赖关系图】
在问题1-4区域，用带箭头的连接线清晰展示依赖关系：
- 问题2 ←← 问题1（问题2使用问题1的粉丝投票估算结果）
- 问题3 ←← 问题1（问题3使用问题1的估算结果分析特征影响）
- 问题4 ←←← 问题1+问题2+问题3（问题4综合前三问结论设计新系统）

【方法模型标注】
为每个核心模型添加小图标和简要公式：
- 约束优化：min L(V) = λ||V-V_prior||² s.t. 淘汰约束
- 贝叶斯推断：V ~ Dirichlet(α)
- 随机森林：特征重要性排序
- 强化学习：状态-动作-奖励循环图

【数据流向标注】
- 用不同颜色的箭头区分数据流向：
  • 蓝色：原始数据流
  • 绿色：处理后数据流
  • 橙色：模型输出流
  • 紫色：反馈验证流

【时间节点标注】
在流程图侧边添加时间轴参考：
- 数据预处理：第1天
- 模型求解：第2-3天
- 结果分析：第3天
- 论文撰写：第3-4天

【创新点高亮】
用星形或勋章图标标注三个创新点：
★ 创新1：约束优化+贝叶斯推断双方案融合框架
★ 创新2：争议率+公平淘汰率双维度评估体系
★ 创新3：强化学习投票规则参数自动优化
```

---

## 四、精简版提示词（快速生成）

### 如果需要快速生成简化版流程图：

```
请生成一张学术风格的建模流程图，展示以下六个阶段：

1. 数据输入 → 2. 数据预处理 → 3. 四问建模求解（并列：问题1投票估算、问题2方法对比、问题3特征分析、问题4系统设计）→ 4. 结果分析检验 → 5. 论文撰写 → 6. 成果输出

核心要点：
- 问题1：约束优化+贝叶斯推断，准确率100%
- 问题2：随机森林+SHAP，差异率28.36%
- 问题3：双模型验证，年龄重要性75.4%
- 问题4：强化学习，争议率降低8.7pp

风格：学术论文风格，蓝绿橙配色，高清分辨率
```

---

## 五、使用建议

### 5.1 生成后检查要点

| 检查项 | 要求 | 自检 |
|-------|------|------|
| 阶段完整性 | 包含6个主要阶段 | □ |
| 问题并列 | 问题1-4并列展示且有依赖关系线 | □ |
| 指标标注 | 核心数据清晰可见 | □ |
| 视觉效果 | 配色协调，图标得体 | □ |
| 分辨率 | 高清适合打印 | □ |
| 文字清晰 | 中英文混排无错误 | □ |

### 5.2 微调建议

- 如果生成的图过于复杂，可要求"简化为3层结构"
- 如果配色不满意，可指定"使用科技蓝+学术灰配色"
- 如果箭头不清晰，可要求"加粗箭头线条，增大箭头尺寸"
- 如果想要更学术化，可要求"添加LaTeX公式标注"

### 5.3 最终输出文件命名建议

- 中文版：`图1_整体建模工作流程图.png`
- 英文版：`Figure1_Overall_Modeling_Workflow.png`

---

## 六、项目核心信息速查

供生成图片时参考的核心数据：

| 模块 | 核心方法 | 关键指标 |
|------|---------|---------|
| 数据预处理 | 缺失值处理+特征工程 | 421条→97字段 |
| 问题1 | 约束优化+贝叶斯推断 | 准确率100%，CI±14.4% |
| 问题2 | 随机森林+SHAP | 差异率28.36%，CV准确率0.61 |
| 问题3 | 线性回归+随机森林 | 年龄重要性75.4%，CV R²=0.13 |
| 问题4 | 强化学习+动态权重 | 争议率-8.65pp，公平率+17.01pp |

---

**文档生成时间**：2026年MCM竞赛

**适用对象**：2026年MCM C题参赛团队

**文档版本**：v1.0

**使用工具**：NanoBanana AI图像生成
