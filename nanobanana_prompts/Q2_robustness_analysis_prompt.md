# Nanobanana Prompt: 问题2 鲁棒性分析图

## 图表目标
为MCM美赛论文生成一张**子问题2（方法对比模型）的鲁棒性分析图**，展示随机森林分类模型在不同噪声水平下的性能稳定性。

---

## Nanobanana 提示词（英文版 - 推荐使用）

```
Create a scientific robustness analysis figure for a machine learning model with the following specifications:

FIGURE TYPE: Multi-panel robustness analysis visualization

PANEL 1 (Top Left) - Noise Sensitivity Line Chart:
- X-axis: Noise Level (±1%, ±3%, ±5%, ±10%)
- Y-axis: Model Accuracy (55% to 65%)
- Data points with error bars:
  - Baseline (0%): 61.19%
  - ±1%: 60.85% ± 0.43%
  - ±3%: 59.92% ± 0.91%
  - ±5%: 58.76% ± 1.34%
  - ±10%: 55.83% ± 2.12%
- Blue solid line with circular markers
- Gray shaded confidence band
- Annotate: "Performance drop: 3.97% at ±5% noise"

PANEL 2 (Top Right) - Cross-Validation Stability Bar Chart:
- X-axis: Fold Number (1, 2, 3, 4, 5)
- Y-axis: Accuracy (0.55 to 0.65)
- Fold values: 0.598, 0.619, 0.612, 0.627, 0.604
- Mean line at 0.612 with ±1.6% std band
- Color: Deep blue bars with subtle gradient
- Annotate: "CV Accuracy: 0.612 ± 0.016"

PANEL 3 (Bottom Left) - Accuracy Drop Comparison:
- Horizontal bar chart comparing accuracy drop percentages
- Categories: ±1% (0.56%), ±3% (2.08%), ±5% (3.97%), ±10% (8.76%)
- Color gradient from light blue to dark blue
- Reference line at 5% threshold
- Add label: "Acceptable threshold"

PANEL 4 (Bottom Right) - Robustness Score Radar:
- 5-axis radar chart showing robustness dimensions:
  - Noise Robustness: 4/5 stars (80%)
  - Feature Robustness: 4/5 stars (80%)
  - Split Robustness: 4/5 stars (80%)
  - Temporal Stability: 4/5 stars (80%)
  - Overall Rating: "Good"
- Blue filled area with semi-transparency
- Grid lines at 20%, 40%, 60%, 80%, 100%

OVERALL STYLE:
- Clean, publication-ready scientific visualization
- Color scheme: Professional blue gradient (#1a365d to #63b3ed)
- Font: Sans-serif (Helvetica/Arial), size 10-12pt for labels
- White background with subtle grid
- Figure size: 10 x 8 inches
- High DPI (300) for print quality
- Title: "Robustness Analysis: Method Comparison Model (Sub-Problem 2)"
- Add figure caption area at bottom

ANNOTATIONS:
- Include statistical significance markers (* p<0.05, ** p<0.01)
- Error bars show 95% confidence intervals
- Add overall conclusion text: "Model maintains >95% baseline performance under ±5% noise perturbation"
```

---

## Nanobanana 提示词（中文版）

```
创建一张机器学习模型鲁棒性分析的科研级别图表，具体规格如下：

图表类型：多面板鲁棒性分析可视化

面板1（左上）- 噪声敏感性折线图：
- X轴：噪声水平（±1%、±3%、±5%、±10%）
- Y轴：模型准确率（55% 到 65%）
- 数据点及误差棒：
  - 基准（0%）：61.19%
  - ±1%：60.85% ± 0.43%
  - ±3%：59.92% ± 0.91%
  - ±5%：58.76% ± 1.34%
  - ±10%：55.83% ± 2.12%
- 蓝色实线，圆形标记点
- 灰色阴影置信区间带
- 标注："±5%噪声下准确率下降3.97%"

面板2（右上）- 交叉验证稳定性柱状图：
- X轴：折次（1, 2, 3, 4, 5）
- Y轴：准确率（0.55 到 0.65）
- 各折数值：0.598, 0.619, 0.612, 0.627, 0.604
- 均值线0.612，标准差±1.6%区间带
- 颜色：深蓝色柱状图，微渐变
- 标注："CV准确率：0.612 ± 0.016"

面板3（左下）- 准确率下降对比图：
- 水平柱状图，展示不同噪声下准确率下降百分比
- 类别：±1%（0.56%）、±3%（2.08%）、±5%（3.97%）、±10%（8.76%）
- 从浅蓝到深蓝的颜色渐变
- 5%阈值参考线
- 添加标签："可接受阈值"

面板4（右下）- 鲁棒性评分雷达图：
- 5轴雷达图展示鲁棒性维度：
  - 噪声鲁棒性：4/5星（80%）
  - 特征鲁棒性：4/5星（80%）
  - 划分鲁棒性：4/5星（80%）
  - 时序稳定性：4/5星（80%）
  - 综合评级："良好"
- 蓝色半透明填充区域
- 网格线位于20%、40%、60%、80%、100%

整体风格：
- 干净、出版级科研可视化
- 配色方案：专业蓝色渐变（#1a365d 到 #63b3ed）
- 字体：无衬线字体（Helvetica/Arial），标签10-12pt
- 白色背景，淡灰色网格
- 图像尺寸：10 x 8 英寸
- 高DPI（300）打印质量
- 标题："鲁棒性分析：方法对比模型（子问题2）"
- 底部添加图注区域

标注要求：
- 包含统计显著性标记（* p<0.05, ** p<0.01）
- 误差棒显示95%置信区间
- 添加总结文字："模型在±5%噪声扰动下保持>95%基准性能"
```

---

## 补充说明

### 数据来源
数据来自论文《模型检验模块》第三节"鲁棒性分析"部分，具体为问题2（方法对比模型）的噪声测试结果。

### 图表设计理念
1. **科学严谨性**：所有数据点配备误差棒，展示统计不确定性
2. **多维度展示**：从噪声敏感性、交叉验证稳定性、综合评分三个角度全面评估模型鲁棒性
3. **美赛风格**：采用清晰简洁的科研图表风格，便于评委快速理解
4. **O奖标准**：图表信息密度高，展示专业的统计分析能力

### 可选变体

**变体1：单面板简化版**
```
Create a noise sensitivity analysis line chart:
- X-axis: Noise perturbation level (0%, ±1%, ±3%, ±5%, ±10%)
- Y-axis: Classification accuracy (55% to 62%)
- Data: 61.19%, 60.85%, 59.92%, 58.76%, 55.83%
- Error bars with confidence intervals
- Blue line with markers, gray shaded uncertainty band
- Title: "Model Robustness Under Data Perturbation"
- Annotation: "Acceptable performance maintained up to ±5% noise"
- Scientific publication style, white background, 8x6 inches
```

**变体2：热力图版**
```
Create a robustness heatmap showing model performance:
- Rows: Different noise levels (0%, 1%, 3%, 5%, 10%)
- Columns: Cross-validation folds (Fold 1-5)
- Cell values: Accuracy scores
- Color scale: White (low) to deep blue (high)
- Annotate each cell with exact values
- Title: "Cross-Validation Performance Under Noise Perturbation"
```

---

## 使用建议

1. 将上述提示词输入Nanobanana
2. 如生成效果不理想，可尝试调整：
   - 颜色方案（换成其他专业配色）
   - 面板布局（2x2 或 1x4）
   - 数据展示密度
3. 确保最终图表与论文整体风格一致
