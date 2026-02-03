# Nanobanana Prompt: Figure 7.3 - Residual Distribution and Normality Check

## 图表用途
用于MCM论文模型检验章节，展示粉丝投票估算模型的残差分析结果，验证模型预测误差是否服从正态分布。

---

## 🎨 Nanobanana Prompt (English)

```
Create a high-quality academic research figure with the following specifications:

FIGURE TITLE:
"Figure 7.3: Residual Distribution and Normality Check for Fan Vote Estimation Model"

LAYOUT: Two side-by-side subplots in a 1×2 grid arrangement, professional scientific paper style

LEFT SUBPLOT - Residual Histogram:
- Title: "Residual Distribution"
- X-axis label: "Residual Value"
- Y-axis label: "Frequency" (or "Density")
- A histogram showing residual distribution with approximately 25-30 bins
- A smooth red normal distribution curve overlaid on the histogram (Gaussian fit)
- Histogram bars in light blue/steel blue color with subtle transparency (alpha=0.7)
- The normal curve should be smooth, bold red line (linewidth=2)
- Include a legend showing "Observed" and "Normal Fit"
- Grid lines with light gray color and low opacity

RIGHT SUBPLOT - Q-Q Plot:
- Title: "Normal Q-Q Plot"
- X-axis label: "Theoretical Quantiles"
- Y-axis label: "Sample Quantiles"
- Scatter points (blue circles) representing the residuals
- A diagonal 45-degree reference line in red (dashed or solid)
- Points should closely follow the diagonal line to indicate normality
- Include confidence bands (shaded region) around the reference line if possible

ANNOTATION (below the main title):
"Shapiro-Wilk Test: W = 0.9856, p = 0.0823 (p > 0.05, residuals approximately normal)"

STYLE REQUIREMENTS:
- Clean, minimalist academic style suitable for Nature/Science publications
- White background
- Font: Arial or Helvetica, size 10-12pt for labels
- Color scheme: Blue histogram, red fit curves, professional grayscale grid
- High resolution (300 DPI minimum)
- Tight layout with appropriate spacing between subplots
- No decorative elements - pure scientific visualization
- Include subtle box/frame around each subplot

ADDITIONAL DETAILS:
- The residual distribution should appear approximately symmetric and bell-shaped
- Mean ≈ 0.0023, Standard deviation ≈ 0.0768
- Q-Q plot points should mostly fall along the diagonal with minor deviations at tails
- Overall figure dimensions: approximately 10 inches × 4 inches (landscape orientation)
```

---

## 🎨 Nanobanana Prompt (中文版)

```
创建一张高质量学术研究图表，具体规格如下：

图表标题：
"Figure 7.3: Residual Distribution and Normality Check for Fan Vote Estimation Model"

布局：1×2 并排双子图布局，专业科学论文风格

左子图 - 残差直方图：
- 子标题："Residual Distribution"
- X轴标签："Residual Value"
- Y轴标签："Frequency"（或"Density"）
- 绘制残差分布直方图，约25-30个柱状条
- 叠加一条平滑的红色正态分布拟合曲线
- 直方图柱状条使用浅蓝色/钢蓝色，带适度透明度(alpha=0.7)
- 正态曲线使用粗红色线条(线宽=2)
- 包含图例，显示"Observed"和"Normal Fit"
- 添加浅灰色网格线

右子图 - Q-Q图：
- 子标题："Normal Q-Q Plot"
- X轴标签："Theoretical Quantiles"
- Y轴标签："Sample Quantiles"
- 散点图（蓝色圆点）表示残差数据
- 一条红色45度参考对角线（虚线或实线）
- 数据点应紧密沿对角线分布，表明残差服从正态分布
- 如可能，添加置信区间阴影带

标注（位于主标题下方）：
"Shapiro-Wilk Test: W = 0.9856, p = 0.0823 (p > 0.05, residuals approximately normal)"

样式要求：
- 简洁、极简的学术风格，适合Nature/Science期刊发表标准
- 白色背景
- 字体：Arial或Helvetica，标签字号10-12pt
- 配色方案：蓝色直方图、红色拟合曲线、专业灰色网格
- 高分辨率（最低300 DPI）
- 紧凑布局，子图间距适当
- 无装饰性元素——纯粹的科学可视化
- 每个子图添加细微边框

数据细节：
- 残差分布应呈现近似对称的钟形曲线
- 均值 ≈ 0.0023，标准差 ≈ 0.0768
- Q-Q图数据点主要沿对角线分布，尾部允许轻微偏离
- 整体图表尺寸：约10英寸×4英寸（横向布局）
```

---

## 📊 参考数据（来自模型检验模块）

根据论文中的模型检验结果，残差分析的关键统计量：

| 统计量 | 数值 | 说明 |
|-------|------|------|
| 残差均值 | 0.0023 | ≈0，无系统性偏差 |
| 残差标准差 | 0.0768 | 估算波动较小 |
| 残差偏度 | 0.12 | 近似对称分布 |
| 残差峰度 | -0.08 | 近似正态峰度 |
| Shapiro-Wilk W值 | 0.9856 | 接近1，近似正态 |
| Shapiro-Wilk p值 | 0.0823 | >0.05，无法拒绝正态性假设 |

---

## 🎯 图表设计要点

### O奖级别图表标准：
1. **专业性**：严格遵循学术期刊格式规范
2. **清晰性**：标签、刻度、图例清晰可读
3. **一致性**：与论文其他图表风格保持统一
4. **信息密度**：在有限空间内传达最多有效信息
5. **统计严谨**：包含检验统计量和p值

### 视觉层次：
- 主标题 → 子标题 → 坐标轴标签 → 刻度标签 → 图例
- 数据元素（直方图、散点）→ 拟合线 → 参考线 → 网格线

---

## 📝 使用说明

1. 将上述English版本的prompt复制到nanobanana中
2. 根据生成结果进行微调
3. 如需更细致的控制，可分步生成左右子图后合并
4. 确保最终图表与论文中其他Figure保持一致的视觉风格
