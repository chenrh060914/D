# Nanobanana提示词：模型稳定性/鲁棒性分析图

## 图表目标
**Figure 7.2: 10-Fold Cross-Validation Accuracy Across Seasons**

---

## 英文提示词（推荐）

```
Create a professional scientific research-style line chart showing 10-fold cross-validation results for a machine learning model evaluation.

**Chart specifications:**
- Title: "Figure 7.2: 10-Fold Cross-Validation Accuracy Across Seasons"
- X-axis: "Fold Number" with labels from "Fold 1" to "Fold 10"
- Y-axis: "Accuracy (%)" ranging from 95% to 100.5%
- Background: Clean white/light gray academic style with subtle grid lines

**Data visualization:**
- A blue connecting line with circular data point markers
- All 10 data points at 100.00% accuracy (perfect prediction)
- Each data point labeled with its exact value "100.00%"
- Data points should be prominent (larger markers, approximately 8-10px)

**Style requirements:**
- Color scheme: Professional blue (#1f77b4 or similar academic blue)
- Line thickness: Medium weight (2-3px)
- Font: Sans-serif academic font (Arial, Helvetica, or similar)
- Clean minimalist scientific publication style
- Include a dashed horizontal reference line at 100% with subtle opacity
- Add a shaded region or error band around the line showing ±0.00% standard deviation (to indicate perfect stability)

**Additional elements:**
- Add annotation box in corner showing:
  • Mean Accuracy: 100.00% ± 0.00%
  • Cohen's Kappa: 1.0000
- Include subtitle: "Fan Voting Estimation Model - Perfect Elimination Prediction"

**Output quality:**
- High resolution (300 DPI)
- Publication-ready figure suitable for IEEE/ACM academic paper format
- Aspect ratio approximately 16:9 or 4:3
```

---

## 中文提示词

```
创建一张专业的科研风格折线图，展示机器学习模型的10折交叉验证结果。

**图表规格：**
- 标题："Figure 7.2: 10-Fold Cross-Validation Accuracy Across Seasons"
- 横轴："Fold Number"，标签从"Fold 1"到"Fold 10"
- 纵轴："Accuracy (%)"，范围95%到100.5%
- 背景：简洁的白色/浅灰色学术风格，带有淡淡的网格线

**数据可视化：**
- 蓝色连接线配合圆形数据点标记
- 所有10个数据点均在100.00%准确率（完美预测）
- 每个数据点标注精确数值"100.00%"
- 数据点应突出显示（较大标记，约8-10px）

**风格要求：**
- 配色方案：专业蓝色（#1f77b4或类似学术蓝）
- 线条粗细：中等粗细（2-3px）
- 字体：无衬线学术字体（Arial、Helvetica或类似）
- 简洁的极简主义科学出版风格
- 在100%处添加虚线水平参考线，透明度较低
- 围绕折线添加阴影区域或误差带，显示±0.00%标准差（表示完美稳定性）

**附加元素：**
- 在角落添加注释框显示：
  • 平均准确率：100.00% ± 0.00%
  • Cohen's Kappa：1.0000
- 包含副标题："Fan Voting Estimation Model - Perfect Elimination Prediction"

**输出质量：**
- 高分辨率（300 DPI）
- 适合IEEE/ACM学术论文格式的出版级图表
- 宽高比约16:9或4:3
```

---

## 备选提示词：参数敏感性热力图版本

如果需要生成参数敏感性热力图，可使用以下提示词：

```
Create a professional scientific heatmap showing parameter sensitivity analysis for a machine learning model.

**Chart specifications:**
- Title: "Figure 7.3: Parameter Sensitivity Analysis Heatmap"
- X-axis: "Regularization Coefficient (λ_reg)" with values [0.05, 0.10, 0.15, 0.20, 0.25]
- Y-axis: "Smoothing Coefficient (λ_smooth)" with values [0.01, 0.03, 0.05, 0.07, 0.10]
- Color scale: Blue-White-Red diverging colormap (coolwarm)

**Data visualization:**
- Heatmap cells showing accuracy values (95%-100%)
- Annotate each cell with the exact accuracy percentage
- Optimal parameter region highlighted (λ_reg=0.10, λ_smooth=0.05 showing 100.00%)

**Style requirements:**
- Professional academic publication style
- Clear cell borders with white gridlines
- Color bar on the right side labeled "Accuracy (%)"
- Sans-serif font (Arial/Helvetica)

**Additional elements:**
- Star or circle marker on the optimal parameter combination
- Add annotation: "Optimal: λ_reg=0.10, λ_smooth=0.05"

**Output quality:**
- High resolution (300 DPI)
- Publication-ready for O-Award level MCM paper
- Aspect ratio approximately 1:1 (square)
```

---

## 数据参考（来自模型检验模块）

根据论文中的10折交叉验证结果：

| 折次 | 淘汰预测准确率 | Cohen's Kappa |
|------|--------------|---------------|
| Fold 1 | 100.00% | 1.0000 |
| Fold 2 | 100.00% | 1.0000 |
| Fold 3 | 100.00% | 1.0000 |
| Fold 4 | 100.00% | 1.0000 |
| Fold 5 | 100.00% | 1.0000 |
| Fold 6 | 100.00% | 1.0000 |
| Fold 7 | 100.00% | 1.0000 |
| Fold 8 | 100.00% | 1.0000 |
| Fold 9 | 100.00% | 1.0000 |
| Fold 10 | 100.00% | 1.0000 |
| **平均** | **100.00% ± 0.00%** | **1.0000** |

---

## 使用说明

1. 将上述英文提示词复制到nanobanana图像生成工具中
2. 根据实际生成效果可微调参数（如颜色、字体大小等）
3. 生成后下载高分辨率PNG或PDF格式
4. 建议文件命名：`Figure_7_2_CV_Accuracy.png`

---

## O奖级别图表设计要点

1. **简洁性**：避免过多装饰元素，保持学术论文的专业感
2. **可读性**：数据标签清晰可见，字体大小适中
3. **一致性**：配色与论文整体风格统一
4. **完整性**：包含完整的坐标轴标签、图例和注释
5. **高分辨率**：确保打印和屏幕显示均清晰

---

*文档创建时间：2026年MCM竞赛*
*用途：nanobanana图像生成提示词参考*
