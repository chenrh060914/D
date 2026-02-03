# Nanobanana 提示词：模型预测准确性对比图

## 图表标题
**Figure 7.1: Comparison of Elimination Prediction Accuracy Across Models**

---

## 📋 Nanobanana 提示词（英文版，推荐）

```
Create a professional scientific research bar chart for an academic paper with the following specifications:

CHART TYPE: Vertical bar chart comparing model prediction accuracy

TITLE: "Figure 7.1: Comparison of Elimination Prediction Accuracy Across Models"

DATA:
- Bar 1: "Our Dual-Scheme Model" with accuracy value 100.00%
- Bar 2: "Random Guess Baseline" with accuracy value 11.74%
- Bar 3: "Naive Baseline Model" with accuracy value 50.00%

AXES:
- X-axis: Model Type (categorical labels for each model)
- Y-axis: Prediction Accuracy (%) ranging from 0% to 100% with grid lines at 20% intervals

VISUAL STYLE:
- Clean, minimalist academic publication style
- Color scheme: Professional gradient blues (#1E88E5 for main model, #64B5F6 for baselines) or a distinct highlight color (green #4CAF50) for the best-performing model
- White background with subtle light gray grid lines
- High contrast for readability

ANNOTATIONS:
- Display exact percentage values at the top of each bar (100.00%, 11.74%, 50.00%)
- Add horizontal dashed reference line at 100% level
- Include relative improvement annotation: "751.9% improvement over random baseline"

TYPOGRAPHY:
- Title: Bold, 14pt sans-serif font (Arial or Helvetica)
- Axis labels: 12pt sans-serif
- Value annotations: 10pt bold
- Clean, professional LaTeX-style rendering

ADDITIONAL ELEMENTS:
- Error bars if applicable (optional)
- Light shadow effect on bars for depth
- Subtle gradient fill on bars from bottom to top

RESOLUTION: High resolution suitable for academic publication (300+ DPI)
ASPECT RATIO: 16:10 or 4:3 horizontal orientation
OUTPUT FORMAT: PNG with transparent or white background
```

---

## 📋 Nanobanana 提示词（中文版）

```
创建一张专业科研风格的柱状图，用于学术论文发表，具体规格如下：

图表类型：垂直柱状图，对比模型预测准确率

标题："Figure 7.1: Comparison of Elimination Prediction Accuracy Across Models"

数据：
- 柱1："Our Dual-Scheme Model"（双方案融合模型），准确率 100.00%
- 柱2："Random Guess Baseline"（随机猜测基准），准确率 11.74%
- 柱3："Naive Baseline Model"（朴素基准模型），准确率 50.00%

坐标轴：
- X轴：模型类型（各模型的分类标签）
- Y轴：预测准确率（%），范围从0%到100%，网格线间隔20%

视觉风格：
- 简洁、极简的学术出版风格
- 配色方案：专业蓝色渐变（主模型#1E88E5，基准模型#64B5F6），或为最佳模型使用突出颜色（绿色#4CAF50）
- 白色背景配淡灰色网格线
- 高对比度以保证可读性

标注：
- 在每个柱顶部显示精确百分比数值（100.00%、11.74%、50.00%）
- 在100%水平线处添加水平虚线参考线
- 添加相对提升标注："相比随机基准提升751.9%"

字体排版：
- 标题：粗体，14pt无衬线字体（Arial或Helvetica）
- 坐标轴标签：12pt无衬线
- 数值标注：10pt粗体
- 干净专业的LaTeX风格渲染

分辨率：适合学术出版的高分辨率（300+ DPI）
宽高比：16:10或4:3横向
输出格式：PNG，透明或白色背景
```

---

## 📋 简洁版提示词（快速生成）

```
Scientific bar chart for academic paper: "Figure 7.1: Comparison of Elimination Prediction Accuracy Across Models". 

Three vertical bars showing:
1. "Our Model" = 100% (highlighted in green #4CAF50)
2. "Random Baseline" = 11.74% (blue #64B5F6)  
3. "Naive Baseline" = 50% (blue #42A5F5)

Y-axis: "Prediction Accuracy (%)" from 0-100%. 
Display exact values on bar tops.
Clean white background, gray gridlines, professional academic style.
High resolution 300 DPI, 16:10 aspect ratio.
```

---

## 📊 数据来源说明

根据模型求解模块和模型检验模块的结果：

| 模型 | 准确率 | 说明 |
|------|--------|------|
| Our Dual-Scheme Model | **100.00%** | 约束优化 + 贝叶斯推断双方案融合 |
| Random Guess Baseline | **11.74%** | 随机猜测的期望准确率 (1/n_contestants) |
| Naive Baseline Model | **50.00%** | 简单50%概率猜测（可选基准） |

**关键指标**：
- 有效预测周数：264周
- Cohen's Kappa系数：1.0000（完全一致）
- 相对提升：751.9%（模型 vs 随机基准）

---

## 🎨 配色建议

### 方案一：蓝绿对比（推荐）
- 主模型（最优）：`#4CAF50` (Material Green 500)
- 基准模型1：`#2196F3` (Material Blue 500)
- 基准模型2：`#64B5F6` (Material Blue 300)

### 方案二：单色渐变
- 主模型（最优）：`#1565C0` (深蓝)
- 基准模型1：`#42A5F5` (中蓝)
- 基准模型2：`#90CAF9` (浅蓝)

### 方案三：学术期刊风格
- 主模型（最优）：`#D32F2F` (红色突出)
- 基准模型：`#757575` (灰色)

---

## 📐 图表布局参考

```
    100% ┼──────────────────────────────────────
         │   ████████                            
     80% ┼   ████████  ← 100.00%                
         │   ████████                            
     60% ┼   ████████                   ████████
         │   ████████                   ████████ ← 50.00%
     40% ┼   ████████                   ████████
         │   ████████                   ████████
     20% ┼   ████████  ████████         ████████
         │   ████████  ████████ ← 11.74%████████
      0% ┼───████████──████████─────────████████───
             Our       Random          Naive
            Model      Guess          Baseline
```

---

## 🔧 可选增强元素

1. **误差条（Error Bars）**：如果有交叉验证结果，可添加置信区间
2. **统计显著性标注**：添加 `***` 表示 p<0.001 的显著性水平
3. **子图标题**：`(a) Elimination Prediction Accuracy`
4. **图例**：说明颜色含义

---

**生成时间**：2026年MCM竞赛
**适用对象**：问题1模型检验章节可视化
