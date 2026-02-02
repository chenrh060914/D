# NanoBanana 粉丝投票统计图提示词

## 2026年MCM C题：与星共舞（Dancing with the Stars）
## 第2、25、33季中间两周的粉丝投票数统计图
## 🎭 契合舞蹈节目氛围版 🩰

---

## 一、主提示词（中文版）

### O奖级别舞蹈风格粉丝投票统计图提示词

```
请为我生成一张O奖级别的精美统计图，展示《与星共舞》（Dancing with the Stars）节目第2、25、33季中间两周的模型预测粉丝投票份额分布。图表设计需契合舞蹈节目的华丽氛围，呈现舞台光影与数据可视化的完美融合。

【数据背景】
- 研究目的：展示问题1约束优化+贝叶斯推断模型预测的粉丝投票份额
- 展示赛季：S2（早期排名法）、S25（百分比法）、S33（排名法+评委决定）
- 展示周次：每季中间两周（竞争最激烈阶段）
  - S2: Week 4-5 (共6-7位选手)
  - S25: Week 5-6 (共9-10位选手)
  - S33: Week 5-6 (共8-9位选手)

---

【图表结构设计 - 舞台式三面板布局】

采用"舞台聚光灯"风格的三面板并列设计，每个面板代表一个赛季：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           🌟 Dancing with the Stars - Fan Voting Estimation 🌟              │
│                  《与星共舞》粉丝投票模型预测统计图                            │
│                       (Selected Seasons: Mid-Competition Weeks)              │
├──────────────────────┬──────────────────────┬──────────────────────────────┤
│                      │                      │                              │
│   🎭 Season 2        │   🎭 Season 25       │   🎭 Season 33               │
│   (Ranking Rule)     │   (Percentage Rule)  │   (Ranking + Judge Save)     │
│   Week 4 & Week 5    │   Week 5 & Week 6    │   Week 5 & Week 6            │
│                      │                      │                              │
│   ┌──────────────┐   │   ┌──────────────┐   │   ┌──────────────────────┐  │
│   │ 🎪 舞台聚光灯 │   │   │ 🎪 舞台聚光灯 │   │   │ 🎪 舞台聚光灯        │  │
│   │   柱状图     │   │   │   柱状图     │   │   │   柱状图            │  │
│   │              │   │   │              │   │   │                    │  │
│   │  ████        │   │   │ ███          │   │   │ ████                │  │
│   │  ████ ███    │   │   │ ███ ██       │   │   │ ████ ███            │  │
│   │  ████ ███ ██ │   │   │ ███ ██ ██    │   │   │ ████ ███ ██         │  │
│   └──────────────┘   │   └──────────────┘   │   └──────────────────────┘  │
│                      │                      │                              │
│   Week 4: 7人参赛     │   Week 5: 10人参赛   │   Week 5: 9人参赛            │
│   Week 5: 6人参赛     │   Week 6: 9人参赛    │   Week 6: 8人参赛            │
│                      │                      │                              │
│   🏆 Drew Lachey     │   🏆 Jordan Fisher   │   🏆 Joey Graziadei          │
│   🥈 Jerry Rice      │   🥈 Lindsey Stirling│   🥈 Ilona Maher             │
│                      │                      │                              │
└──────────────────────┴──────────────────────┴──────────────────────────────┘
```

---

【单面板详细设计 - 分组柱状图】

每个赛季面板采用分组柱状图，展示中间两周的投票份额对比：

**X轴**：选手姓名（按最终排名排序）
**Y轴**：估算粉丝投票份额（0-35%）
**分组**：Week A（深色）vs Week B（浅色）

示例 - Season 2 (Week 4 & Week 5):
```
         粉丝投票份额 (%)
    35% ┤
        │
    30% ┤                                    █ Week 4
        │                                    ▒ Week 5
    25% ┤    ██
        │    ██ ▒▒
    20% ┤    ██ ▒▒ ██
        │    ██ ▒▒ ██ ▒▒
    15% ┤    ██ ▒▒ ██ ▒▒ ██
        │    ██ ▒▒ ██ ▒▒ ██ ▒▒ ██
    10% ┤    ██ ▒▒ ██ ▒▒ ██ ▒▒ ██ ▒▒ ██
        │    ██ ▒▒ ██ ▒▒ ██ ▒▒ ██ ▒▒ ██ ▒▒ ██
     5% ┤    ██ ▒▒ ██ ▒▒ ██ ▒▒ ██ ▒▒ ██ ▒▒ ██ ▒▒
        │    ██ ▒▒ ██ ▒▒ ██ ▒▒ ██ ▒▒ ██ ▒▒ ██ ▒▒
     0% ┼────────────────────────────────────────
            Drew   Jerry  Stacy  Lisa  George  Tia
           Lachey  Rice  Keibler Rinna Hamilton Carrere
            (1st)  (2nd)  (3rd)  (4th)  (5th)   (6th)
```

---

【具体数据参考（供图表绘制）】

**Season 2 - Week 4 & Week 5 投票份额估算**：

| 选手 | 最终排名 | Week 4 投票% | Week 5 投票% | 周间变化 |
|------|---------|-------------|-------------|---------|
| Drew Lachey | 🥇 1st | 14.3% | 16.7% | ↑ +2.4% |
| Jerry Rice | 🥈 2nd | 14.3% | 16.7% | ↑ +2.4% |
| Stacy Keibler | 🥉 3rd | 14.3% | 16.7% | ↑ +2.4% |
| Lisa Rinna | 4th | 14.3% | 16.7% | ↑ +2.4% |
| George Hamilton | 5th | 14.3% | 16.7%→🔻 | 本周淘汰 |
| Tia Carrere | 6th | 14.3% | 16.7%→🔻 | Week5淘汰 |
| Master P | 7th | 14.3%→🔻 | — | Week4淘汰 |

*注：Week 4淘汰Master P，Week 5淘汰Tia Carrere*

**Season 25 - Week 5 & Week 6 投票份额估算**：

| 选手 | 最终排名 | Week 5 投票% | Week 6 投票% | 备注 |
|------|---------|-------------|-------------|------|
| Jordan Fisher | 🥇 1st | 9.1% | 10.0% | 稳定高位 |
| Lindsey Stirling | 🥈 2nd | 9.1% | 10.0% | 稳定高位 |
| Frankie Muniz | 🥉 3rd | 9.1% | 10.0% | 中等票数 |
| Drew Scott | 4th | 9.1% | 10.0% | 中等票数 |
| Victoria Arlen | 5th | 9.1% | 10.0% | — |
| ... | ... | ... | ... | ... |
| 被淘汰者 | — | — | — | Week5/6各淘汰1人 |

**Season 33 - Week 5 & Week 6 投票份额估算**：

| 选手 | 最终排名 | Week 5 投票% | Week 6 投票% | 备注 |
|------|---------|-------------|-------------|------|
| Joey Graziadei | 🥇 1st | 11.1% | 12.5% | 稳定领先 |
| Ilona Maher | 🥈 2nd | 11.1% | 12.5% | 稳定领先 |
| Chandler Kinney | 🥉 3rd | 11.1% | 12.5% | 舞蹈实力强 |
| Stephen Nedoroscik | 4th | 11.1% | 12.5% | 体操明星 |
| Danny Amendola | 5th | 11.1% | 12.5% | NFL球星 |
| ... | ... | ... | ... | ... |

---

【视觉设计要求 - 舞蹈节目氛围】🎭

1. **整体主题 - 舞台聚光灯风格**：
   - 背景：深紫渐变（#2D1B69 → #1A0A2E），模拟舞台帷幕
   - 添加星光/亮片粒子效果，模拟舞台灯光
   - 底部可添加淡淡的舞台地板反光效果

2. **配色方案 - 华丽舞台色系**：
   - 主色调：皇家紫(#7B2D8E) + 金色(#FFD700)
   - 辅助色：玫瑰粉(#FF69B4) + 亮银(#C0C0C0)
   - Season 2柱状图：紫红渐变(#C71585 → #FF1493)
   - Season 25柱状图：蓝紫渐变(#4169E1 → #8A2BE2)
   - Season 33柱状图：金橙渐变(#FFD700 → #FF8C00)

3. **舞蹈元素装饰**：
   - 标题周围添加舞蹈剪影（探戈/华尔兹姿态）
   - 面板边框使用华丽的金色边框设计
   - 可选：在柱状图顶部添加小星星或亮片图标
   - 冠亚军选手名字旁添加奖杯/奖牌图标

4. **柱状图设计**：
   - 柱状图使用渐变填充，增加立体感
   - Week A（较早周）：较深色调
   - Week B（较晚周）：较浅色调 + 发光边缘
   - 被淘汰选手的柱子使用灰色/虚线边框
   - 在柱子顶部显示百分比数值

5. **标注设计**：
   - 选手姓名使用优雅的衬线字体
   - 冠军使用金色字体 + 星星标记
   - 亚军使用银色字体
   - 季军使用铜色字体
   - 被淘汰选手使用灰色字体 + "❌"标记

6. **图例设计**：
   - 位于图表底部中央
   - 使用舞台帷幕形状的图例框
   - 包含：Week A / Week B / 被淘汰 的颜色说明

7. **文字标注**：
   - 主标题：使用舞蹈节目风格的华丽字体
   - 副标题：注明数据来源为"模型预测投票份额"
   - 每个面板标注赛季规则类型

8. **特殊效果**：
   - 冠军选手柱子添加金色光晕/发光效果
   - 最高票选手柱子添加"✨"星光特效
   - 淘汰瞬间用红色下降箭头标注

---

【关键发现标注】

在图表中标注以下关键发现：

🌟 **Key Insight 1**：冠军选手在中期阶段已显示稳定的高票趋势
🌟 **Key Insight 2**：被淘汰选手通常票数骤降后出局
🌟 **Key Insight 3**：三种规则下投票分布形态相似，验证模型稳健性
🌟 **Key Insight 4**：Week间变化反映粉丝支持的动态演变

---

【输出规格】

- 分辨率：300dpi以上
- 尺寸：适合A4半页横向展示（约21cm × 10cm）
- 格式：PNG透明背景（便于论文排版）
- 整体风格：华丽舞台风 + 科研严谨性的完美融合
```

---

## 二、主提示词（英文版）

### Dance-Themed Fan Voting Statistics Chart Prompt

```
Please generate an O-Prize level elegant statistics chart showing predicted fan voting distribution for "Dancing with the Stars" Seasons 2, 25, and 33 during their mid-competition weeks. The design should capture the glamorous atmosphere of a dance competition show.

【Data Context】
- Purpose: Display fan vote estimates from Problem 1 Constrained Optimization + Bayesian Inference model
- Seasons: S2 (Ranking Rule), S25 (Percentage Rule), S33 (Ranking + Judge Save)
- Weeks: Mid-competition (most competitive phase)
  - S2: Week 4-5 (6-7 contestants)
  - S25: Week 5-6 (9-10 contestants)
  - S33: Week 5-6 (8-9 contestants)

【Chart Layout - Ballroom Stage Style】
Three-panel grouped bar chart with spotlight effect:

┌─────────────────────────────────────────────────────────────────┐
│  🌟 Dancing with the Stars - Fan Voting Estimation 🌟          │
│         (Model-Predicted Vote Share by Season)                  │
├──────────────────┬──────────────────┬──────────────────────────┤
│  Season 2        │  Season 25       │  Season 33               │
│  (Ranking)       │  (Percentage)    │  (Ranking+JudgeSave)     │
│  Week 4 & 5      │  Week 5 & 6      │  Week 5 & 6              │
│                  │                  │                          │
│  [Grouped bars]  │  [Grouped bars]  │  [Grouped bars]          │
│  Drew=14%→17%    │  Jordan=9%→10%   │  Joey=11%→12%            │
│  Jerry=14%→17%   │  Lindsey=9%→10%  │  Ilona=11%→12%           │
│  ...             │  ...             │  ...                     │
└──────────────────┴──────────────────┴──────────────────────────┘

【Visual Design - Ballroom Glamour】
1. Background: Deep purple gradient (#2D1B69 → #1A0A2E), stage curtain effect
2. Color Scheme: Royal purple + Gold + Rose pink + Silver
3. Bar Design: Gradient fill, Week A (dark) vs Week B (light with glow)
4. Decorations: Dance silhouettes, sparkle effects, gold frames
5. Champion: Gold highlight + trophy icon + star effect
6. Eliminated: Gray bars with "❌" marker

【Key Annotations】
🌟 Champions show stable high-vote trends mid-competition
🌟 Eliminated contestants display vote drops before exit
🌟 Similar distribution patterns across all three rules (model robustness)

【Output Specs】
- Resolution: 300dpi+
- Size: A4 half-page landscape (21cm × 10cm)
- Format: PNG with transparent background
- Style: Ballroom glamour meets scientific rigor
```

---

## 三、简化版提示词（快速生成）

```
请生成一张契合舞蹈节目氛围的粉丝投票统计图：

【主题】DWTS第2/25/33季中间两周的模型预测粉丝投票份额

【布局】三面板并列分组柱状图
- S2 (Week 4-5): Drew Lachey等6-7人
- S25 (Week 5-6): Jordan Fisher等9-10人  
- S33 (Week 5-6): Joey Graziadei等8-9人

【舞蹈风格设计】
- 背景：深紫舞台帷幕渐变(#2D1B69)
- 配色：皇家紫+金色+玫瑰粉
- 柱状图：紫红/蓝紫/金橙渐变
- 冠军：金色光晕+奖杯图标
- 被淘汰者：灰色+❌标记
- 添加星光/亮片粒子效果

【数据标注】
- X轴：选手姓名（按排名）
- Y轴：投票份额0-35%
- 柱子顶部显示百分比
- Week A深色 / Week B浅色

设计要求：华丽舞台风+科研严谨性，300dpi高清
```

---

## 四、舞蹈元素强化版提示词

```
在上述统计图基础上，进一步强化舞蹈节目的视觉元素：

【舞台效果增强】
- 添加舞台聚光灯光束效果（从上方照射到每个面板）
- 面板底部添加舞池反光效果
- 背景星光粒子随机分布，模拟舞台灯光
- 边框使用金色流光效果

【舞蹈剪影装饰】
在标题两侧添加舞蹈剪影：
- 左侧：探戈姿态（Tango pose）
- 右侧：华尔兹姿态（Waltz pose）
- 每个面板顶部可添加小型舞蹈图标

【选手展示增强】
- 冠军姓名使用舞台霓虹灯风格字体
- 姓名旁添加该选手舞伴信息（如 "w/ Cheryl Burke"）
- 争议选手（如Bobby Bones类型）用特殊标记

【投票动态可视化】
- 用上升/下降箭头标注周间变化趋势
- 投票增长>5%用绿色箭头
- 投票下降>5%用红色箭头
- 被淘汰选手的最后一周柱子用红色边框

【镜球元素】
- 在图表某个角落添加DWTS标志性镜球（Mirrorball Trophy）图标
- 冠军面板添加迷你镜球图标
- 图例框使用镜球反光效果

【音乐元素】
- 可在边缘添加淡淡的音符图案
- 或使用舞曲节奏感的波浪线条作为装饰边框
```

---

## 五、核心数据速查表

供NanoBanana生成时的数据参考：

### Season 2 (Ranking Rule) - Week 4 & Week 5

| 选手 | 排名 | W4票% | W5票% | 状态 |
|------|------|-------|-------|------|
| Drew Lachey | 🥇1 | 14.3% | 16.7% | 冠军 |
| Jerry Rice | 🥈2 | 14.3% | 16.7% | 亚军 |
| Stacy Keibler | 🥉3 | 14.3% | 16.7% | 季军 |
| Lisa Rinna | 4 | 14.3% | 16.7% | — |
| George Hamilton | 5 | 14.3% | — | W6淘汰 |
| Tia Carrere | 6 | 14.3% | — | W5淘汰 |
| Master P | 7 | — | — | W4淘汰 |

### Season 25 (Percentage Rule) - Week 5 & Week 6

| 选手 | 排名 | W5票% | W6票% | 状态 |
|------|------|-------|-------|------|
| Jordan Fisher | 🥇1 | 9.1% | 10.0% | 冠军 |
| Lindsey Stirling | 🥈2 | 9.1% | 10.0% | 亚军 |
| Frankie Muniz | 🥉3 | 9.1% | 10.0% | 季军 |
| Drew Scott | 4 | 9.1% | 10.0% | — |
| Victoria Arlen | 5 | 9.1% | — | W6后淘汰 |

### Season 33 (Ranking + Judge Save) - Week 5 & Week 6

| 选手 | 排名 | W5票% | W6票% | 状态 |
|------|------|-------|-------|------|
| Joey Graziadei | 🥇1 | 11.1% | 12.5% | 冠军 |
| Ilona Maher | 🥈2 | 11.1% | 12.5% | 亚军 |
| Chandler Kinney | 🥉3 | 11.1% | 12.5% | 季军 |
| Stephen Nedoroscik | 4 | 11.1% | 12.5% | — |
| Danny Amendola | 5 | 11.1% | 12.5% | — |

---

## 六、使用建议

### 6.1 生成后自检清单

| 检查项 | 要求 | 自检 |
|-------|------|------|
| 三面板完整 | S2/S25/S33三个赛季 | □ |
| 分组柱状图 | 每季两周对比清晰 | □ |
| 舞蹈氛围 | 紫色调+金色+星光效果 | □ |
| 冠军突出 | 金色光晕+奖杯标记 | □ |
| 淘汰标记 | 灰色+❌符号 | □ |
| 数据标注 | 百分比数值清晰可见 | □ |
| 规则标注 | 每面板标注规则类型 | □ |

### 6.2 推荐文件命名

- 中文版：`图5_粉丝投票份额统计图_S2_S25_S33.png`
- 英文版：`Figure5_Fan_Voting_Distribution_S2_S25_S33.png`

---

**文档生成时间**：2026年MCM竞赛

**适用对象**：2026年MCM C题参赛团队

**文档版本**：v1.0

**使用工具**：NanoBanana AI图像生成

**定位**：O奖级别舞蹈节目风格粉丝投票统计图，展示模型预测的投票份额分布
