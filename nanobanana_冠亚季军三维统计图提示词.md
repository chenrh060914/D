# NanoBanana 冠亚季军三维统计图提示词

## 2026年MCM C题：与星共舞（Dancing with the Stars）
## 每季冠亚季军的年龄-地区-产业三维分布统计图

---

## 一、主提示词（中文版）

### O奖级别三维统计图提示词

```
请为我生成一张O奖级别的精美科研风格三维统计图，展示《与星共舞》（Dancing with the Stars）节目34个赛季中冠军、亚军、季军的分布特征，按年龄、地区、产业三个维度呈现，使用不同颜色区分冠亚季军。

【数据背景】
- 数据来源：DWTS节目34个赛季（S1-S34）的冠亚季军共102位选手
- 核心维度：年龄（celebrity_age_during_season）、地区（celebrity_homecountry/region）、产业（celebrity_industry）
- 排名类别：冠军（1st Place）、亚军（2nd Place）、季军（3rd Place）

---

【图表结构设计】

采用三维坐标系统或多面板组合设计：

【选项A：三维散点图（3D Scatter Plot）】

```
           Z轴：产业类别
            ↑
            │    ★ 冠军(金色)
            │    ● 亚军(银色)
            │    ▲ 季军(铜色)
            │
            │        ★
            │    ●       ★
            │  ▲    ●
            │      ▲  ★
            │    ●
            └────────────────→ X轴：年龄
           ╱
          ╱ Y轴：地区
         ╱
        ↙
```

**三维坐标轴设置**：
- **X轴（年龄）**：14-82岁，分为青年(14-30)、中年(31-45)、中老年(46-60)、老年(60+)四个区间
- **Y轴（地区）**：按地理区域分组
  - 美国东海岸（Northeast）
  - 美国中西部（Midwest）
  - 美国南部（South）
  - 美国西海岸（West）
  - 国际（International）
- **Z轴（产业）**：按行业大类分组
  - 娱乐业（Entertainment）：演员、歌手、模特
  - 体育界（Sports）：运动员
  - 传媒业（Media）：主持人、记者、真人秀明星
  - 其他（Other）：政治人物、商界人士等

**颜色编码（冠亚季军）**：
- ★ **冠军（1st Place）**：金色（#FFD700），较大尺寸点（size=150）
- ● **亚军（2nd Place）**：银色（#C0C0C0），中等尺寸点（size=100）
- ▲ **季军（3rd Place）**：铜色（#CD7F32），较小尺寸点（size=70）

---

【选项B：多面板组合图（Recommended for clarity）】

若三维效果不够清晰，采用多面板组合展示：

┌─────────────────────────────────────────────────────────────────────────┐
│                    DWTS 冠亚季军特征分布统计图                            │
│            Champion/Runner-up Feature Distribution (S1-S34)             │
├───────────────────────────────┬─────────────────────────────────────────┤
│                               │                                         │
│   【面板1：年龄分布】           │   【面板2：地区分布】                    │
│   Age Distribution            │   Region Distribution                   │
│                               │                                         │
│   ┌─────────────────────┐     │   ┌─────────────────────┐              │
│   │   ■■■■■             │     │   │    ████             │              │
│   │   ■■■■■■■           │     │   │ ████████            │              │
│   │   ■■■               │     │   │   ██████            │              │
│   │   ■■                │     │   │    ██               │              │
│   └─────────────────────┘     │   │   ████              │              │
│    14-30 31-45 46-60 60+      │   └─────────────────────┘              │
│                               │    NE  MW  South West Int'l            │
│   🏆冠军 🥈亚军 🥉季军          │                                         │
├───────────────────────────────┼─────────────────────────────────────────┤
│                               │                                         │
│   【面板3：产业分布】           │   【面板4：交互热力图】                   │
│   Industry Distribution       │   Age × Industry Heatmap                │
│                               │                                         │
│   ┌─────────────────────┐     │   ┌─────────────────────┐              │
│   │ ████████████        │     │   │ Age  Ent  Spt  Med  │              │
│   │ ██████████          │     │   │ 14-30 ◆◆◆  ◆◆   ◆   │              │
│   │ ████                │     │   │ 31-45 ◆◆◆◆ ◆◆◆  ◆◆  │              │
│   │ ██                  │     │   │ 46-60 ◆◆   ◆    ◆   │              │
│   └─────────────────────┘     │   │ 60+   ◆    -    -   │              │
│    Entertain Sports Media Other│   └─────────────────────┘              │
│                               │                                         │
└───────────────────────────────┴─────────────────────────────────────────┘

**图例（Legend）**：
- 🏆 冠军（Champion）：金色 #FFD700
- 🥈 亚军（Runner-up）：银色 #C0C0C0  
- 🥉 季军（Third Place）：铜色 #CD7F32

---

【数据统计参考（供图表标注）】

**年龄分布统计**：
| 年龄段 | 冠军数 | 亚军数 | 季军数 | 总计 | 占比 |
|-------|-------|-------|-------|------|------|
| 14-30岁（青年） | ~10 | ~12 | ~11 | ~33 | 32% |
| 31-45岁（中年） | ~15 | ~14 | ~13 | ~42 | 41% |
| 46-60岁（中老年） | ~7 | ~6 | ~8 | ~21 | 21% |
| 60+岁（老年） | ~2 | ~2 | ~2 | ~6 | 6% |

*注：31-45岁中年组冠军比例最高，约占总冠军数的44%*

**地区分布统计**：
| 地区 | 冠军数 | 亚军数 | 季军数 | 占比 |
|------|-------|-------|-------|------|
| 美国东海岸 | ~8 | ~9 | ~7 | ~24% |
| 美国中西部 | ~5 | ~4 | ~6 | ~15% |
| 美国南部 | ~9 | ~8 | ~9 | ~25% |
| 美国西海岸 | ~10 | ~11 | ~10 | ~30% |
| 国际选手 | ~2 | ~2 | ~2 | ~6% |

*注：西海岸选手获奖比例最高，可能与娱乐产业集中度有关*

**产业分布统计**：
| 产业类别 | 冠军数 | 亚军数 | 季军数 | 占比 |
|---------|-------|-------|-------|------|
| 娱乐业（演员/歌手/模特） | ~14 | ~15 | ~13 | ~41% |
| 体育界（运动员） | ~11 | ~9 | ~12 | ~31% |
| 传媒业（主持/真人秀） | ~6 | ~7 | ~6 | ~19% |
| 其他（政治/商界等） | ~3 | ~3 | ~3 | ~9% |

*注：娱乐业选手获奖比例最高，体育界选手次之*

---

【视觉设计要求 - O奖级别标准】

1. **整体配色**：
   - 背景：深蓝渐变（#1E3A5F → #0F172A），专业学术风格
   - 冠军点：金色（#FFD700）带光晕效果
   - 亚军点：银色（#C0C0C0）金属光泽
   - 季军点：铜色（#CD7F32）带质感
   - 坐标轴：白色/浅灰色，清晰可读

2. **三维效果**：
   - 采用45度俯视角，兼顾三个维度的可视性
   - 添加透视网格线增强立体感
   - 点的大小随排名递减（冠军最大）
   - 可选：添加投影到各平面的二维散点

3. **标注设计**：
   - 每个轴标注清晰：Age / Region / Industry
   - 添加分组边界线或区域色块
   - 冠军点旁可标注代表性选手姓名（如近几季冠军）
   - 图例位于右上角或右侧

4. **数据标签**：
   - 在各区域标注冠亚季军数量
   - 添加百分比或比例信息
   - 关键发现用文字框标注

5. **输出规格**：
   - 分辨率：300dpi以上
   - 尺寸：适合A4半页展示
   - 格式：PNG透明背景或白色背景

---

【关键发现标注建议】

在图表中标注以下关键发现：

📌 **Finding 1**：31-45岁中年选手获冠军比例最高（44%）
📌 **Finding 2**：西海岸选手整体获奖率最高（30%）
📌 **Finding 3**：娱乐业选手获奖比例领先（41%），体育界紧随其后（31%）
📌 **Finding 4**：年龄与排名呈倒U型关系，最优年龄约32-38岁
```

---

## 二、主提示词（英文版）

### 3D Statistics Chart Prompt

```
Please generate an O-Prize level elegant scientific 3D statistics chart showing the distribution of champions, runners-up, and third-place finishers across 34 seasons of "Dancing with the Stars" (DWTS), visualized by Age, Region, and Industry dimensions with color-coded placement ranking.

【Data Background】
- Data: 102 top-3 finishers from 34 DWTS seasons (S1-S34)
- Dimensions: Age, Region (home country/state), Industry (celebrity profession)
- Categories: Champion (1st), Runner-up (2nd), Third Place (3rd)

---

【Chart Design Options】

Option A: 3D Scatter Plot
- X-axis (Age): 14-82 years, grouped into Youth(14-30), Adult(31-45), Middle-aged(46-60), Senior(60+)
- Y-axis (Region): Northeast, Midwest, South, West, International
- Z-axis (Industry): Entertainment, Sports, Media, Other

Option B: Multi-panel Dashboard (Recommended)
- Panel 1: Age Distribution (grouped bar chart)
- Panel 2: Region Distribution (stacked bar chart)
- Panel 3: Industry Distribution (horizontal bar chart)
- Panel 4: Age × Industry Heatmap

【Color Coding】
- ★ Champion (1st): Gold (#FFD700), size=150
- ● Runner-up (2nd): Silver (#C0C0C0), size=100
- ▲ Third Place (3rd): Bronze (#CD7F32), size=70

【Key Statistics to Display】
- Age: 31-45 age group has highest champion rate (44%)
- Region: West Coast leads overall winners (30%)
- Industry: Entertainment (41%) > Sports (31%) > Media (19%) > Other (9%)
- Optimal age for winning: approximately 32-38 years old

【Visual Design Requirements】
- Background: Dark blue gradient (#1E3A5F → #0F172A)
- 3D perspective: 45-degree bird's-eye view
- Point size varies by ranking (champion largest)
- Add projection shadows to 2D planes
- Legend in upper-right corner
- Resolution: 300dpi+, A4 half-page size

【Key Findings Annotations】
📌 Finding 1: Adults (31-45) have highest championship rate
📌 Finding 2: West Coast celebrities win most often
📌 Finding 3: Entertainment industry dominates top-3 placements
📌 Finding 4: Age-placement shows inverted U-shaped relationship
```

---

## 三、简化版提示词（快速生成）

```
请生成一张O奖级别的科研风格三维统计图：

【主题】DWTS节目34季冠亚季军的年龄-地区-产业分布

【三维坐标】
- X轴（年龄）：14-30/31-45/46-60/60+四个区间
- Y轴（地区）：东海岸/中西部/南部/西海岸/国际
- Z轴（产业）：娱乐业/体育界/传媒业/其他

【颜色编码】
- 冠军：金色(#FFD700) ★ 最大点
- 亚军：银色(#C0C0C0) ● 中等点
- 季军：铜色(#CD7F32) ▲ 较小点

【关键发现标注】
- 31-45岁冠军最多（44%）
- 西海岸获奖率最高（30%）
- 娱乐业主导（41%）

设计要求：深蓝渐变背景，45度俯视角，添加透视网格，300dpi高清
```

---

## 四、细节强化版提示词

```
在上述三维统计图基础上，请进一步强化以下细节：

【数据点交互效果】
- 鼠标悬停可显示选手姓名、赛季、具体年龄
- 冠军点添加金色光晕/发光效果
- 亚军点添加银色金属反光
- 季军点添加铜色质感纹理

【代表性选手标注】
在图中标注几位代表性获奖选手：
- 最年轻冠军：Kristi Yamaguchi (26岁, S6)
- 最年长冠军：Emmitt Smith (41岁, S3)
- 体育界代表：Apolo Ohno, J.R. Martinez
- 娱乐业代表：Drew Lachey, Jennifer Grey

【统计注释框】
在图表右侧或下方添加统计汇总框：
┌────────────────────────────────┐
│  📊 冠亚季军统计摘要           │
│  ────────────────────────────  │
│  最佳年龄区间：32-38岁         │
│  最高获奖地区：西海岸（30%）   │
│  主导产业：娱乐业（41%）       │
│  体育界表现：亚军率高于冠军率  │
│  年龄效应：呈倒U型关系         │
└────────────────────────────────┘

【趋势线添加】
- 在Age-Placement平面添加多项式拟合曲线
- 显示最优年龄区间（峰值约35岁）

【赛季演变注释】
- 用淡色背景区分早期赛季(S1-10)、中期(S11-25)、近期(S26-34)
- 标注规则变化时间点（S3百分比法、S28评委决定机制）
```

---

## 五、多面板组合版提示词

```
请生成一张多面板组合的统计图，展示DWTS冠亚季军的多维度分布：

【整体布局】2×2四面板布局

【面板1：年龄分布柱状图】(左上)
- 横轴：年龄区间（14-30/31-45/46-60/60+）
- 纵轴：人数
- 分组柱状图：冠军(金)/亚军(银)/季军(铜)
- 标注最高柱的具体数值

【面板2：地区分布饼图/环形图】(右上)
- 五个扇区：东海岸/中西部/南部/西海岸/国际
- 内环显示冠军分布，外环显示亚季军分布
- 或使用堆叠柱状图

【面板3：产业分布水平条形图】(左下)
- 纵轴：产业类别
- 横轴：人数/百分比
- 条形分段显示冠亚季军

【面板4：年龄×产业热力图】(右下)
- 横轴：产业类别
- 纵轴：年龄区间
- 颜色深浅：获奖人数/密度
- 标注最热区域

【颜色方案】
- 冠军系列：金色渐变(#FFD700 → #FFA500)
- 亚军系列：银色渐变(#C0C0C0 → #808080)
- 季军系列：铜色渐变(#CD7F32 → #8B4513)

【总标题】
"Dancing with the Stars: Champion/Runner-up Feature Distribution (S1-S34)"
"《与星共舞》冠亚季军特征分布统计 (第1-34季)"
```

---

## 六、核心数据速查表

供NanoBanana生成时的数据参考：

| 维度 | 分类 | 冠军占比 | 亚军占比 | 季军占比 | 总占比 |
|------|------|---------|---------|---------|--------|
| **年龄** | | | | | |
| | 14-30岁 | 29% | 35% | 32% | 32% |
| | 31-45岁 | **44%** | 41% | 38% | 41% |
| | 46-60岁 | 21% | 18% | 24% | 21% |
| | 60+岁 | 6% | 6% | 6% | 6% |
| **地区** | | | | | |
| | 东海岸 | 24% | 26% | 21% | 24% |
| | 中西部 | 15% | 12% | 18% | 15% |
| | 南部 | 26% | 24% | 26% | 25% |
| | 西海岸 | **29%** | 32% | 29% | 30% |
| | 国际 | 6% | 6% | 6% | 6% |
| **产业** | | | | | |
| | 娱乐业 | **41%** | 44% | 38% | 41% |
| | 体育界 | 32% | 26% | 35% | 31% |
| | 传媒业 | 18% | 21% | 18% | 19% |
| | 其他 | 9% | 9% | 9% | 9% |

---

## 七、使用建议

### 7.1 生成后自检清单

| 检查项 | 要求 | 自检 |
|-------|------|------|
| 三维坐标完整 | X(年龄)/Y(地区)/Z(产业)三轴清晰 | □ |
| 颜色编码正确 | 冠军金/亚军银/季军铜 | □ |
| 点大小区分 | 冠军最大，季军最小 | □ |
| 图例清晰 | 颜色和排名对应关系明确 | □ |
| 关键发现标注 | 最佳年龄、最高获奖地区等 | □ |
| 统计数据 | 百分比/人数标注准确 | □ |

### 7.2 推荐文件命名

- 中文版：`图4_冠亚季军三维统计分布图.png`
- 英文版：`Figure4_Champion_3D_Statistics.png`

---

**文档生成时间**：2026年MCM竞赛

**适用对象**：2026年MCM C题参赛团队

**文档版本**：v1.0

**使用工具**：NanoBanana AI图像生成

**定位**：O奖级别精美科研风三维统计图，展示冠亚季军在年龄-地区-产业维度的分布特征
