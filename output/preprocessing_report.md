# 数据预处理报告摘要

## 1. 数据概览
- 总记录数: 421
- 原始字段数: 53
- 处理后字段数: 97

## 2. 数据质量评估

### 2.1 缺失值处理
- 存在缺失值的列数: 34
- 主要缺失值来源:
  - judge4列的N/A值: 第4评委部分周次未参与评分
  - 0值标记: 表示选手在该周已被淘汰
- 处理策略: N/A转换为np.nan，0值保留作为淘汰标记

### 2.2 异常值检测
- 检测到含异常值的列数: 6
- 检测方法: IQR方法 (Q1-1.5*IQR, Q3+1.5*IQR)
- 处理策略: 保留异常值（属于正常业务数据范围）

### 2.3 重复值检测
- 重复行数: 0

## 3. 评分数据分析
- 评分列数量: 44
- 有效评分范围: 2.0 - 13.33
- 评分均值: 7.92
- 评分标准差: 1.5

## 4. 特征工程
- 新增特征:
  - 各周评委总分 (week*_total_score)
  - 各周评委平均分 (week*_avg_score)
  - 累积总分 (cumulative_total_score)
  - 整体平均分 (overall_avg_score)
  - 评分趋势 (score_trend)
  - 赛季规则标记 (season_rule)
  - 参与周数 (active_weeks)
  - 是否冠军 (is_winner)
  - 类别编码 (industry_encoded, country_encoded)

## 5. 数据划分说明
- 按赛季规则划分:
  - 排名法 (Season 1-2): 16 条
  - 百分比法 (Season 3-27): 306 条
  - 排名法+评委决定 (Season 28-34): 99 条

## 6. 输出文件清单
- processed_main_data.csv: 处理后的核心数据
- processed_supplementary_data.csv: 处理后的补充数据
- question1_data.csv: 问题1专用数据集
- question2_data.csv: 问题2专用数据集
- question3_data.csv: 问题3专用数据集
- question4_data.csv: 问题4专用数据集
- data_dictionary.csv: 数据字典
- 01-08_*.png: 可视化分析图表

---
生成时间: 2026年MCM竞赛
版本: v1.0