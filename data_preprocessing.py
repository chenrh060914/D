#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
2026 MCM Problem C: Dancing with the Stars - 数据预处理模块
==============================================================================

本模块实现对DWTS比赛数据的完整预处理流程，包括：
1. 数据质量评估（缺失值、异常值、重复值检测）
2. 数据清洗（缺失值处理、数据类型转换）
3. 特征工程（特征提取、编码、标准化）
4. 数据可视化分析（预处理前后对比）
5. 数据导出（CSV格式）

作者: MCM 2026 Team
日期: 2026年2月
版本: 1.0
==============================================================================
"""

# =============================================================================
# 第一部分：导入依赖库
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import warnings
import os

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']  # 设置字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.style.use('seaborn-v0_8-whitegrid')  # 使用seaborn白色网格样式
warnings.filterwarnings('ignore')

# 设置输出目录
OUTPUT_DIR = '/home/runner/work/D/D/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 配置常量
# =============================================================================
# 退赛选手的淘汰周标记
WITHDREW_WEEK = -1

# 可视化配置
VIS_SAMPLE_WEEKS = [1, 5, 10]  # 用于缺失值可视化采样的周数
VIS_SAMPLE_JUDGES = [1, 4]     # 用于缺失值可视化采样的评委
VIS_SCORE_WEEKS = [1, 3, 5, 7, 9, 10]  # 用于评分分布可视化的周数

# 异常值检测配置
OUTLIER_DETECTION_LIMIT = 10  # 异常值检测显示的列数上限

# =============================================================================
# 第二部分：数据加载
# =============================================================================

def load_data():
    """
    加载核心数据和补充数据
    
    Returns:
        df_main: 核心数据DataFrame (数据.csv)
        df_supp: 补充数据DataFrame (补充数据.xlsx)
    """
    print("=" * 70)
    print("【数据加载】")
    print("=" * 70)
    
    # 加载核心数据
    df_main = pd.read_csv('/home/runner/work/D/D/数据.csv')
    print(f"✓ 核心数据加载完成: {df_main.shape[0]} 行 × {df_main.shape[1]} 列")
    
    # 加载补充数据
    df_supp = pd.read_excel('/home/runner/work/D/D/补充数据.xlsx')
    print(f"✓ 补充数据加载完成: {df_supp.shape[0]} 行 × {df_supp.shape[1]} 列")
    
    return df_main, df_supp


# =============================================================================
# 第三部分：数据质量评估
# =============================================================================

def assess_data_quality(df, data_name="数据"):
    """
    评估数据质量，包括缺失值、异常值、重复值等
    
    Args:
        df: 待评估的DataFrame
        data_name: 数据集名称（用于输出）
    
    Returns:
        quality_report: 数据质量报告字典
    """
    print("\n" + "=" * 70)
    print(f"【数据质量评估 - {data_name}】")
    print("=" * 70)
    
    quality_report = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'missing_values': {},
        'duplicates': 0,
        'data_types': {},
        'outliers': {}
    }
    
    # 1. 检测缺失值
    print("\n--- 1. 缺失值检测 ---")
    missing_info = df.isnull().sum()
    na_string_count = df.apply(lambda x: (x == 'N/A').sum() if x.dtype == 'object' else 0)
    
    # 合并统计
    for col in df.columns:
        total_missing = missing_info[col] + na_string_count[col]
        if total_missing > 0:
            quality_report['missing_values'][col] = {
                'null_count': int(missing_info[col]),
                'na_string_count': int(na_string_count[col]),
                'total': int(total_missing),
                'percentage': round(total_missing / len(df) * 100, 2)
            }
    
    if quality_report['missing_values']:
        print(f"  存在缺失值的列数: {len(quality_report['missing_values'])}")
        for col, info in list(quality_report['missing_values'].items())[:5]:
            print(f"  - {col}: {info['total']}个 ({info['percentage']}%)")
        if len(quality_report['missing_values']) > 5:
            print(f"  ... 等共{len(quality_report['missing_values'])}个列")
    else:
        print("  ✓ 未检测到缺失值")
    
    # 2. 检测重复值
    print("\n--- 2. 重复值检测 ---")
    quality_report['duplicates'] = df.duplicated().sum()
    print(f"  重复行数: {quality_report['duplicates']}")
    
    # 3. 数据类型分析
    print("\n--- 3. 数据类型分析 ---")
    for dtype in df.dtypes.unique():
        cols = df.dtypes[df.dtypes == dtype].index.tolist()
        quality_report['data_types'][str(dtype)] = len(cols)
        print(f"  {dtype}: {len(cols)} 列")
    
    # 4. 数值型列的异常值检测（使用IQR方法）
    print("\n--- 4. 异常值检测 (IQR方法) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols[:OUTLIER_DETECTION_LIMIT]:  # 只显示前N列的检测结果
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 计算异常值数量（排除0值，因为0代表已淘汰）
        non_zero_data = df[col][df[col] != 0]
        outliers = non_zero_data[(non_zero_data < lower_bound) | (non_zero_data > upper_bound)]
        
        if len(outliers) > 0:
            quality_report['outliers'][col] = {
                'count': len(outliers),
                'percentage': round(len(outliers) / len(non_zero_data) * 100, 2) if len(non_zero_data) > 0 else 0,
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2)
            }
    
    if quality_report['outliers']:
        print(f"  检测到含异常值的列数: {len(quality_report['outliers'])}")
        for col, info in list(quality_report['outliers'].items())[:3]:
            print(f"  - {col}: {info['count']}个异常值 ({info['percentage']}%)")
    else:
        print("  ✓ 未检测到明显异常值")
    
    return quality_report


def assess_score_columns(df):
    """
    专门评估评分列的数据质量和分布
    
    Args:
        df: 核心数据DataFrame
    
    Returns:
        score_analysis: 评分数据分析报告
    """
    print("\n" + "=" * 70)
    print("【评分数据专项分析】")
    print("=" * 70)
    
    score_analysis = {
        'score_columns': [],
        'na_pattern': {},
        'zero_pattern': {},
        'score_range': {}
    }
    
    # 识别评分列 (week*_judge*_score 格式)
    score_cols = [col for col in df.columns if 'judge' in col and 'score' in col]
    score_analysis['score_columns'] = score_cols
    print(f"\n识别到 {len(score_cols)} 个评分列")
    
    # 分析N/A模式（第4评委缺席情况）
    print("\n--- N/A值分析（第4评委缺席情况）---")
    for col in score_cols:
        na_count = (df[col] == 'N/A').sum() if df[col].dtype == 'object' else 0
        if na_count > 0:
            score_analysis['na_pattern'][col] = na_count
    
    # 统计judge4的N/A情况
    judge4_cols = [col for col in score_cols if 'judge4' in col]
    judge4_na_total = sum(score_analysis['na_pattern'].get(col, 0) for col in judge4_cols)
    print(f"  第4评委(judge4)列N/A总数: {judge4_na_total}")
    print(f"  说明: N/A表示该周第4评委未参与评分（仅3位评委）")
    
    # 分析0值模式（已淘汰选手）
    print("\n--- 0值分析（已淘汰选手标记）---")
    zero_count_total = 0
    for col in score_cols:
        try:
            # 转换为数值型后计算0值
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            zero_count = (numeric_col == 0).sum()
            if zero_count > 0:
                score_analysis['zero_pattern'][col] = zero_count
                zero_count_total += zero_count
        except Exception:
            pass
    
    print(f"  评分列中0值总数: {zero_count_total}")
    print(f"  说明: 0值表示该选手在该周已被淘汰，不参与评分")
    
    # 分析有效评分范围
    print("\n--- 有效评分范围分析 ---")
    all_scores = []
    for col in score_cols:
        try:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            valid_scores = numeric_col[(numeric_col > 0) & (numeric_col.notna())]
            all_scores.extend(valid_scores.tolist())
        except Exception:
            pass
    
    if all_scores:
        score_analysis['score_range'] = {
            'min': round(min(all_scores), 2),
            'max': round(max(all_scores), 2),
            'mean': round(np.mean(all_scores), 2),
            'std': round(np.std(all_scores), 2)
        }
        print(f"  最小值: {score_analysis['score_range']['min']}")
        print(f"  最大值: {score_analysis['score_range']['max']}")
        print(f"  均值: {score_analysis['score_range']['mean']}")
        print(f"  标准差: {score_analysis['score_range']['std']}")
    
    return score_analysis


# =============================================================================
# 第四部分：数据清洗
# =============================================================================

def clean_main_data(df):
    """
    清洗核心数据
    
    处理策略:
    1. N/A值 -> np.nan (便于后续计算)
    2. 评分列转换为数值型
    3. 0值保留（作为"已淘汰"标记）
    4. 创建赛季规则标记
    
    Args:
        df: 原始核心数据DataFrame
    
    Returns:
        df_cleaned: 清洗后的数据
    """
    print("\n" + "=" * 70)
    print("【数据清洗】")
    print("=" * 70)
    
    df_cleaned = df.copy()
    
    # 1. 处理N/A字符串为np.nan
    print("\n--- 1. N/A值处理 ---")
    na_replaced = 0
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            mask = df_cleaned[col] == 'N/A'
            na_replaced += mask.sum()
            df_cleaned.loc[mask, col] = np.nan
    print(f"  ✓ 已将 {na_replaced} 个'N/A'字符串转换为np.nan")
    
    # 2. 评分列转换为数值型
    print("\n--- 2. 评分列数值化 ---")
    score_cols = [col for col in df_cleaned.columns if 'judge' in col and 'score' in col]
    for col in score_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    print(f"  ✓ 已将 {len(score_cols)} 个评分列转换为数值型")
    
    # 3. 处理celebrity_homestate空值（用"Unknown"填充）
    print("\n--- 3. 地区字段缺失值处理 ---")
    if 'celebrity_homestate' in df_cleaned.columns:
        missing_state = df_cleaned['celebrity_homestate'].isna().sum()
        df_cleaned['celebrity_homestate'] = df_cleaned['celebrity_homestate'].fillna('Unknown')
        print(f"  ✓ celebrity_homestate: {missing_state}个缺失值已用'Unknown'填充")
    
    # 4. 创建赛季规则标记
    print("\n--- 4. 创建赛季规则标记 ---")
    def get_season_rule(season):
        """
        根据赛季确定投票合并规则
        - 1-2季: 排名法(Ranking)
        - 3-27季: 百分比法(Percentage)
        - 28-34季: 排名法+评委决定(Ranking+JudgeSave)
        """
        if season <= 2:
            return 'Ranking'
        elif season <= 27:
            return 'Percentage'
        else:
            return 'Ranking_JudgeSave'
    
    df_cleaned['season_rule'] = df_cleaned['season'].apply(get_season_rule)
    rule_counts = df_cleaned['season_rule'].value_counts()
    for rule, count in rule_counts.items():
        print(f"  - {rule}: {count}条记录")
    
    # 5. 年龄字段数值化
    print("\n--- 5. 年龄字段处理 ---")
    if 'celebrity_age_during_season' in df_cleaned.columns:
        df_cleaned['celebrity_age_during_season'] = pd.to_numeric(
            df_cleaned['celebrity_age_during_season'], errors='coerce'
        )
        age_stats = df_cleaned['celebrity_age_during_season'].describe()
        print(f"  ✓ 年龄范围: {age_stats['min']:.0f} - {age_stats['max']:.0f}岁")
        print(f"  ✓ 平均年龄: {age_stats['mean']:.1f}岁")
    
    print(f"\n✓ 数据清洗完成: {df_cleaned.shape[0]} 行 × {df_cleaned.shape[1]} 列")
    
    return df_cleaned


def clean_supplementary_data(df):
    """
    清洗补充数据（社交媒体粉丝数据）
    
    Args:
        df: 补充数据DataFrame
    
    Returns:
        df_cleaned: 清洗后的补充数据
    """
    print("\n" + "=" * 70)
    print("【补充数据清洗】")
    print("=" * 70)
    
    df_cleaned = df.copy()
    
    # 识别粉丝数列
    follower_cols = [col for col in df_cleaned.columns if 'follower' in col.lower() or 'subscriber' in col.lower()]
    
    print(f"识别到 {len(follower_cols)} 个粉丝数相关列:")
    for col in follower_cols:
        # 计算有效值比例
        valid_count = df_cleaned[col].notna().sum()
        valid_pct = valid_count / len(df_cleaned) * 100
        print(f"  - {col}: {valid_count}个有效值 ({valid_pct:.1f}%)")
    
    # 对粉丝数列进行对数转换（后续分析用）
    for col in follower_cols:
        log_col_name = f"{col}_log"
        df_cleaned[log_col_name] = np.log1p(df_cleaned[col])
    
    print(f"\n✓ 补充数据清洗完成: {df_cleaned.shape[0]} 行 × {df_cleaned.shape[1]} 列")
    
    return df_cleaned


# =============================================================================
# 第五部分：特征工程
# =============================================================================

def engineer_features(df):
    """
    特征工程：构建分析所需的衍生特征
    
    包括:
    1. 各周评委总分
    2. 各周评委平均分
    3. 累积评分
    4. 评分趋势特征
    5. 类别特征编码
    
    Args:
        df: 清洗后的数据
    
    Returns:
        df_featured: 添加特征后的数据
    """
    print("\n" + "=" * 70)
    print("【特征工程】")
    print("=" * 70)
    
    df_featured = df.copy()
    
    # 1. 计算各周评委总分
    print("\n--- 1. 计算各周评委总分 ---")
    for week in range(1, 12):
        judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        existing_cols = [col for col in judge_cols if col in df_featured.columns]
        
        if existing_cols:
            # 计算总分（排除NaN）
            df_featured[f'week{week}_total_score'] = df_featured[existing_cols].sum(axis=1, skipna=True)
            # 计算平均分（排除NaN）
            df_featured[f'week{week}_avg_score'] = df_featured[existing_cols].mean(axis=1, skipna=True)
            # 计算有效评委数
            df_featured[f'week{week}_judge_count'] = df_featured[existing_cols].notna().sum(axis=1)
    
    print("  ✓ 已创建各周总分、平均分、有效评委数特征")
    
    # 2. 计算累积评分特征
    print("\n--- 2. 计算累积评分特征 ---")
    total_score_cols = [col for col in df_featured.columns if '_total_score' in col]
    df_featured['cumulative_total_score'] = df_featured[total_score_cols].sum(axis=1, skipna=True)
    
    avg_score_cols = [col for col in df_featured.columns if '_avg_score' in col]
    df_featured['overall_avg_score'] = df_featured[avg_score_cols].mean(axis=1, skipna=True)
    print("  ✓ 已创建累积总分和整体平均分特征")
    
    # 3. 计算评分趋势（简单线性趋势）
    print("\n--- 3. 计算评分趋势特征 ---")
    def calculate_trend(row):
        """计算选手评分的线性趋势斜率"""
        scores = []
        weeks = []
        for week in range(1, 12):
            score_col = f'week{week}_avg_score'
            if score_col in row.index and pd.notna(row[score_col]) and row[score_col] > 0:
                scores.append(row[score_col])
                weeks.append(week)
        
        if len(scores) >= 2:
            slope, _, _, _, _ = stats.linregress(weeks, scores)
            return slope
        return 0
    
    df_featured['score_trend'] = df_featured.apply(calculate_trend, axis=1)
    print("  ✓ 已创建评分趋势特征")
    
    # 4. 解析比赛结果，提取淘汰周数
    print("\n--- 4. 解析比赛结果 ---")
    def parse_result(result):
        """解析结果字符串，提取关键信息"""
        if pd.isna(result):
            return {'is_winner': False, 'eliminated_week': None, 'final_rank': None}
        
        result = str(result)
        
        if '1st Place' in result:
            return {'is_winner': True, 'eliminated_week': None, 'final_rank': 1}
        elif '2nd Place' in result:
            return {'is_winner': False, 'eliminated_week': None, 'final_rank': 2}
        elif '3rd Place' in result:
            return {'is_winner': False, 'eliminated_week': None, 'final_rank': 3}
        elif 'Eliminated Week' in result:
            try:
                week = int(result.split('Week')[-1].strip())
                return {'is_winner': False, 'eliminated_week': week, 'final_rank': None}
            except ValueError:
                return {'is_winner': False, 'eliminated_week': None, 'final_rank': None}
        elif 'Withdrew' in result:
            return {'is_winner': False, 'eliminated_week': WITHDREW_WEEK, 'final_rank': None}  # 使用常量表示退赛
        else:
            return {'is_winner': False, 'eliminated_week': None, 'final_rank': None}
    
    result_parsed = df_featured['results'].apply(parse_result)
    df_featured['is_winner'] = result_parsed.apply(lambda x: x['is_winner'])
    df_featured['eliminated_week'] = result_parsed.apply(lambda x: x['eliminated_week'])
    df_featured['final_rank'] = result_parsed.apply(lambda x: x['final_rank'])
    
    winner_count = df_featured['is_winner'].sum()
    print(f"  ✓ 冠军数量: {winner_count}")
    print(f"  ✓ 已创建is_winner, eliminated_week, final_rank特征")
    
    # 5. 类别特征编码
    print("\n--- 5. 类别特征编码 ---")
    
    # 行业编码
    if 'celebrity_industry' in df_featured.columns:
        le_industry = LabelEncoder()
        df_featured['industry_encoded'] = le_industry.fit_transform(
            df_featured['celebrity_industry'].fillna('Unknown')
        )
        industry_mapping = dict(zip(le_industry.classes_, range(len(le_industry.classes_))))
        print(f"  ✓ 行业编码完成，共{len(industry_mapping)}个类别")
    
    # 国家/地区编码
    if 'celebrity_homecountry/region' in df_featured.columns:
        le_country = LabelEncoder()
        df_featured['country_encoded'] = le_country.fit_transform(
            df_featured['celebrity_homecountry/region'].fillna('Unknown')
        )
        print(f"  ✓ 国家/地区编码完成")
    
    # 赛季规则编码
    if 'season_rule' in df_featured.columns:
        rule_mapping = {'Ranking': 0, 'Percentage': 1, 'Ranking_JudgeSave': 2}
        df_featured['season_rule_encoded'] = df_featured['season_rule'].map(rule_mapping)
        print(f"  ✓ 赛季规则编码完成")
    
    # 6. 计算参与周数
    print("\n--- 6. 计算参与周数 ---")
    def count_active_weeks(row):
        """计算选手实际参与比赛的周数"""
        count = 0
        for week in range(1, 12):
            score_col = f'week{week}_total_score'
            if score_col in row.index and pd.notna(row[score_col]) and row[score_col] > 0:
                count += 1
        return count
    
    df_featured['active_weeks'] = df_featured.apply(count_active_weeks, axis=1)
    print(f"  ✓ 参与周数范围: {df_featured['active_weeks'].min()} - {df_featured['active_weeks'].max()}周")
    
    print(f"\n✓ 特征工程完成，当前特征数: {len(df_featured.columns)}")
    
    return df_featured


def prepare_model_data(df):
    """
    为各问题模型准备专用数据集
    
    Args:
        df: 特征工程后的数据
    
    Returns:
        model_datasets: 包含各模型所需数据的字典
    """
    print("\n" + "=" * 70)
    print("【模型专用数据准备】")
    print("=" * 70)
    
    model_datasets = {}
    
    # 问题1专用数据：粉丝投票估算
    print("\n--- 问题1: 粉丝投票估算模型数据 ---")
    q1_cols = ['celebrity_name', 'season', 'season_rule', 'placement', 'results']
    
    # 添加各周评分列
    for week in range(1, 12):
        for judge in range(1, 5):
            col = f'week{week}_judge{judge}_score'
            if col in df.columns:
                q1_cols.append(col)
        # 添加各周总分
        total_col = f'week{week}_total_score'
        if total_col in df.columns:
            q1_cols.append(total_col)
    
    q1_cols = [col for col in q1_cols if col in df.columns]
    model_datasets['question1'] = df[q1_cols].copy()
    print(f"  ✓ 问题1数据: {model_datasets['question1'].shape}")
    
    # 问题2专用数据：方法对比分析
    print("\n--- 问题2: 方法对比分析数据 ---")
    q2_cols = ['celebrity_name', 'season', 'season_rule', 'placement', 'results',
               'cumulative_total_score', 'overall_avg_score']
    
    # 添加周评分特征
    for week in range(1, 12):
        for col_type in ['total_score', 'avg_score']:
            col = f'week{week}_{col_type}'
            if col in df.columns:
                q2_cols.append(col)
    
    q2_cols = [col for col in q2_cols if col in df.columns]
    model_datasets['question2'] = df[q2_cols].copy()
    print(f"  ✓ 问题2数据: {model_datasets['question2'].shape}")
    
    # 问题3专用数据：特征影响分析
    print("\n--- 问题3: 特征影响分析数据 ---")
    q3_cols = ['celebrity_name', 'ballroom_partner', 'celebrity_industry', 
               'celebrity_homestate', 'celebrity_homecountry/region',
               'celebrity_age_during_season', 'season', 'placement',
               'cumulative_total_score', 'overall_avg_score', 'score_trend',
               'industry_encoded', 'country_encoded', 'active_weeks',
               'is_winner', 'final_rank']
    
    q3_cols = [col for col in q3_cols if col in df.columns]
    model_datasets['question3'] = df[q3_cols].copy()
    print(f"  ✓ 问题3数据: {model_datasets['question3'].shape}")
    
    # 问题4专用数据：新系统设计（使用全部特征）
    print("\n--- 问题4: 新系统设计数据 ---")
    model_datasets['question4'] = df.copy()
    print(f"  ✓ 问题4数据: {model_datasets['question4'].shape}")
    
    return model_datasets


# =============================================================================
# 第六部分：数据可视化
# =============================================================================

def visualize_data_quality(df_original, df_cleaned, output_dir):
    """
    数据质量可视化分析
    
    Args:
        df_original: 原始数据
        df_cleaned: 清洗后数据
        output_dir: 图片输出目录
    """
    print("\n" + "=" * 70)
    print("【数据可视化分析】")
    print("=" * 70)
    
    # 1. 缺失值热力图
    print("\n--- 1. 生成缺失值热力图 ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 选择关键列进行可视化
    key_cols = ['celebrity_name', 'celebrity_industry', 'celebrity_homestate',
                'celebrity_age_during_season', 'season', 'placement']
    score_sample_cols = [f'week{w}_judge{j}_score' for w in VIS_SAMPLE_WEEKS for j in VIS_SAMPLE_JUDGES]
    vis_cols = key_cols + [col for col in score_sample_cols if col in df_original.columns]
    
    # 原始数据缺失值
    missing_original = df_original[vis_cols].isnull() | (df_original[vis_cols] == 'N/A')
    sns.heatmap(missing_original.head(50).T, cbar=True, yticklabels=True, 
                cmap='YlOrRd', ax=axes[0])
    axes[0].set_title('Original Data - Missing Values (First 50 Records)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Record Index')
    axes[0].set_ylabel('Columns')
    
    # 清洗后数据缺失值
    missing_cleaned = df_cleaned[[col for col in vis_cols if col in df_cleaned.columns]].isnull()
    sns.heatmap(missing_cleaned.head(50).T, cbar=True, yticklabels=True,
                cmap='YlOrRd', ax=axes[1])
    axes[1].set_title('Cleaned Data - Missing Values (First 50 Records)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Record Index')
    axes[1].set_ylabel('Columns')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_missing_values_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: 01_missing_values_heatmap.png")
    
    # 2. 评分分布可视化
    print("\n--- 2. 生成评分分布图 ---")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 收集各周评分
    for idx, week in enumerate(VIS_SCORE_WEEKS):
        ax = axes[idx // 3, idx % 3]
        total_col = f'week{week}_total_score'
        
        if total_col in df_cleaned.columns:
            # 只绘制有效评分（排除0值）
            valid_scores = df_cleaned[df_cleaned[total_col] > 0][total_col]
            if len(valid_scores) > 0:
                sns.histplot(valid_scores, bins=20, kde=True, ax=ax, color='steelblue', alpha=0.7)
                ax.axvline(valid_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {valid_scores.mean():.1f}')
                ax.set_title(f'Week {week} Total Score Distribution', fontsize=11, fontweight='bold')
                ax.set_xlabel('Total Score')
                ax.set_ylabel('Frequency')
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: 02_score_distribution.png")
    
    # 3. 赛季分布与规则变化可视化
    print("\n--- 3. 生成赛季分布图 ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 各赛季参赛人数
    season_counts = df_cleaned['season'].value_counts().sort_index()
    colors = ['#2E86AB' if s <= 2 else '#A23B72' if s <= 27 else '#F18F01' 
              for s in season_counts.index]
    axes[0].bar(season_counts.index, season_counts.values, color=colors, edgecolor='white')
    axes[0].set_title('Number of Contestants per Season', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Season')
    axes[0].set_ylabel('Number of Contestants')
    
    # 添加规则分区标注
    axes[0].axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].axvline(x=27.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].text(1.5, axes[0].get_ylim()[1]*0.9, 'Ranking\nMethod', ha='center', fontsize=9)
    axes[0].text(15, axes[0].get_ylim()[1]*0.9, 'Percentage\nMethod', ha='center', fontsize=9)
    axes[0].text(31, axes[0].get_ylim()[1]*0.9, 'Ranking+\nJudge Save', ha='center', fontsize=9)
    
    # 赛季规则分布饼图
    if 'season_rule' in df_cleaned.columns:
        rule_counts = df_cleaned['season_rule'].value_counts()
        colors_pie = ['#2E86AB', '#A23B72', '#F18F01']
        axes[1].pie(rule_counts.values, labels=rule_counts.index, autopct='%1.1f%%',
                    colors=colors_pie, startangle=90, explode=[0.02]*len(rule_counts))
        axes[1].set_title('Season Rule Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_season_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: 03_season_distribution.png")
    
    # 4. 行业分布可视化
    print("\n--- 4. 生成行业分布图 ---")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    industry_counts = df_cleaned['celebrity_industry'].value_counts()
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(industry_counts)))
    bars = ax.barh(industry_counts.index, industry_counts.values, color=colors_bar, edgecolor='white')
    ax.set_title('Celebrity Industry Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Contestants')
    ax.set_ylabel('Industry')
    
    # 添加数值标签
    for bar, val in zip(bars, industry_counts.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val), 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_industry_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: 04_industry_distribution.png")
    
    # 5. 年龄分布可视化
    print("\n--- 5. 生成年龄分布图 ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 年龄直方图
    age_data = df_cleaned['celebrity_age_during_season'].dropna()
    sns.histplot(age_data, bins=25, kde=True, ax=axes[0], color='coral', alpha=0.7)
    axes[0].axvline(age_data.mean(), color='darkred', linestyle='--', linewidth=2,
                    label=f'Mean: {age_data.mean():.1f}')
    axes[0].axvline(age_data.median(), color='blue', linestyle='--', linewidth=2,
                    label=f'Median: {age_data.median():.1f}')
    axes[0].set_title('Celebrity Age Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Age During Season')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # 年龄与最终排名箱线图
    placement_data = df_cleaned[df_cleaned['placement'].notna()].copy()
    placement_data['placement_group'] = placement_data['placement'].apply(
        lambda x: 'Top 3' if x <= 3 else ('4-6' if x <= 6 else '7+')
    )
    sns.boxplot(x='placement_group', y='celebrity_age_during_season', 
                data=placement_data, ax=axes[1], palette='Set2',
                order=['Top 3', '4-6', '7+'])
    axes[1].set_title('Age Distribution by Final Placement', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Placement Group')
    axes[1].set_ylabel('Age')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_age_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: 05_age_distribution.png")
    
    # 6. 特征相关性热力图
    print("\n--- 6. 生成特征相关性热力图 ---")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 选择数值型特征
    numeric_features = ['celebrity_age_during_season', 'season', 'placement',
                        'cumulative_total_score', 'overall_avg_score', 'score_trend',
                        'active_weeks']
    numeric_features = [f for f in numeric_features if f in df_cleaned.columns]
    
    corr_matrix = df_cleaned[numeric_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 10})
    ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: 06_correlation_heatmap.png")
    
    # 7. 评分趋势可视化（选取典型案例）
    print("\n--- 7. 生成评分趋势案例图 ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 选取4个典型案例：冠军、亚军、争议案例
    cases = [
        ('Kelly Monaco', 'Season 1 Winner'),
        ('Jerry Rice', 'Season 2 - Controversial (Low scores, 2nd place)'),
        ('Bristol Palin', 'Season 11 - Controversial'),
        ('Bobby Bones', 'Season 27 - Controversial Winner')
    ]
    
    for idx, (name, description) in enumerate(cases):
        ax = axes[idx // 2, idx % 2]
        
        contestant_data = df_cleaned[df_cleaned['celebrity_name'] == name]
        if len(contestant_data) > 0:
            scores = []
            weeks = []
            for week in range(1, 12):
                col = f'week{week}_avg_score'
                if col in contestant_data.columns:
                    score = contestant_data[col].values[0]
                    if pd.notna(score) and score > 0:
                        scores.append(score)
                        weeks.append(week)
            
            if scores:
                ax.plot(weeks, scores, 'o-', linewidth=2, markersize=8, color='steelblue')
                ax.fill_between(weeks, scores, alpha=0.3, color='steelblue')
                ax.set_ylim(0, 11)
        
        ax.set_title(f'{name}\n({description})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Week')
        ax.set_ylabel('Average Judge Score')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_score_trends_cases.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: 07_score_trends_cases.png")
    
    # 8. 异常值检测可视化
    print("\n--- 8. 生成异常值检测图 ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 累积总分箱线图
    ax = axes[0, 0]
    if 'cumulative_total_score' in df_cleaned.columns:
        valid_data = df_cleaned[df_cleaned['cumulative_total_score'] > 0]['cumulative_total_score']
        bp = ax.boxplot(valid_data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax.set_title('Cumulative Total Score - Outlier Detection', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score')
        
        # 标注异常值数量
        Q1, Q3 = np.percentile(valid_data, [25, 75])
        IQR = Q3 - Q1
        outliers = valid_data[(valid_data < Q1 - 1.5*IQR) | (valid_data > Q3 + 1.5*IQR)]
        ax.text(0.95, 0.95, f'Outliers: {len(outliers)}', transform=ax.transAxes,
                ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # 年龄箱线图
    ax = axes[0, 1]
    age_data = df_cleaned['celebrity_age_during_season'].dropna()
    bp = ax.boxplot(age_data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    ax.set_title('Celebrity Age - Outlier Detection', fontsize=11, fontweight='bold')
    ax.set_ylabel('Age')
    
    # Week 1 vs Week 5 评分散点图
    ax = axes[1, 0]
    if 'week1_total_score' in df_cleaned.columns and 'week5_total_score' in df_cleaned.columns:
        valid_data = df_cleaned[(df_cleaned['week1_total_score'] > 0) & (df_cleaned['week5_total_score'] > 0)]
        ax.scatter(valid_data['week1_total_score'], valid_data['week5_total_score'],
                   alpha=0.6, c='coral', edgecolors='white', s=60)
        ax.set_xlabel('Week 1 Total Score')
        ax.set_ylabel('Week 5 Total Score')
        ax.set_title('Week 1 vs Week 5 Score Comparison', fontsize=11, fontweight='bold')
        
        # 添加对角线参考
        lims = [max(ax.get_xlim()[0], ax.get_ylim()[0]),
                min(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    
    # 参与周数分布
    ax = axes[1, 1]
    if 'active_weeks' in df_cleaned.columns:
        week_counts = df_cleaned['active_weeks'].value_counts().sort_index()
        ax.bar(week_counts.index, week_counts.values, color='mediumpurple', edgecolor='white')
        ax.set_xlabel('Number of Active Weeks')
        ax.set_ylabel('Number of Contestants')
        ax.set_title('Distribution of Active Weeks', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_outlier_detection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: 08_outlier_detection.png")
    
    print(f"\n✓ 可视化分析完成，共生成8张图表")


# =============================================================================
# 第七部分：数据导出
# =============================================================================

def export_data(df_main, df_supp, model_datasets, output_dir):
    """
    导出预处理后的数据
    
    Args:
        df_main: 处理后的核心数据
        df_supp: 处理后的补充数据
        model_datasets: 各模型专用数据集
        output_dir: 输出目录
    """
    print("\n" + "=" * 70)
    print("【数据导出】")
    print("=" * 70)
    
    # 1. 导出完整处理后的核心数据
    main_path = f'{output_dir}/processed_main_data.csv'
    df_main.to_csv(main_path, index=False, encoding='utf-8-sig')
    print(f"✓ 核心数据已导出: {main_path}")
    print(f"  大小: {df_main.shape[0]} 行 × {df_main.shape[1]} 列")
    
    # 2. 导出补充数据
    supp_path = f'{output_dir}/processed_supplementary_data.csv'
    df_supp.to_csv(supp_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 补充数据已导出: {supp_path}")
    print(f"  大小: {df_supp.shape[0]} 行 × {df_supp.shape[1]} 列")
    
    # 3. 导出各问题专用数据集
    print("\n--- 各问题专用数据集导出 ---")
    for name, data in model_datasets.items():
        path = f'{output_dir}/{name}_data.csv'
        data.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"✓ {name} 数据已导出: {path} ({data.shape})")
    
    # 4. 导出数据字典
    print("\n--- 导出数据字典 ---")
    data_dict = []
    for col in df_main.columns:
        dtype = str(df_main[col].dtype)
        non_null = df_main[col].notna().sum()
        sample = str(df_main[col].dropna().iloc[0]) if non_null > 0 else 'N/A'
        if len(sample) > 50:
            sample = sample[:50] + '...'
        
        data_dict.append({
            'Column Name': col,
            'Data Type': dtype,
            'Non-Null Count': non_null,
            'Null Count': len(df_main) - non_null,
            'Sample Value': sample
        })
    
    dict_df = pd.DataFrame(data_dict)
    dict_path = f'{output_dir}/data_dictionary.csv'
    dict_df.to_csv(dict_path, index=False, encoding='utf-8-sig')
    print(f"✓ 数据字典已导出: {dict_path}")
    
    print(f"\n✓ 所有数据导出完成！输出目录: {output_dir}")


# =============================================================================
# 第八部分：生成预处理报告
# =============================================================================

def generate_report(quality_report, score_analysis, df_cleaned, output_dir):
    """
    生成数据预处理摘要报告
    
    Args:
        quality_report: 数据质量报告
        score_analysis: 评分数据分析报告
        df_cleaned: 清洗后的数据
        output_dir: 输出目录
    """
    print("\n" + "=" * 70)
    print("【生成预处理报告】")
    print("=" * 70)
    
    report_lines = [
        "# 数据预处理报告摘要",
        "",
        "## 1. 数据概览",
        f"- 总记录数: {quality_report['total_rows']}",
        f"- 原始字段数: {quality_report['total_cols']}",
        f"- 处理后字段数: {len(df_cleaned.columns)}",
        "",
        "## 2. 数据质量评估",
        "",
        "### 2.1 缺失值处理",
        f"- 存在缺失值的列数: {len(quality_report['missing_values'])}",
        "- 主要缺失值来源:",
        "  - judge4列的N/A值: 第4评委部分周次未参与评分",
        "  - 0值标记: 表示选手在该周已被淘汰",
        "- 处理策略: N/A转换为np.nan，0值保留作为淘汰标记",
        "",
        "### 2.2 异常值检测",
        f"- 检测到含异常值的列数: {len(quality_report['outliers'])}",
        "- 检测方法: IQR方法 (Q1-1.5*IQR, Q3+1.5*IQR)",
        "- 处理策略: 保留异常值（属于正常业务数据范围）",
        "",
        "### 2.3 重复值检测",
        f"- 重复行数: {quality_report['duplicates']}",
        "",
        "## 3. 评分数据分析",
        f"- 评分列数量: {len(score_analysis['score_columns'])}",
        f"- 有效评分范围: {score_analysis['score_range']['min']} - {score_analysis['score_range']['max']}",
        f"- 评分均值: {score_analysis['score_range']['mean']}",
        f"- 评分标准差: {score_analysis['score_range']['std']}",
        "",
        "## 4. 特征工程",
        "- 新增特征:",
        "  - 各周评委总分 (week*_total_score)",
        "  - 各周评委平均分 (week*_avg_score)",
        "  - 累积总分 (cumulative_total_score)",
        "  - 整体平均分 (overall_avg_score)",
        "  - 评分趋势 (score_trend)",
        "  - 赛季规则标记 (season_rule)",
        "  - 参与周数 (active_weeks)",
        "  - 是否冠军 (is_winner)",
        "  - 类别编码 (industry_encoded, country_encoded)",
        "",
        "## 5. 数据划分说明",
        "- 按赛季规则划分:",
        f"  - 排名法 (Season 1-2): {len(df_cleaned[df_cleaned['season_rule'] == 'Ranking'])} 条",
        f"  - 百分比法 (Season 3-27): {len(df_cleaned[df_cleaned['season_rule'] == 'Percentage'])} 条",
        f"  - 排名法+评委决定 (Season 28-34): {len(df_cleaned[df_cleaned['season_rule'] == 'Ranking_JudgeSave'])} 条",
        "",
        "## 6. 输出文件清单",
        "- processed_main_data.csv: 处理后的核心数据",
        "- processed_supplementary_data.csv: 处理后的补充数据",
        "- question1_data.csv: 问题1专用数据集",
        "- question2_data.csv: 问题2专用数据集",
        "- question3_data.csv: 问题3专用数据集",
        "- question4_data.csv: 问题4专用数据集",
        "- data_dictionary.csv: 数据字典",
        "- 01-08_*.png: 可视化分析图表",
        "",
        "---",
        "生成时间: 2026年MCM竞赛",
        "版本: v1.0"
    ]
    
    report_path = f'{output_dir}/preprocessing_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 预处理报告已生成: {report_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """
    主执行函数
    """
    print("\n")
    print("=" * 70)
    print("   2026 MCM Problem C: Dancing with the Stars")
    print("   数据预处理模块 v1.0")
    print("=" * 70)
    print("\n")
    
    # 1. 数据加载
    df_main, df_supp = load_data()
    
    # 2. 数据质量评估
    quality_report = assess_data_quality(df_main, "核心数据")
    assess_data_quality(df_supp, "补充数据")
    score_analysis = assess_score_columns(df_main)
    
    # 3. 数据清洗
    df_main_cleaned = clean_main_data(df_main)
    df_supp_cleaned = clean_supplementary_data(df_supp)
    
    # 4. 特征工程
    df_featured = engineer_features(df_main_cleaned)
    
    # 5. 准备模型专用数据
    model_datasets = prepare_model_data(df_featured)
    
    # 6. 数据可视化
    visualize_data_quality(df_main, df_featured, OUTPUT_DIR)
    
    # 7. 数据导出
    export_data(df_featured, df_supp_cleaned, model_datasets, OUTPUT_DIR)
    
    # 8. 生成报告
    generate_report(quality_report, score_analysis, df_featured, OUTPUT_DIR)
    
    print("\n")
    print("=" * 70)
    print("   数据预处理完成！")
    print(f"   输出目录: {OUTPUT_DIR}")
    print("=" * 70)
    print("\n")
    
    return df_featured, df_supp_cleaned, model_datasets


if __name__ == "__main__":
    df_processed, df_supp_processed, model_data = main()
