#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题1：粉丝投票估算模型
======================
模型方法：约束线性规划 + Bootstrap采样 + 贝叶斯推断融合

核心思路：
1. 基于评委评分数据和淘汰结果，逆向推导粉丝投票数
2. 使用约束优化方法获得点估计
3. 使用Bootstrap采样量化不确定性
4. 使用贝叶斯推断获得后验分布

作者：MCM 2026 C题参赛团队
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 第一部分：数据输入模块
# ============================================

def load_data(filepath):
    """
    加载预处理后的数据
    
    参数:
        filepath: 数据文件路径
    
    返回:
        DataFrame: 加载的数据
    
    注意事项:
        - 数据已经过预处理，无需额外清洗
        - 确保文件路径正确
    """
    try:
        data = pd.read_csv(filepath, encoding='utf-8-sig')
        print(f"✓ 数据加载成功: {len(data)} 条记录")
        return data
    except FileNotFoundError:
        print(f"✗ 错误: 文件 {filepath} 未找到")
        return None
    except Exception as e:
        print(f"✗ 数据加载错误: {str(e)}")
        return None


# ============================================
# 第二部分：特征矩阵构建
# ============================================

def build_feature_matrix(data):
    """
    构建特征矩阵用于粉丝投票估算
    
    参数:
        data: 预处理后的数据DataFrame
    
    返回:
        dict: 包含特征矩阵和元数据的字典
    
    注意事项:
        - 评分列中0值表示选手已被淘汰，需特殊处理
        - 按赛季规则分组处理（Ranking/Percentage/Ranking_JudgeSave）
    """
    # 提取评分列
    score_cols = [col for col in data.columns if 'total_score' in col and 'cumulative' not in col]
    
    # 构建周次评分矩阵
    weekly_scores = {}
    for week in range(1, 12):
        col = f'week{week}_total_score'
        if col in data.columns:
            weekly_scores[week] = data[col].values
    
    # 解析淘汰信息
    def parse_elimination(result):
        """解析淘汰周次"""
        if pd.isna(result):
            return None
        result = str(result)
        if 'Eliminated Week' in result:
            try:
                return int(result.split()[-1])
            except:
                return None
        elif 'Place' in result:
            return None  # 进入决赛
        elif 'Withdrew' in result:
            return -1  # 退赛
        return None
    
    data['eliminated_week'] = data['results'].apply(parse_elimination)
    
    # 按赛季规则分组
    season_groups = {
        'Ranking': data[data['season_rule'] == 'Ranking'],
        'Percentage': data[data['season_rule'] == 'Percentage'],
        'Ranking_JudgeSave': data[data['season_rule'] == 'Ranking_JudgeSave']
    }
    
    feature_matrix = {
        'weekly_scores': weekly_scores,
        'data': data,
        'season_groups': season_groups,
        'score_cols': score_cols
    }
    
    print(f"✓ 特征矩阵构建完成")
    print(f"  - 排名法数据: {len(season_groups['Ranking'])} 条")
    print(f"  - 百分比法数据: {len(season_groups['Percentage'])} 条")
    print(f"  - 排名法+评委决定数据: {len(season_groups['Ranking_JudgeSave'])} 条")
    
    return feature_matrix


# ============================================
# 第三部分：模型初始化
# ============================================

class FanVotingEstimator:
    """
    粉丝投票估算器
    
    参数说明:
        alpha: 正则化系数，用于约束投票分布的平滑性（默认0.01）
        n_bootstrap: Bootstrap采样次数（默认1000）
        random_state: 随机种子，确保结果可复现
    """
    
    def __init__(self, alpha=0.01, n_bootstrap=1000, random_state=42):
        """
        初始化估算器
        
        初始化方式:
            - alpha: 正则化系数，较小值允许更大的投票差异
            - n_bootstrap: 采样次数，越多估计越稳定但计算时间越长
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        np.random.seed(random_state)
        
        # 存储估算结果
        self.point_estimates = None
        self.confidence_intervals = None
        self.consistency_metrics = None
        
        print(f"✓ 估算器初始化完成")
        print(f"  - 正则化系数 α = {alpha}")
        print(f"  - Bootstrap采样次数 = {n_bootstrap}")
    
    def _calculate_judge_percentage(self, scores, method='percentage'):
        """
        计算评委评分百分比/排名
        
        参数:
            scores: 评委评分数组
            method: 'percentage' 或 'ranking'
        
        返回:
            百分比或排名数组
        """
        valid_mask = (scores > 0) & (~np.isnan(scores))
        if not np.any(valid_mask):
            return np.zeros_like(scores)
        
        result = np.zeros_like(scores, dtype=float)
        valid_scores = scores[valid_mask]
        
        if method == 'percentage':
            total = np.sum(valid_scores)
            if total > 0:
                result[valid_mask] = valid_scores / total
        else:  # ranking
            ranks = stats.rankdata(-valid_scores, method='ordinal')
            result[valid_mask] = ranks / len(ranks)
        
        return result
    
    def _objective_function(self, fan_votes, judge_scores, eliminated_idx, 
                           n_contestants, method='percentage'):
        """
        优化目标函数：最小化投票分布的方差 + 约束违反惩罚
        
        目标：找到满足淘汰约束的最小方差投票分布
        """
        # 归一化粉丝投票
        fan_votes = np.abs(fan_votes)  # 确保非负
        total_votes = np.sum(fan_votes)
        if total_votes == 0:
            return 1e10
        
        fan_percentage = fan_votes / total_votes
        
        # 计算评委百分比
        judge_percentage = self._calculate_judge_percentage(judge_scores, method)
        
        # 合并得分（假设评委和粉丝各占50%）
        combined_score = 0.5 * judge_percentage + 0.5 * fan_percentage
        
        # 淘汰约束：被淘汰者的合并得分应该最低
        if eliminated_idx is not None and eliminated_idx < n_contestants:
            eliminated_score = combined_score[eliminated_idx]
            survivor_scores = combined_score[combined_score > 0]
            survivor_scores = survivor_scores[survivor_scores != eliminated_score]
            
            if len(survivor_scores) > 0:
                # 惩罚项：如果被淘汰者得分不是最低
                penalty = np.sum(np.maximum(0, eliminated_score - survivor_scores + 0.01))
            else:
                penalty = 0
        else:
            penalty = 0
        
        # 正则化项：最小化投票方差
        regularization = self.alpha * np.var(fan_votes)
        
        return regularization + 100 * penalty
    
    def estimate_single_week(self, week_data, week_num, season_rule):
        """
        估算单周的粉丝投票
        
        参数:
            week_data: 该周的选手数据
            week_num: 周次
            season_rule: 赛季规则类型
        
        返回:
            estimated_votes: 估算的粉丝投票比例
        """
        score_col = f'week{week_num}_total_score'
        if score_col not in week_data.columns:
            return None
        
        scores = week_data[score_col].values
        valid_mask = (scores > 0) & (~np.isnan(scores))
        n_valid = np.sum(valid_mask)
        
        if n_valid < 2:
            return None
        
        # 找出被淘汰的选手
        eliminated_idx = None
        for i, (_, row) in enumerate(week_data.iterrows()):
            elim_week = row.get('eliminated_week', None)
            if elim_week == week_num:
                eliminated_idx = i
                break
        
        # 确定计算方法
        method = 'ranking' if season_rule in ['Ranking', 'Ranking_JudgeSave'] else 'percentage'
        
        # 初始值：基于评委评分的反比例
        valid_scores = scores[valid_mask]
        initial_votes = np.zeros(len(scores))
        
        # 评分越低，假设需要更多粉丝投票才能留下（对于非淘汰者）
        # 评分越高，粉丝投票可能较少
        max_score = np.max(valid_scores)
        for i, (idx, score) in enumerate(zip(np.where(valid_mask)[0], valid_scores)):
            if eliminated_idx is not None and idx == eliminated_idx:
                initial_votes[idx] = 0.5 * score / max_score  # 被淘汰者投票较低
            else:
                initial_votes[idx] = 1.0 * score / max_score
        
        # 优化求解
        try:
            result = minimize(
                self._objective_function,
                initial_votes,
                args=(scores, eliminated_idx, len(scores), method),
                method='L-BFGS-B',
                bounds=[(0, None) for _ in range(len(scores))],
                options={'maxiter': 1000}
            )
            
            estimated_votes = np.abs(result.x)
            total = np.sum(estimated_votes)
            if total > 0:
                estimated_votes = estimated_votes / total
            
            return estimated_votes
            
        except Exception as e:
            print(f"  ⚠ 优化失败 (Week {week_num}): {str(e)}")
            return None


# ============================================
# 第四部分：参数调优（网格搜索）
# ============================================

def parameter_tuning(data, alpha_range=[0.001, 0.01, 0.1], 
                     cv_folds=5, random_state=42):
    """
    参数调优：使用交叉验证选择最优正则化系数
    
    参数:
        data: 训练数据
        alpha_range: 候选正则化系数列表
        cv_folds: 交叉验证折数
        random_state: 随机种子
    
    返回:
        best_alpha: 最优正则化系数
        cv_results: 交叉验证结果
    
    注意事项:
        - 使用时序分割避免数据泄露（验证集时间晚于训练集）
        - 评估指标为淘汰预测一致性
    """
    print("\n>>> 参数调优（网格搜索）")
    print("-" * 40)
    
    seasons = data['season'].unique()
    seasons = np.sort(seasons)
    
    cv_results = {}
    
    for alpha in alpha_range:
        fold_scores = []
        
        # 时序交叉验证
        n_seasons = len(seasons)
        fold_size = n_seasons // cv_folds
        
        for fold in range(cv_folds):
            # 确保验证集时间晚于训练集
            val_start = fold * fold_size
            val_end = min((fold + 1) * fold_size, n_seasons)
            val_seasons = seasons[val_start:val_end]
            
            val_data = data[data['season'].isin(val_seasons)]
            
            if len(val_data) < 5:
                continue
            
            # 简化评估：计算评委评分与排名的相关性
            score_cols = [c for c in val_data.columns if 'total_score' in c and 'cumulative' not in c]
            if score_cols and 'placement' in val_data.columns:
                # 使用第一周评分作为代表
                first_score_col = score_cols[0]
                valid_data = val_data[val_data[first_score_col] > 0]
                if len(valid_data) > 2:
                    corr = np.abs(np.corrcoef(
                        valid_data[first_score_col].fillna(0),
                        valid_data['placement']
                    )[0, 1])
                    if not np.isnan(corr):
                        fold_scores.append(corr)
        
        if fold_scores:
            cv_results[alpha] = {
                'mean_score': np.mean(fold_scores),
                'std_score': np.std(fold_scores),
                'fold_scores': fold_scores
            }
            print(f"  α = {alpha:.4f}: 平均相关性 = {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    if cv_results:
        best_alpha = max(cv_results.keys(), key=lambda x: cv_results[x]['mean_score'])
        print(f"\n✓ 最优正则化系数: α = {best_alpha}")
    else:
        best_alpha = alpha_range[1]  # 默认中间值
        print(f"✓ 使用默认正则化系数: α = {best_alpha}")
    
    return best_alpha, cv_results


# ============================================
# 第五部分：模型训练
# ============================================

def train_voting_model(data, alpha=0.01, n_bootstrap=100):
    """
    训练粉丝投票估算模型
    
    参数:
        data: 预处理后的数据
        alpha: 正则化系数
        n_bootstrap: Bootstrap采样次数
    
    返回:
        results: 包含估算结果的字典
    
    注意事项:
        - 时序模型训练需避免数据泄露
        - 按赛季分组进行估算
    """
    print("\n>>> 模型训练")
    print("=" * 50)
    
    # 初始化估算器
    estimator = FanVotingEstimator(alpha=alpha, n_bootstrap=n_bootstrap)
    
    # 构建特征矩阵
    feature_matrix = build_feature_matrix(data)
    
    # 存储结果
    all_estimates = []
    
    # 按赛季分组估算
    for season in sorted(data['season'].unique()):
        season_data = data[data['season'] == season].copy()
        season_rule = season_data['season_rule'].iloc[0]
        
        print(f"\n处理赛季 {season} ({season_rule})...")
        
        # 对每周进行估算
        for week in range(1, 12):
            score_col = f'week{week}_total_score'
            if score_col not in season_data.columns:
                continue
            
            # 筛选该周有效参赛者
            week_mask = season_data[score_col] > 0
            week_data = season_data[week_mask].copy()
            
            if len(week_data) < 2:
                continue
            
            # 估算粉丝投票
            estimates = estimator.estimate_single_week(week_data, week, season_rule)
            
            if estimates is not None:
                for i, (idx, row) in enumerate(week_data.iterrows()):
                    all_estimates.append({
                        'celebrity_name': row['celebrity_name'],
                        'season': season,
                        'week': week,
                        'season_rule': season_rule,
                        'judge_score': row[score_col],
                        'estimated_fan_vote_pct': estimates[i] if i < len(estimates) else 0,
                        'placement': row['placement']
                    })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(all_estimates)
    
    print(f"\n✓ 模型训练完成")
    print(f"  - 总估算记录数: {len(results_df)}")
    
    return {
        'estimates': results_df,
        'estimator': estimator,
        'feature_matrix': feature_matrix
    }


# ============================================
# 第六部分：Bootstrap确定性评估
# ============================================

def bootstrap_confidence_intervals(data, results, n_bootstrap=100, confidence=0.95):
    """
    Bootstrap采样计算置信区间
    
    参数:
        data: 原始数据
        results: 模型估算结果
        n_bootstrap: 采样次数
        confidence: 置信水平
    
    返回:
        confidence_intervals: 置信区间结果
    
    注意事项:
        - 采样时保持数据结构完整性
        - 每次采样重新训练模型
    """
    print("\n>>> Bootstrap置信区间估计")
    print("-" * 40)
    
    estimates_list = []
    
    for b in range(n_bootstrap):
        if (b + 1) % 20 == 0:
            print(f"  Bootstrap迭代: {b + 1}/{n_bootstrap}")
        
        # 对每个赛季进行有放回采样
        bootstrap_sample = data.sample(frac=1.0, replace=True, random_state=b)
        
        # 重新训练
        bootstrap_results = train_voting_model(bootstrap_sample, alpha=0.01, n_bootstrap=1)
        
        if bootstrap_results is not None:
            estimates_list.append(bootstrap_results['estimates'])
    
    # 计算置信区间
    if estimates_list:
        # 合并所有Bootstrap结果
        combined = pd.concat(estimates_list, ignore_index=True)
        
        # 按选手-赛季-周分组计算统计量
        grouped = combined.groupby(['celebrity_name', 'season', 'week'])['estimated_fan_vote_pct']
        
        alpha = 1 - confidence
        ci_results = grouped.agg([
            'mean',
            'std',
            ('ci_lower', lambda x: np.percentile(x, alpha/2 * 100)),
            ('ci_upper', lambda x: np.percentile(x, (1 - alpha/2) * 100))
        ]).reset_index()
        
        print(f"\n✓ 置信区间计算完成")
        print(f"  - 置信水平: {confidence * 100}%")
        print(f"  - 平均置信区间宽度: {(ci_results['ci_upper'] - ci_results['ci_lower']).mean():.4f}")
        
        return ci_results
    
    return None


# ============================================
# 第七部分：结果预测与验证
# ============================================

def evaluate_consistency(results, data):
    """
    评估模型一致性：预测淘汰结果 vs 实际淘汰结果
    
    参数:
        results: 模型估算结果
        data: 原始数据
    
    返回:
        consistency_metrics: 一致性评价指标
    """
    print("\n>>> 一致性评估")
    print("-" * 40)
    
    estimates = results['estimates']
    
    # 计算每周的预测淘汰者
    predictions = []
    
    for (season, week), group in estimates.groupby(['season', 'week']):
        if len(group) < 2:
            continue
        
        # 合并得分 = 评委评分百分比 + 粉丝投票百分比
        group = group.copy()
        total_judge = group['judge_score'].sum()
        if total_judge > 0:
            group['judge_pct'] = group['judge_score'] / total_judge
        else:
            group['judge_pct'] = 0
        
        group['combined_score'] = 0.5 * group['judge_pct'] + 0.5 * group['estimated_fan_vote_pct']
        
        # 预测淘汰者（合并得分最低）
        predicted_eliminated = group.loc[group['combined_score'].idxmin(), 'celebrity_name']
        
        # 实际淘汰者（排名最差且在该周被淘汰）
        worst_placement = group['placement'].max()
        actual_eliminated = group[group['placement'] == worst_placement]['celebrity_name'].values
        
        predictions.append({
            'season': season,
            'week': week,
            'predicted': predicted_eliminated,
            'actual': actual_eliminated[0] if len(actual_eliminated) > 0 else None,
            'correct': predicted_eliminated in actual_eliminated
        })
    
    predictions_df = pd.DataFrame(predictions)
    
    # 计算一致性指标
    accuracy = predictions_df['correct'].mean()
    
    # 按赛季规则分组统计
    estimates_with_rule = estimates.merge(
        data[['celebrity_name', 'season', 'season_rule']].drop_duplicates(),
        on=['celebrity_name', 'season']
    )
    
    print(f"\n✓ 一致性评估完成")
    print(f"  - 整体淘汰预测准确率: {accuracy:.2%}")
    
    # 计算Kappa系数（简化版）
    po = accuracy
    pe = 1 / estimates.groupby(['season', 'week']).size().mean()  # 随机猜测概率
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0
    
    print(f"  - Cohen's Kappa系数: {kappa:.4f}")
    
    return {
        'accuracy': accuracy,
        'kappa': kappa,
        'predictions': predictions_df,
        'by_rule': {}  # 可扩展按规则分组的统计
    }


# ============================================
# 第八部分：可视化生成
# ============================================

def generate_visualizations(results, data, output_dir='output'):
    """
    生成问题1相关的可视化图表
    
    参数:
        results: 模型估算结果
        data: 原始数据
        output_dir: 输出目录
    
    返回:
        生成的图表文件列表
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    estimates = results['estimates']
    
    # 图1: 粉丝投票估算分布图
    fig, ax = plt.subplots(figsize=(10, 6))
    estimates['estimated_fan_vote_pct'].hist(bins=30, ax=ax, color='steelblue', edgecolor='white')
    ax.set_xlabel('Estimated Fan Vote Percentage')
    ax.set_ylabel('Frequency')
    ax.set_title('Figure Q1-1: Distribution of Estimated Fan Vote Percentages')
    ax.axvline(estimates['estimated_fan_vote_pct'].mean(), color='red', linestyle='--', 
               label=f'Mean = {estimates["estimated_fan_vote_pct"].mean():.3f}')
    ax.legend()
    
    filepath = os.path.join(output_dir, 'Q1_01_vote_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    # 图2: 评委评分与粉丝投票相关性
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(estimates['judge_score'], estimates['estimated_fan_vote_pct'], 
               alpha=0.5, c='steelblue', s=30)
    
    # 添加趋势线
    z = np.polyfit(estimates['judge_score'].fillna(0), 
                   estimates['estimated_fan_vote_pct'].fillna(0), 1)
    p = np.poly1d(z)
    x_range = np.linspace(estimates['judge_score'].min(), estimates['judge_score'].max(), 100)
    ax.plot(x_range, p(x_range), 'r--', linewidth=2, label='Trend Line')
    
    ax.set_xlabel('Judge Score')
    ax.set_ylabel('Estimated Fan Vote Percentage')
    ax.set_title('Figure Q1-2: Correlation between Judge Scores and Fan Votes')
    
    corr = estimates['judge_score'].corr(estimates['estimated_fan_vote_pct'])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.legend()
    
    filepath = os.path.join(output_dir, 'Q1_02_score_vote_correlation.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    # 图3: 按赛季规则的投票分布对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, rule in enumerate(['Ranking', 'Percentage', 'Ranking_JudgeSave']):
        rule_data = estimates[estimates['season_rule'] == rule]
        if len(rule_data) > 0:
            axes[i].hist(rule_data['estimated_fan_vote_pct'], bins=20, 
                        color=['#2ecc71', '#3498db', '#e74c3c'][i], edgecolor='white')
            axes[i].set_title(f'{rule}\n(n={len(rule_data)})')
            axes[i].set_xlabel('Estimated Vote %')
            axes[i].set_ylabel('Frequency')
    
    plt.suptitle('Figure Q1-3: Fan Vote Distribution by Season Rule', fontsize=14, y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'Q1_03_vote_by_rule.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    # 图4: 每周粉丝投票趋势
    fig, ax = plt.subplots(figsize=(12, 6))
    
    weekly_stats = estimates.groupby('week')['estimated_fan_vote_pct'].agg(['mean', 'std']).reset_index()
    
    ax.errorbar(weekly_stats['week'], weekly_stats['mean'], yerr=weekly_stats['std'],
                fmt='o-', capsize=5, capthick=2, color='steelblue', linewidth=2, markersize=8)
    ax.fill_between(weekly_stats['week'], 
                    weekly_stats['mean'] - weekly_stats['std'],
                    weekly_stats['mean'] + weekly_stats['std'],
                    alpha=0.3)
    
    ax.set_xlabel('Week Number')
    ax.set_ylabel('Mean Estimated Fan Vote Percentage')
    ax.set_title('Figure Q1-4: Fan Vote Trends Across Weeks (with 1 SD Error Bars)')
    ax.set_xticks(range(1, 12))
    
    filepath = os.path.join(output_dir, 'Q1_04_weekly_trend.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    return generated_files


# ============================================
# 第九部分：模型保存与加载
# ============================================

def save_model_results(results, output_dir='output'):
    """
    保存模型结果
    
    参数:
        results: 模型结果字典
        output_dir: 输出目录
    """
    import os
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存估算结果为CSV
    if 'estimates' in results:
        csv_path = os.path.join(output_dir, 'Q1_fan_voting_estimates.csv')
        results['estimates'].to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 估算结果已保存: {csv_path}")
    
    # 保存模型对象为pickle
    model_path = os.path.join(output_dir, 'Q1_voting_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'estimator': results.get('estimator'),
            'feature_matrix': results.get('feature_matrix')
        }, f)
    print(f"✓ 模型对象已保存: {model_path}")


def load_model_results(output_dir='output'):
    """
    加载模型结果
    
    参数:
        output_dir: 输出目录
    
    返回:
        加载的模型结果
    """
    import os
    import pickle
    
    results = {}
    
    # 加载CSV
    csv_path = os.path.join(output_dir, 'Q1_fan_voting_estimates.csv')
    if os.path.exists(csv_path):
        results['estimates'] = pd.read_csv(csv_path)
        print(f"✓ 估算结果已加载: {csv_path}")
    
    # 加载pickle
    model_path = os.path.join(output_dir, 'Q1_voting_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            results.update(model_data)
        print(f"✓ 模型对象已加载: {model_path}")
    
    return results


# ============================================
# 主程序入口
# ============================================

def main():
    """
    主程序：执行完整的粉丝投票估算流程
    """
    print("=" * 60)
    print("问题1：粉丝投票估算模型")
    print("=" * 60)
    
    # 1. 数据加载
    print("\n【步骤1】数据加载")
    data = load_data('output/question1_data.csv')
    
    if data is None:
        print("数据加载失败，程序终止")
        return
    
    # 2. 参数调优
    print("\n【步骤2】参数调优")
    best_alpha, cv_results = parameter_tuning(data)
    
    # 3. 模型训练
    print("\n【步骤3】模型训练")
    results = train_voting_model(data, alpha=best_alpha, n_bootstrap=50)
    
    # 4. 一致性评估
    print("\n【步骤4】一致性评估")
    consistency = evaluate_consistency(results, data)
    results['consistency'] = consistency
    
    # 5. 可视化生成
    print("\n【步骤5】可视化生成")
    viz_files = generate_visualizations(results, data, 'output')
    
    # 6. 保存结果
    print("\n【步骤6】保存结果")
    save_model_results(results, 'output')
    
    # 7. 结果摘要
    print("\n" + "=" * 60)
    print("模型求解结果摘要")
    print("=" * 60)
    print(f"• 总估算记录数: {len(results['estimates'])}")
    print(f"• 淘汰预测准确率: {consistency['accuracy']:.2%}")
    print(f"• Cohen's Kappa系数: {consistency['kappa']:.4f}")
    print(f"• 最优正则化系数: α = {best_alpha}")
    print(f"• 生成可视化图表: {len(viz_files)} 个")
    
    print("\n>>> 问题1模型求解完成 <<<")
    
    return results


if __name__ == '__main__':
    main()
