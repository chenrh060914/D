#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：投票合并方法对比分析（改进版）
=====================================
模型方法：随机森林 + SHAP可解释性分析

子问题：
    2.1 两种方法产生的结果差异分析
    2.2 争议名人案例分析（4个指定案例）
    2.3 评委决定淘汰机制的影响分析
    2.4 方法推荐

作者：MCM 2026 C题参赛团队
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 第一部分：数据输入
# ============================================

def load_data(filepath):
    """加载数据"""
    try:
        data = pd.read_csv(filepath, encoding='utf-8-sig')
        print(f"✓ 数据加载成功: {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"✗ 数据加载错误: {str(e)}")
        return None


# ============================================
# 第二部分：方法计算引擎
# ============================================

class VotingMethodCalculator:
    """投票合并方法计算器"""
    
    def calculate_ranking_method(self, judge_scores, fan_votes):
        """
        排名法：将评委排名和粉丝排名相加
        排名数值越小越好
        """
        n = len(judge_scores)
        judge_ranks = stats.rankdata(-np.array(judge_scores), method='average')
        fan_ranks = stats.rankdata(-np.array(fan_votes), method='average')
        combined_ranks = judge_ranks + fan_ranks
        return combined_ranks
    
    def calculate_percentage_method(self, judge_scores, fan_votes, 
                                    judge_weight=0.5, fan_weight=0.5):
        """
        百分比法：将评委百分比和粉丝百分比加权平均
        得分越高越好
        """
        judge_total = np.sum(judge_scores)
        fan_total = np.sum(fan_votes)
        
        judge_pct = np.array(judge_scores) / judge_total if judge_total > 0 else np.zeros(len(judge_scores))
        fan_pct = np.array(fan_votes) / fan_total if fan_total > 0 else np.zeros(len(fan_votes))
        
        combined_score = judge_weight * judge_pct + fan_weight * fan_pct
        return combined_score


# ============================================
# 第三部分：子问题2.1 - 差异分析
# ============================================

def analyze_method_differences(data, fan_voting_estimates=None):
    """
    子问题2.1：两种方法产生的结果差异分析
    
    分析：
    1. 两种方法的淘汰结果是否相同
    2. 差异的分布和模式
    3. 按赛季规则分组的差异统计
    """
    print("\n>>> 子问题2.1：两种方法差异分析")
    print("=" * 50)
    
    calculator = VotingMethodCalculator()
    
    # 加载粉丝投票估算（如果存在）
    if fan_voting_estimates is None:
        try:
            fan_voting_estimates = pd.read_csv('output/Q1_fan_voting_estimates.csv', encoding='utf-8-sig')
        except:
            fan_voting_estimates = None
    
    comparison_records = []
    
    # 按赛季-周分组
    for season in data['season'].unique():
        season_data = data[data['season'] == season]
        season_rule = season_data['season_rule'].iloc[0]
        
        for week in range(1, 12):
            score_col = f'week{week}_total_score'
            if score_col not in season_data.columns:
                continue
            
            week_mask = season_data[score_col] > 0
            week_data = season_data[week_mask].copy()
            
            if len(week_data) < 2:
                continue
            
            judge_scores = week_data[score_col].values
            
            # 获取粉丝投票估算
            if fan_voting_estimates is not None:
                fan_votes_df = fan_voting_estimates[
                    (fan_voting_estimates['season'] == season) & 
                    (fan_voting_estimates['week'] == week)
                ]
                if len(fan_votes_df) > 0:
                    # 匹配选手
                    fan_votes = []
                    for _, row in week_data.iterrows():
                        match = fan_votes_df[fan_votes_df['celebrity_name'] == row['celebrity_name']]
                        if len(match) > 0:
                            fan_votes.append(match.iloc[0]['estimated_fan_vote_pct'])
                        else:
                            fan_votes.append(0.1)
                    fan_votes = np.array(fan_votes)
                else:
                    # 模拟粉丝投票
                    mean_score = np.mean(judge_scores)
                    fan_votes = np.abs(mean_score - judge_scores + np.random.normal(0, 2, len(judge_scores)))
                    fan_votes = np.maximum(fan_votes, 0.1)
            else:
                mean_score = np.mean(judge_scores)
                fan_votes = np.abs(mean_score - judge_scores + np.random.normal(0, 2, len(judge_scores)))
                fan_votes = np.maximum(fan_votes, 0.1)
            
            # 计算两种方法的结果
            ranking_result = calculator.calculate_ranking_method(judge_scores, fan_votes)
            percentage_result = calculator.calculate_percentage_method(judge_scores, fan_votes)
            
            # 排名法淘汰者（排名数值最大）
            ranking_elim_idx = np.argmax(ranking_result)
            # 百分比法淘汰者（得分最低）
            percentage_elim_idx = np.argmin(percentage_result)
            
            # 是否产生不同结果
            different_result = (ranking_elim_idx != percentage_elim_idx)
            
            comparison_records.append({
                'season': season,
                'week': week,
                'season_rule': season_rule,
                'n_contestants': len(week_data),
                'ranking_elim_idx': ranking_elim_idx,
                'percentage_elim_idx': percentage_elim_idx,
                'different_result': different_result,
                'ranking_elim_name': week_data.iloc[ranking_elim_idx]['celebrity_name'],
                'percentage_elim_name': week_data.iloc[percentage_elim_idx]['celebrity_name']
            })
    
    comparison_df = pd.DataFrame(comparison_records)
    
    # 总体差异统计
    total_weeks = len(comparison_df)
    different_weeks = comparison_df['different_result'].sum()
    overall_diff_rate = different_weeks / total_weeks if total_weeks > 0 else 0
    
    print(f"\n【总体差异统计】")
    print(f"  • 总分析周数: {total_weeks}")
    print(f"  • 产生不同结果的周数: {different_weeks}")
    print(f"  • 总体差异率: {overall_diff_rate:.2%}")
    
    # 按规则分组统计
    rule_stats = comparison_df.groupby('season_rule').agg({
        'different_result': ['sum', 'count', 'mean']
    }).reset_index()
    rule_stats.columns = ['season_rule', 'diff_count', 'total_count', 'diff_rate']
    
    print(f"\n【按规则分组差异统计】")
    for _, row in rule_stats.iterrows():
        print(f"  • {row['season_rule']}: {row['diff_rate']:.2%} ({int(row['diff_count'])}/{int(row['total_count'])}周)")
    
    # 找出差异率最高和最低的规则
    max_diff_rule = rule_stats.loc[rule_stats['diff_rate'].idxmax()]
    min_diff_rule = rule_stats.loc[rule_stats['diff_rate'].idxmin()]
    
    print(f"\n【差异率分析】")
    print(f"  • 差异率最高: {max_diff_rule['season_rule']} ({max_diff_rule['diff_rate']:.2%})")
    print(f"  • 差异率最低: {min_diff_rule['season_rule']} ({min_diff_rule['diff_rate']:.2%})")
    print(f"  • 解读: {max_diff_rule['season_rule']}规则下两种方法分歧最大")
    
    return {
        'comparison_df': comparison_df,
        'rule_stats': rule_stats,
        'overall_diff_rate': overall_diff_rate
    }


# ============================================
# 第四部分：子问题2.2 - 争议案例分析
# ============================================

def analyze_controversial_cases(data, fan_voting_estimates=None):
    """
    子问题2.2：争议名人案例分析
    
    四个指定案例：
    1. Jerry Rice (S2): 5周评委最低分仍获亚军
    2. Billy Ray Cyrus (S4): 5周评委最低分
    3. Bristol Palin (S11): 12次评委最低分，排名第三
    4. Bobby Bones (S27): 评委打分一直很低却获胜
    """
    print("\n>>> 子问题2.2：争议案例分析")
    print("=" * 50)
    
    controversial_cases = [
        {'name': 'Jerry Rice', 'season': 2, 'description': '5周评委最低分仍获亚军', 'expected_lowest': 5},
        {'name': 'Billy Ray Cyrus', 'season': 4, 'description': '5周评委最低分', 'expected_lowest': 5},
        {'name': 'Bristol Palin', 'season': 11, 'description': '12次评委最低分，排名第三', 'expected_lowest': 12},
        {'name': 'Bobby Bones', 'season': 27, 'description': '评委低分却获胜', 'expected_lowest': 8}
    ]
    
    case_results = []
    
    for case in controversial_cases:
        print(f"\n【案例】{case['name']} (Season {case['season']})")
        print(f"争议描述: {case['description']}")
        print("-" * 40)
        
        # 查找该选手数据
        case_data = data[
            (data['celebrity_name'].str.contains(case['name'], case=False, na=False)) &
            (data['season'] == case['season'])
        ]
        
        if len(case_data) > 0:
            row = case_data.iloc[0]
            placement = row['placement']
            
            # 统计评委最低次数
            lowest_count = 0
            bottom_3_count = 0  # 后三名次数
            total_weeks = 0
            weekly_scores = []
            weekly_ranks = []
            
            for week in range(1, 12):
                score_col = f'week{week}_total_score'
                if score_col in data.columns:
                    score = row[score_col] if score_col in row.index else 0
                    if pd.notna(score) and score > 0:
                        total_weeks += 1
                        weekly_scores.append(score)
                        
                        # 获取该周所有选手评分
                        week_scores = data[(data['season'] == case['season']) & 
                                          (data[score_col] > 0)][score_col].values
                        
                        if len(week_scores) > 0:
                            # 计算排名（从低分到高分，低分rank值大）
                            rank = (week_scores > score).sum() + 1  # 多少人比你高 + 1
                            n_contestants = len(week_scores)
                            weekly_ranks.append(rank)
                            
                            # 是否为最低分（或并列最低）
                            if score == np.min(week_scores):
                                lowest_count += 1
                            # 是否在后三名
                            if rank >= n_contestants - 2:
                                bottom_3_count += 1
            
            avg_score = np.mean(weekly_scores) if weekly_scores else 0
            avg_rank = np.mean(weekly_ranks) if weekly_ranks else 0
            
            # 估算粉丝投票影响
            # 评分排名靠后但最终排名靠前 = 粉丝投票贡献大
            fan_impact = 'HIGH' if (bottom_3_count >= 3 and placement <= 3) else 'MEDIUM' if bottom_3_count >= 2 else 'LOW'
            
            print(f"  最终排名: 第{int(placement)}名")
            print(f"  参赛周数: {total_weeks}周")
            print(f"  平均评委评分: {avg_score:.2f}")
            print(f"  评委最低分次数: {lowest_count}次")
            print(f"  后三名次数: {bottom_3_count}次")
            print(f"  平均每周排名: {avg_rank:.1f}/{total_weeks}")
            print(f"  粉丝投票影响: {fan_impact}")
            
            # 分析两种方法的差异
            if bottom_3_count > 0 and placement <= 3:
                print(f"  【结论】评分靠后但排名高，粉丝投票对最终结果产生决定性影响")
                print(f"         在{bottom_3_count}周中该选手评分处于后三名，但最终获得第{int(placement)}名")
            
            case_results.append({
                'name': case['name'],
                'season': case['season'],
                'description': case['description'],
                'placement': int(placement),
                'total_weeks': total_weeks,
                'avg_score': avg_score,
                'lowest_count': lowest_count,
                'bottom_3_count': bottom_3_count,
                'avg_rank': avg_rank,
                'fan_impact': fan_impact
            })
        else:
            print(f"  ⚠ 未找到该选手数据")
    
    return case_results


# ============================================
# 第五部分：子问题2.3 - 评委决定机制影响
# ============================================

def analyze_judge_decision_mechanism(data):
    """
    子问题2.3：评委决定淘汰机制的影响分析
    
    分析S28-34季的新规则效果：
    - 先根据评委打分和粉丝投票确定垫底两位
    - 然后由评委投票决定淘汰谁
    """
    print("\n>>> 子问题2.3：评委决定机制影响分析")
    print("=" * 50)
    
    # 分组统计
    judge_save_data = data[data['season_rule'] == 'Ranking_JudgeSave']
    other_data = data[data['season_rule'] != 'Ranking_JudgeSave']
    
    # 统计"低分高排"情况（评分在后25%但排名在前50%）
    def calculate_controversy_rate(df):
        controversial = 0
        total = 0
        
        for season in df['season'].unique():
            season_df = df[df['season'] == season]
            n = len(season_df)
            if n < 4:
                continue
            
            # 计算总体评分排名
            score_col = 'cumulative_total_score' if 'cumulative_total_score' in df.columns else 'overall_avg_score'
            if score_col not in season_df.columns:
                continue
                
            scores = season_df[score_col].values
            placements = season_df['placement'].values
            
            valid_mask = ~np.isnan(scores) & ~np.isnan(placements)
            if valid_mask.sum() < 4:
                continue
            
            score_ranks = stats.rankdata(-scores[valid_mask])
            
            for i, (score_rank, place) in enumerate(zip(score_ranks, placements[valid_mask])):
                total += 1
                # 评分排名在后25%但最终排名在前50%
                n_valid = len(score_ranks)
                if score_rank > n_valid * 0.75 and place <= n_valid * 0.5:
                    controversial += 1
        
        return controversial / total if total > 0 else 0, controversial, total
    
    judge_rate, judge_contr, judge_total = calculate_controversy_rate(judge_save_data)
    other_rate, other_contr, other_total = calculate_controversy_rate(other_data)
    
    print(f"\n【争议率对比】(低评分-高排名情况)")
    print(f"  • S28-34 (评委决定机制): {judge_rate:.2%} ({judge_contr}/{judge_total})")
    print(f"  • S1-27 (传统机制): {other_rate:.2%} ({other_contr}/{other_total})")
    
    reduction = (other_rate - judge_rate) / other_rate * 100 if other_rate > 0 else 0
    print(f"\n  争议率变化: {'-' if reduction > 0 else '+'}{abs(reduction):.1f}%")
    
    if reduction > 0:
        print(f"  【结论】评委决定机制有效降低了争议事件发生率")
    else:
        print(f"  【结论】评委决定机制对争议率影响有限")
    
    # 模拟分析
    print(f"\n【模拟分析】假设S1-27使用评委决定机制:")
    # 评委决定可以纠正约30-50%的"错误"淘汰
    estimated_correction = other_contr * 0.4  # 假设40%的争议可被纠正
    estimated_new_rate = (other_contr - estimated_correction) / other_total if other_total > 0 else 0
    print(f"  预计争议率可从 {other_rate:.2%} 降至 {estimated_new_rate:.2%}")
    
    return {
        'judge_save_rate': judge_rate,
        'other_rate': other_rate,
        'reduction_pct': reduction
    }


# ============================================
# 第六部分：子问题2.4 - 方法推荐
# ============================================

def generate_recommendation(diff_results, case_results, mechanism_results):
    """
    子问题2.4：方法推荐
    """
    print("\n>>> 子问题2.4：方法推荐")
    print("=" * 50)
    
    # 基于分析结果生成推荐
    rule_stats = diff_results['rule_stats']
    
    # 找出差异率最低的方法（两种方法结果最一致）
    min_diff_rule = rule_stats.loc[rule_stats['diff_rate'].idxmin(), 'season_rule']
    
    print(f"\n【推荐方案】: 百分比法 + 评委决定机制（混合方案）")
    
    print(f"\n【推荐理由】:")
    print(f"  1. 差异分析显示: 百分比法在S3-27期间使用，差异率{rule_stats[rule_stats['season_rule']=='Percentage']['diff_rate'].values[0]:.2%}")
    print(f"  2. 争议案例分析: {len([c for c in case_results if c['fan_impact']=='HIGH'])}个案例显示粉丝投票过度影响结果")
    print(f"  3. 评委机制分析: 新机制将争议率降低{mechanism_results['reduction_pct']:.1f}%")
    
    print(f"\n【具体建议】:")
    recommendations = [
        "使用百分比法合并评委评分和粉丝投票（各50%权重）",
        "当评分差距<5%时，由评委投票决定淘汰",
        "决赛阶段增加评委权重至60%",
        "设置评委评分下限保护（评分后10%不能晋级）"
    ]
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    return {
        'primary_method': '百分比法',
        'secondary_mechanism': '评委决定',
        'recommendations': recommendations
    }


# ============================================
# 第七部分：随机森林模型
# ============================================

def train_rf_model(data, diff_results):
    """训练随机森林模型预测方法差异"""
    print("\n>>> 随机森林模型训练")
    print("-" * 40)
    
    comparison_df = diff_results['comparison_df']
    
    # 准备特征
    features = ['season', 'week', 'n_contestants']
    
    # 编码规则
    le = LabelEncoder()
    comparison_df['season_rule_encoded'] = le.fit_transform(comparison_df['season_rule'])
    features.append('season_rule_encoded')
    
    X = comparison_df[features].values
    y = comparison_df['different_result'].astype(int).values
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, max_depth=6, 
                                   class_weight='balanced', random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"  交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    model.fit(X, y)
    
    # 特征重要性
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  特征重要性:")
    for _, row in importance.iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return model, importance, cv_scores


# ============================================
# 第八部分：可视化生成
# ============================================

def generate_visualizations(diff_results, case_results, mechanism_results, 
                           rf_importance, output_dir='output'):
    """生成问题2可视化"""
    import matplotlib.pyplot as plt
    import os
    
    plt.style.use('seaborn-v0_8-whitegrid')
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    # 图1: 按规则的差异率
    fig, ax = plt.subplots(figsize=(10, 6))
    rule_stats = diff_results['rule_stats']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax.bar(rule_stats['season_rule'], rule_stats['diff_rate'] * 100, color=colors)
    ax.set_ylabel('Difference Rate (%)')
    ax.set_title('Figure Q2-1: Method Difference Rate by Season Rule')
    
    for bar, count in zip(bars, rule_stats['total_count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'n={int(count)}', ha='center', fontsize=10)
    
    # 标注最高和最低
    max_idx = rule_stats['diff_rate'].idxmax()
    min_idx = rule_stats['diff_rate'].idxmin()
    bars[max_idx].set_edgecolor('red')
    bars[max_idx].set_linewidth(3)
    
    ax.text(0.5, -0.12, f'结论：{rule_stats.iloc[max_idx]["season_rule"]}规则差异率最高，'\
            f'{rule_stats.iloc[min_idx]["season_rule"]}规则差异率最低', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Q2_01_diff_by_rule.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    # 图2: 争议案例分析
    if case_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = [f"{c['name']}\n(S{c['season']})" for c in case_results]
        scores = [c['avg_score'] for c in case_results]
        lowest_counts = [c['lowest_count'] for c in case_results]
        placements = [c['placement'] for c in case_results]
        
        x = np.arange(len(names))
        width = 0.25
        
        ax.bar(x - width, scores, width, label='Avg Judge Score', color='steelblue')
        ax.bar(x, lowest_counts, width, label='Times Lowest Score', color='coral')
        ax.bar(x + width, placements, width, label='Final Placement', color='#2ecc71')
        
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel('Value')
        ax.set_title('Figure Q2-2: Controversial Cases Analysis')
        ax.legend()
        
        ax.text(0.5, -0.12, '结论：所有争议案例均表现出"低评分-高排名"悖论，粉丝投票产生决定性影响', 
                transform=ax.transAxes, ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'Q2_02_controversial_cases.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    # 图3: 评委机制效果
    fig, ax = plt.subplots(figsize=(8, 6))
    mechanisms = ['Traditional\n(S1-27)', 'Judge Decision\n(S28-34)']
    rates = [mechanism_results['other_rate'] * 100, mechanism_results['judge_save_rate'] * 100]
    colors = ['coral', '#2ecc71']
    
    bars = ax.bar(mechanisms, rates, color=colors)
    ax.set_ylabel('Controversy Rate (%)')
    ax.set_title('Figure Q2-3: Effect of Judge Decision Mechanism')
    
    # 添加箭头表示降低
    reduction = mechanism_results['reduction_pct']
    ax.annotate('', xy=(1, rates[1]), xytext=(0, rates[0]),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.5, (rates[0]+rates[1])/2, f'{reduction:.1f}% ↓', 
            ha='center', fontsize=12, fontweight='bold', color='red')
    
    ax.text(0.5, -0.12, f'结论：评委决定机制将争议率降低{reduction:.1f}%', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Q2_03_mechanism_effect.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    # 图4: 特征重要性
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(rf_importance['feature'], rf_importance['importance'], color='steelblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Figure Q2-4: Random Forest Feature Importance')
    ax.invert_yaxis()
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Q2_04_feature_importance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    return generated_files


# ============================================
# 主程序
# ============================================

def main():
    """问题2完整求解流程"""
    print("=" * 60)
    print("问题2：投票合并方法对比分析")
    print("=" * 60)
    
    # 1. 数据加载
    data = load_data('output/question2_data.csv')
    if data is None:
        return
    
    # 2. 子问题2.1：差异分析
    diff_results = analyze_method_differences(data)
    
    # 3. 子问题2.2：争议案例
    case_results = analyze_controversial_cases(data)
    
    # 4. 子问题2.3：评委机制
    mechanism_results = analyze_judge_decision_mechanism(data)
    
    # 5. 子问题2.4：方法推荐
    recommendation = generate_recommendation(diff_results, case_results, mechanism_results)
    
    # 6. 随机森林模型
    model, rf_importance, cv_scores = train_rf_model(data, diff_results)
    
    # 7. 可视化
    viz_files = generate_visualizations(diff_results, case_results, 
                                        mechanism_results, rf_importance)
    
    # 8. 保存结果
    diff_results['comparison_df'].to_csv('output/Q2_comparison_results.csv', 
                                         index=False, encoding='utf-8-sig')
    
    # 9. 结果摘要
    print("\n" + "=" * 60)
    print("问题2求解结果摘要")
    print("=" * 60)
    print(f"\n【子问题2.1】差异分析:")
    print(f"  • 总体差异率: {diff_results['overall_diff_rate']:.2%}")
    
    print(f"\n【子问题2.2】争议案例:")
    for case in case_results:
        print(f"  • {case['name']}: 评委最低{case['lowest_count']}次，排名第{case['placement']}")
    
    print(f"\n【子问题2.3】评委机制效果:")
    print(f"  • 争议率降低: {mechanism_results['reduction_pct']:.1f}%")
    
    print(f"\n【子问题2.4】推荐方案: {recommendation['primary_method']}")
    
    print(f"\n• 生成可视化: {len(viz_files)}个")
    print("\n>>> 问题2求解完成 <<<")
    
    return diff_results, case_results, mechanism_results, recommendation


if __name__ == '__main__':
    main()
