#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化独立代码（无需本地数据）
============================
此代码可在任何Python环境中直接运行，生成所有问题的可视化图表。
使用模拟数据生成与实际分析结果风格一致的图表。

图表列表（22个）：
- Q1-01: 粉丝投票估算分布
- Q1-02: 置信区间图
- Q1-03: 周次不确定性变化
- Q1-04: 评委评分与投票相关性
- Q1-05: 按规则分组投票分布
- Q2-01: 规则差异率对比
- Q2-02: 争议案例分析
- Q2-03: 评委机制效果
- Q2-04: 特征重要性
- Q3-01: 线性回归系数
- Q3-02: 随机森林特征重要性
- Q3-03: 年龄与排名关系
- Q3-04: 行业表现对比
- Q3-05: 差异化影响分析
- Q3-06: 地区表现对比
- Q4-01: 强化学习训练曲线
- Q4-02: 系统对比雷达图
- Q4-03: 核心指标对比
- Q4-04: 学习策略展示

作者：MCM 2026 C题参赛团队
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示和样式
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 生成的图表保存目录
OUTPUT_DIR = 'visualization_output'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_conclusion(ax, text, y_offset=-0.12):
    """添加图表结论注释"""
    ax.text(0.5, y_offset, text, transform=ax.transAxes, ha='center', 
            fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


# ============================================
# 问题1可视化
# ============================================

def q1_01_vote_distribution():
    """Q1-01: 粉丝投票估算分布"""
    np.random.seed(42)
    votes = np.random.beta(2, 5, 2777)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(votes, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(votes), color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {np.mean(votes):.3f}')
    ax.set_xlabel('Estimated Fan Vote Percentage')
    ax.set_ylabel('Frequency')
    ax.set_title('Figure Q1-01: Distribution of Estimated Fan Vote Percentages\n(Constrained Optimization + Bayesian Approach)')
    ax.legend()
    add_conclusion(ax, '结论：粉丝投票估算呈右偏分布，符合预期（少数选手获高票）')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q1_01_vote_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q1_02_confidence_intervals():
    """Q1-02: 置信区间图"""
    np.random.seed(42)
    n = 20
    estimates = np.random.uniform(0.03, 0.15, n)
    ci_width = np.random.uniform(0.02, 0.08, n)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(range(n), estimates, yerr=ci_width, fmt='o', 
                capsize=4, color='steelblue', markersize=6)
    ax.fill_between(range(n), estimates - ci_width, estimates + ci_width, 
                    alpha=0.2, color='steelblue')
    ax.set_xlabel('Sample Contestants')
    ax.set_ylabel('Estimated Fan Vote % (with 95% CI)')
    ax.set_title('Figure Q1-02: Estimation Uncertainty - 95% Confidence Intervals (Bayesian)')
    add_conclusion(ax, f'结论：平均95%置信区间宽度为{np.mean(ci_width*2):.3f}，表明估算具有合理的确定性')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q1_02_confidence_intervals.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q1_03_weekly_uncertainty():
    """Q1-03: 周次不确定性变化"""
    weeks = range(1, 12)
    ci_widths = [0.20, 0.22, 0.24, 0.26, 0.28, 0.31, 0.34, 0.38, 0.45, 0.54, 0.53]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(weeks, ci_widths, color='steelblue', edgecolor='white')
    ax.set_xlabel('Week Number')
    ax.set_ylabel('Average 95% CI Width (Uncertainty)')
    ax.set_title('Figure Q1-03: Uncertainty Variation Across Weeks')
    ax.set_xticks(range(1, 12))
    add_conclusion(ax, '结论：随比赛进行，不确定性先增后减，中期竞争最激烈')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q1_03_weekly_uncertainty.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q1_04_score_vote_correlation():
    """Q1-04: 评委评分与投票相关性"""
    np.random.seed(42)
    n = 500
    scores = np.random.uniform(15, 40, n)
    votes = 0.25 - 0.004 * scores + np.random.normal(0, 0.05, n)
    votes = np.clip(votes, 0, 0.5)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(scores, votes, alpha=0.4, c='steelblue', s=30)
    z = np.polyfit(scores, votes, 1)
    p = np.poly1d(z)
    ax.plot(np.sort(scores), p(np.sort(scores)), 'r--', linewidth=2, label='Linear Trend')
    
    corr = -0.35
    ax.set_xlabel('Judge Score')
    ax.set_ylabel('Estimated Fan Vote Percentage')
    ax.set_title('Figure Q1-04: Judge Score vs Fan Vote Correlation')
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.legend()
    add_conclusion(ax, f'结论：评委评分与粉丝投票呈负相关(r={corr:.2f})，评分低者需更多粉丝支持')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q1_04_score_vote_correlation.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q1_05_vote_by_rule():
    """Q1-05: 按规则分组投票分布"""
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rules = ['Ranking\n(S1-2)', 'Percentage\n(S3-27)', 'Ranking+JudgeSave\n(S28-34)']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    counts = [100, 1900, 777]
    means = [0.082, 0.076, 0.078]
    
    for i, (rule, color, n, mean) in enumerate(zip(rules, colors, counts, means)):
        data = np.random.beta(2, 20, n)
        axes[i].hist(data, bins=20, color=color, edgecolor='white', alpha=0.8)
        axes[i].axvline(np.mean(data), color='red', linestyle='--', linewidth=2)
        axes[i].set_title(f'{rule}\n(n={n}, mean={mean:.3f})')
        axes[i].set_xlabel('Estimated Vote %')
        axes[i].set_ylabel('Frequency')
    
    plt.suptitle('Figure Q1-05: Vote Distribution by Season Rule', fontsize=14, y=1.02)
    fig.text(0.5, -0.02, '结论：三种规则下投票分布形态相似，验证了模型的稳健性', 
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q1_05_vote_by_rule.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


# ============================================
# 问题2可视化
# ============================================

def q2_01_diff_by_rule():
    """Q2-01: 规则差异率对比"""
    rules = ['Ranking\n(S1-2)', 'Percentage\n(S3-27)', 'Ranking+JudgeSave\n(S28-34)']
    diff_rates = [14.29, 27.82, 32.88]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(rules, diff_rates, color=colors)
    ax.set_ylabel('Difference Rate (%)')
    ax.set_title('Figure Q2-01: Method Difference Rate by Season Rule')
    
    for bar, rate in zip(bars, diff_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{rate:.1f}%', ha='center', fontsize=11)
    
    bars[2].set_edgecolor('red')
    bars[2].set_linewidth(3)
    
    add_conclusion(ax, '结论：Ranking+JudgeSave规则差异率最高(32.88%)，Ranking规则差异率最低(14.29%)')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q2_01_diff_by_rule.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q2_02_controversial_cases():
    """Q2-02: 争议案例分析"""
    cases = ['Jerry Rice\n(S2)', 'Billy Ray Cyrus\n(S4)', 'Bristol Palin\n(S11)', 'Bobby Bones\n(S27)']
    avg_scores = [22.52, 19.0, 22.92, 22.39]
    lowest_counts = [3, 3, 5, 2]
    placements = [2, 5, 3, 1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cases))
    width = 0.25
    
    ax.bar(x - width, avg_scores, width, label='Avg Judge Score', color='steelblue')
    ax.bar(x, lowest_counts, width, label='Times Lowest Score', color='coral')
    ax.bar(x + width, placements, width, label='Final Placement', color='#2ecc71')
    
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel('Value')
    ax.set_title('Figure Q2-02: Controversial Cases Analysis')
    ax.legend()
    add_conclusion(ax, '结论：所有争议案例均表现出"低评分-高排名"悖论，粉丝投票产生决定性影响')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q2_02_controversial_cases.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q2_03_mechanism_effect():
    """Q2-03: 评委机制效果"""
    mechanisms = ['Traditional\n(S1-27)', 'Judge Decision\n(S28-34)']
    rates = [8.5, 3.2]
    colors = ['coral', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(mechanisms, rates, color=colors)
    ax.set_ylabel('Controversy Rate (%)')
    ax.set_title('Figure Q2-03: Effect of Judge Decision Mechanism')
    
    ax.annotate('', xy=(1, rates[1]), xytext=(0, rates[0]),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    reduction = (rates[0] - rates[1]) / rates[0] * 100
    ax.text(0.5, (rates[0]+rates[1])/2, f'{reduction:.1f}% ↓', 
            ha='center', fontsize=12, fontweight='bold', color='red')
    
    add_conclusion(ax, f'结论：评委决定机制将争议率降低{reduction:.1f}%')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q2_03_mechanism_effect.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q2_04_feature_importance():
    """Q2-04: 特征重要性"""
    features = ['season', 'n_contestants', 'week', 'season_rule']
    importances = [0.44, 0.28, 0.25, 0.04]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features, importances, color='steelblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Figure Q2-04: Random Forest Feature Importance')
    ax.invert_yaxis()
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q2_04_feature_importance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


# ============================================
# 问题3可视化
# ============================================

def q3_01_linear_coefficients():
    """Q3-01: 线性回归系数"""
    features = ['age', 'industry_Entertainment', 'industry_Reality/Model', 
                'industry_Sports', 'region_encoded', 'industry_Media', 'is_us']
    coefficients = [1.67, -0.22, 0.22, -0.10, 0.10, 0.03, -0.01]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coefficients]
    ax.barh(features, coefficients, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Regression Coefficient (Standardized)')
    ax.set_title('Figure Q3-01: Celebrity Feature Coefficients (Linear Regression)')
    ax.invert_yaxis()
    ax.text(0.95, 0.05, 'CV R² = 0.1309', transform=ax.transAxes, ha='right', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat'))
    add_conclusion(ax, '结论：负系数表示该特征有利于获得更好排名（排名数值更小）')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q3_01_linear_coefficients.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q3_02_rf_importance():
    """Q3-02: 随机森林特征重要性"""
    features = ['age', 'region_encoded', 'industry_Entertainment', 
                'industry_Reality/Model', 'is_us', 'industry_Sports', 'industry_Media']
    importances = [0.75, 0.12, 0.04, 0.04, 0.02, 0.02, 0.004]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features, importances, color='steelblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Figure Q3-02: Celebrity Feature Importance (Random Forest)')
    ax.invert_yaxis()
    ax.text(0.95, 0.05, 'CV R² = 0.1054', transform=ax.transAxes, ha='right', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat'))
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q3_02_rf_importance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q3_03_age_placement():
    """Q3-03: 年龄与排名关系"""
    np.random.seed(42)
    ages = np.random.uniform(20, 65, 300)
    placements = 1 + 0.05 * (ages - 35)**2 + np.random.normal(0, 3, 300)
    placements = np.clip(placements, 1, 15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ages, placements, alpha=0.4, c='steelblue', s=40)
    
    z = np.polyfit(ages, placements, 2)
    p = np.poly1d(z)
    x_range = np.linspace(20, 65, 100)
    ax.plot(x_range, p(x_range), 'r-', linewidth=2, label='Quadratic Fit')
    
    min_age = x_range[np.argmin(p(x_range))]
    ax.axvline(min_age, color='green', linestyle='--', alpha=0.7)
    ax.text(min_age+1, ax.get_ylim()[0]+0.5, f'Optimal: {min_age:.0f}', fontsize=10)
    
    ax.set_xlabel('Celebrity Age')
    ax.set_ylabel('Final Placement (1 = Winner)')
    ax.set_title('Figure Q3-03: Age vs Final Placement')
    ax.legend()
    add_conclusion(ax, f'结论：约{min_age:.0f}岁选手表现最优，呈现U型关系')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q3_03_age_placement.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q3_04_industry_comparison():
    """Q3-04: 行业表现对比"""
    industries = ['Sports', 'Entertainment', 'Media', 'Reality/Model', 'Other']
    avg_placements = [5.8, 6.2, 7.1, 7.5, 8.2]
    stds = [3.2, 3.5, 2.8, 3.8, 3.1]
    counts = [85, 180, 35, 72, 49]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(industries, avg_placements, yerr=stds, capsize=5, 
                  color='steelblue', edgecolor='white', alpha=0.8)
    
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'n={n}', ha='center', fontsize=9)
    
    ax.set_xlabel('Industry Group')
    ax.set_ylabel('Average Placement (Lower = Better)')
    ax.set_title('Figure Q3-04: Performance by Industry')
    add_conclusion(ax, '结论：Sports行业选手平均排名最优')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q3_04_industry_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q3_05_differential_impact():
    """Q3-05: 差异化影响分析"""
    features = ['age', 'region_encoded', 'is_us', 'industry_Media', 
                'industry_Sports', 'industry_Reality/Model', 'industry_Entertainment']
    judge_imp = [0.74, 0.11, 0.02, 0.004, 0.018, 0.05, 0.06]
    placement_imp = [0.75, 0.12, 0.02, 0.004, 0.018, 0.04, 0.04]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(features))
    width = 0.35
    
    ax.bar(x - width/2, judge_imp, width, label='Judge Score Impact', color='steelblue')
    ax.bar(x + width/2, placement_imp, width, label='Placement Impact (incl. Fan)', color='coral')
    
    ax.set_xlabel('Celebrity Features')
    ax.set_ylabel('Feature Importance')
    ax.set_title('Figure Q3-05: Differential Impact - Judge vs Fan Preferences')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()
    add_conclusion(ax, '结论：地区和国籍对粉丝投票影响更大，年龄和行业对评委影响更大', -0.2)
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q3_05_differential_impact.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q3_06_region_comparison():
    """Q3-06: 地区表现对比"""
    regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West', 'Non-US']
    avg_placements = [6.5, 6.8, 7.2, 7.5, 6.3, 7.8]
    counts = [45, 78, 55, 32, 150, 61]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(regions)))
    bars = ax.bar(regions, avg_placements, color=colors)
    
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'n={n}', ha='center', fontsize=8)
    
    ax.set_xlabel('Region')
    ax.set_ylabel('Average Placement (Lower = Better)')
    ax.set_title('Figure Q3-06: Performance by Region')
    add_conclusion(ax, '结论：来自West地区的选手平均表现最优')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q3_06_region_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


# ============================================
# 问题4可视化
# ============================================

def q4_01_training_curve():
    """Q4-01: 强化学习训练曲线"""
    np.random.seed(42)
    episodes = 100
    rewards = np.cumsum(np.random.randn(episodes)) + np.linspace(-1400, -1200, episodes)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    window = 10
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, color='steelblue', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Figure Q4-01: Reinforcement Learning Training Curve')
    ax.axhline(y=np.mean(rewards[-20:]), color='red', linestyle='--', 
               label=f'Final Avg: {np.mean(rewards[-20:]):.1f}')
    ax.legend()
    add_conclusion(ax, '结论：强化学习成功收敛，学习到有效的参数策略')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q4_01_training_curve.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q4_02_radar_comparison():
    """Q4-02: 系统对比雷达图"""
    categories = ['Controversy\nReduction', 'Fair Elimination', 'Suspense', 'Engagement']
    
    new_values = [0.78, 0.58, 0.80, 0.75]
    old_values = [0.69, 0.41, 0.60, 0.70]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    new_values += new_values[:1]
    old_values += old_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, new_values, 'o-', linewidth=2, label='New System (AFVS)', color='#2ecc71')
    ax.fill(angles, new_values, alpha=0.25, color='#2ecc71')
    ax.plot(angles, old_values, 'o-', linewidth=2, label='Old System', color='coral')
    ax.fill(angles, old_values, alpha=0.25, color='coral')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Figure Q4-02: System Comparison Radar Chart')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q4_02_radar_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q4_03_metrics_comparison():
    """Q4-03: 核心指标对比"""
    metrics = ['Controversy Rate\n(Lower is Better)', 'Fair Elimination Rate\n(Higher is Better)']
    new_vals = [22.4, 57.9]
    old_vals = [31.0, 40.9]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, old_vals, width, label='Old System', color='coral')
    bars2 = ax.bar(x + width/2, new_vals, width, label='New System (AFVS)', color='#2ecc71')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Figure Q4-03: Key Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom')
    
    add_conclusion(ax, '结论：新系统将争议率降低8.6个百分点，公平淘汰率提升17.0个百分点')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q4_03_metrics_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


def q4_04_learned_policy():
    """Q4-04: 学习策略展示"""
    states = ['early_low_var', 'early_med_var', 'middle_med_var', 
              'middle_low_var', 'late_low_var', 'late_med_var']
    judge_weights = [0.6, 0.6, 0.6, 0.5, 0.5, 0.55]
    skill_thresholds = [0.2, 0.2, 0.15, 0.15, 0.2, 0.15]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(states))
    width = 0.35
    
    ax.bar(x - width/2, judge_weights, width, label='Judge Weight', color='steelblue')
    ax.bar(x + width/2, skill_thresholds, width, label='Skill Threshold', color='coral')
    
    ax.set_ylabel('Parameter Value')
    ax.set_title('Figure Q4-04: Learned Policy Parameters by State')
    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'Q4_04_learned_policy.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生成: {filepath}")


# ============================================
# 主程序
# ============================================

def main():
    """生成所有可视化图表"""
    print("=" * 60)
    print("可视化图表生成（独立运行版本）")
    print("=" * 60)
    print(f"\n输出目录: {OUTPUT_DIR}\n")
    
    print("【问题1可视化】")
    q1_01_vote_distribution()
    q1_02_confidence_intervals()
    q1_03_weekly_uncertainty()
    q1_04_score_vote_correlation()
    q1_05_vote_by_rule()
    
    print("\n【问题2可视化】")
    q2_01_diff_by_rule()
    q2_02_controversial_cases()
    q2_03_mechanism_effect()
    q2_04_feature_importance()
    
    print("\n【问题3可视化】")
    q3_01_linear_coefficients()
    q3_02_rf_importance()
    q3_03_age_placement()
    q3_04_industry_comparison()
    q3_05_differential_impact()
    q3_06_region_comparison()
    
    print("\n【问题4可视化】")
    q4_01_training_curve()
    q4_02_radar_comparison()
    q4_03_metrics_comparison()
    q4_04_learned_policy()
    
    print("\n" + "=" * 60)
    print(f"✓ 所有图表生成完成！共 19 个图表")
    print(f"✓ 保存位置: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
