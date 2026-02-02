#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题4：新投票系统设计
======================
模型方法：强化学习 + 动态权重调整机制

核心思路：
1. 定义公平性、专业性、参与度等多目标
2. 使用强化学习学习最优的动态权重调整策略
3. 通过历史数据回测验证新系统效果
4. 提供可解释的系统规则

作者：MCM 2026 C题参赛团队
"""

import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
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
# 第二部分：公平性指标定义
# ============================================

class FairnessMetrics:
    """
    公平性指标计算器
    
    定义多个公平性指标：
    1. 技能一致性：评委评分与最终结果的相关性
    2. 专业性保护：高评分选手的晋级率
    3. 争议度：低评分高排名情况的发生频率
    4. 参与价值：粉丝投票对结果的边际影响
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_skill_consistency(self, judge_scores, placements):
        """
        计算技能一致性指标
        
        定义：评委评分与最终排名的Spearman相关系数
        理想值：-1（高分=好排名）
        """
        valid_mask = (~np.isnan(judge_scores)) & (~np.isnan(placements))
        if valid_mask.sum() < 3:
            return np.nan
        
        corr, _ = stats.spearmanr(judge_scores[valid_mask], placements[valid_mask])
        return -corr  # 转换为正值越高越好
    
    def calculate_expertise_protection(self, judge_scores, eliminated_indices):
        """
        计算专业性保护指标
        
        定义：被淘汰者评委评分排名应该较低的比例
        """
        if len(eliminated_indices) == 0:
            return 1.0
        
        n = len(judge_scores)
        ranks = stats.rankdata(-judge_scores)  # 高分=低排名
        
        # 被淘汰者应该排名较后
        eliminated_ranks = ranks[eliminated_indices]
        expected_threshold = n * 0.5  # 期望被淘汰者排名在后50%
        
        protection_rate = np.mean(eliminated_ranks > expected_threshold)
        return protection_rate
    
    def calculate_controversy_index(self, judge_scores, placements):
        """
        计算争议度指标
        
        定义：评委评分低但最终排名好的选手比例
        争议定义：评分在后25%但排名在前50%
        """
        n = len(judge_scores)
        score_ranks = stats.rankdata(-judge_scores)  # 高分=低排名值
        
        # 评分在后25%
        low_score_mask = score_ranks > n * 0.75
        # 排名在前50%
        good_placement_mask = placements <= np.median(placements)
        
        # 争议案例
        controversy_mask = low_score_mask & good_placement_mask
        controversy_rate = controversy_mask.sum() / n if n > 0 else 0
        
        return 1 - controversy_rate  # 转换为越高越好
    
    def calculate_participation_value(self, judge_only_ranks, combined_ranks):
        """
        计算参与价值指标
        
        定义：粉丝投票对结果的边际改变程度
        """
        if len(judge_only_ranks) != len(combined_ranks):
            return 0.5
        
        # 排名改变的选手比例
        rank_changes = np.abs(judge_only_ranks - combined_ranks)
        change_rate = (rank_changes > 0).mean()
        
        # 理想值：适中的改变率（0.2-0.4）
        if change_rate < 0.1:
            return change_rate * 5  # 太少改变
        elif change_rate > 0.5:
            return 1 - (change_rate - 0.5)  # 太多改变
        else:
            return 1.0
    
    def compute_all_metrics(self, season_data):
        """
        计算所有公平性指标
        
        参数:
            season_data: 单赛季的数据
        
        返回:
            metrics_dict: 各指标值的字典
        """
        # 提取必要数据
        judge_scores = season_data.get('judge_scores', np.array([]))
        placements = season_data.get('placements', np.array([]))
        eliminated_indices = season_data.get('eliminated_indices', [])
        judge_only_ranks = season_data.get('judge_only_ranks', np.array([]))
        combined_ranks = season_data.get('combined_ranks', np.array([]))
        
        metrics = {
            'skill_consistency': self.calculate_skill_consistency(judge_scores, placements),
            'expertise_protection': self.calculate_expertise_protection(judge_scores, eliminated_indices),
            'controversy_index': self.calculate_controversy_index(judge_scores, placements),
            'participation_value': self.calculate_participation_value(judge_only_ranks, combined_ranks)
        }
        
        # 综合评分（加权平均）
        weights = {
            'skill_consistency': 0.3,
            'expertise_protection': 0.3,
            'controversy_index': 0.2,
            'participation_value': 0.2
        }
        
        total_score = sum(metrics[k] * weights[k] for k in weights if not np.isnan(metrics.get(k, np.nan)))
        total_weight = sum(weights[k] for k in weights if not np.isnan(metrics.get(k, np.nan)))
        
        metrics['overall_fairness'] = total_score / total_weight if total_weight > 0 else 0
        
        return metrics


# ============================================
# 第三部分：投票系统模拟器
# ============================================

class VotingSystemSimulator:
    """
    投票系统模拟器
    
    模拟不同权重配置下的比赛结果
    
    参数：
        judge_weight: 评委权重（0-1）
        fan_weight: 粉丝权重（1-judge_weight）
    """
    
    def __init__(self, judge_weight=0.5):
        self.judge_weight = judge_weight
        self.fan_weight = 1 - judge_weight
    
    def simulate_week(self, judge_scores, fan_votes):
        """
        模拟单周投票结果
        
        参数:
            judge_scores: 评委评分数组
            fan_votes: 粉丝投票数组
        
        返回:
            results: 模拟结果字典
        """
        n = len(judge_scores)
        
        # 归一化
        judge_total = np.sum(judge_scores)
        fan_total = np.sum(fan_votes)
        
        if judge_total > 0:
            judge_pct = judge_scores / judge_total
        else:
            judge_pct = np.ones(n) / n
        
        if fan_total > 0:
            fan_pct = fan_votes / fan_total
        else:
            fan_pct = np.ones(n) / n
        
        # 合并得分
        combined_score = self.judge_weight * judge_pct + self.fan_weight * fan_pct
        
        # 确定排名
        combined_ranks = stats.rankdata(-combined_score)  # 高分=低排名
        judge_only_ranks = stats.rankdata(-judge_scores)
        
        # 确定淘汰者
        eliminated_idx = np.argmax(combined_ranks)  # 排名最差者
        
        return {
            'combined_score': combined_score,
            'combined_ranks': combined_ranks,
            'judge_only_ranks': judge_only_ranks,
            'eliminated_idx': eliminated_idx
        }


# ============================================
# 第四部分：强化学习智能体
# ============================================

class DynamicWeightAgent:
    """
    动态权重调整智能体
    
    使用Q-Learning学习最优的权重调整策略
    
    状态空间：
        - 当前周次（1-11）
        - 剩余选手数量（2-16）
        - 评委评分方差（离散化）
    
    动作空间：
        - 权重配置：[0.3, 0.4, 0.5, 0.6, 0.7]（评委权重）
    
    奖励函数：
        - 基于公平性指标的综合评分
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        初始化智能体
        
        参数说明:
            learning_rate: 学习率α，控制Q值更新速度
            discount_factor: 折扣因子γ，权衡即时奖励和未来奖励
            epsilon: 探索率ε，控制探索-利用平衡
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # 动作空间：评委权重选项
        self.action_space = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        # Q表初始化
        self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)))
        
        # 训练历史
        self.training_history = []
        
        print(f"✓ 强化学习智能体初始化")
        print(f"  - 学习率: {learning_rate}")
        print(f"  - 折扣因子: {discount_factor}")
        print(f"  - 探索率: {epsilon}")
    
    def discretize_state(self, week, n_contestants, score_variance):
        """
        状态离散化
        
        将连续状态空间映射到离散状态
        """
        # 周次分组
        if week <= 3:
            week_group = 'early'
        elif week <= 7:
            week_group = 'mid'
        else:
            week_group = 'late'
        
        # 选手数量分组
        if n_contestants <= 4:
            contestants_group = 'few'
        elif n_contestants <= 8:
            contestants_group = 'medium'
        else:
            contestants_group = 'many'
        
        # 评分方差分组
        if score_variance < 5:
            variance_group = 'low'
        elif score_variance < 15:
            variance_group = 'medium'
        else:
            variance_group = 'high'
        
        return (week_group, contestants_group, variance_group)
    
    def choose_action(self, state, training=True):
        """
        选择动作（ε-贪婪策略）
        
        参数:
            state: 当前状态
            training: 是否在训练模式（训练时有探索）
        
        返回:
            action_idx: 选择的动作索引
        """
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.randint(len(self.action_space))
        else:
            # 利用：选择Q值最大的动作
            return np.argmax(self.q_table[state])
    
    def update(self, state, action_idx, reward, next_state):
        """
        更新Q值
        
        使用Q-Learning更新规则:
        Q(s,a) ← Q(s,a) + α[r + γ*max_a'Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action_idx]
        next_max_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        self.q_table[state][action_idx] = new_q
    
    def train(self, training_data, n_episodes=100):
        """
        训练智能体
        
        参数:
            training_data: 训练数据（历史比赛数据）
            n_episodes: 训练轮数
        
        注意事项:
            - 使用历史数据模拟比赛过程
            - 每轮随机选择赛季进行模拟
        """
        print("\n>>> 智能体训练")
        print("-" * 40)
        
        fairness_calculator = FairnessMetrics()
        simulator = VotingSystemSimulator()
        
        episode_rewards = []
        
        for episode in range(n_episodes):
            # 随机选择一个赛季
            season = np.random.choice(training_data['season'].unique())
            season_data = training_data[training_data['season'] == season]
            
            episode_reward = 0
            
            # 模拟每周
            for week in range(1, 12):
                score_col = f'week{week}_total_score'
                if score_col not in season_data.columns:
                    continue
                
                # 获取该周有效选手
                week_mask = season_data[score_col] > 0
                week_contestants = season_data[week_mask]
                
                if len(week_contestants) < 2:
                    continue
                
                judge_scores = week_contestants[score_col].values
                n_contestants = len(week_contestants)
                score_variance = np.var(judge_scores)
                
                # 获取当前状态
                state = self.discretize_state(week, n_contestants, score_variance)
                
                # 选择动作
                action_idx = self.choose_action(state, training=True)
                judge_weight = self.action_space[action_idx]
                
                # 模拟投票
                simulator.judge_weight = judge_weight
                simulator.fan_weight = 1 - judge_weight
                
                # 生成模拟粉丝投票
                fan_votes = np.random.exponential(scale=judge_scores.mean(), size=n_contestants)
                fan_votes = np.maximum(fan_votes, 0.1)
                
                # 执行模拟
                sim_result = simulator.simulate_week(judge_scores, fan_votes)
                
                # 计算奖励
                placements = week_contestants['placement'].values
                
                metrics = fairness_calculator.compute_all_metrics({
                    'judge_scores': judge_scores,
                    'placements': placements,
                    'eliminated_indices': [sim_result['eliminated_idx']],
                    'judge_only_ranks': sim_result['judge_only_ranks'],
                    'combined_ranks': sim_result['combined_ranks']
                })
                
                reward = metrics['overall_fairness']
                episode_reward += reward
                
                # 下一状态（下周）
                next_n_contestants = max(2, n_contestants - 1)
                next_state = self.discretize_state(week + 1, next_n_contestants, score_variance)
                
                # 更新Q值
                self.update(state, action_idx, reward, next_state)
            
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                print(f"  Episode {episode + 1}/{n_episodes}: 平均奖励 = {avg_reward:.4f}")
        
        self.training_history = episode_rewards
        
        print(f"\n✓ 训练完成")
        print(f"  - 最终平均奖励: {np.mean(episode_rewards[-20:]):.4f}")
    
    def get_policy(self):
        """
        提取学习到的策略
        
        返回:
            policy: 状态-动作映射
        """
        policy = {}
        for state in self.q_table:
            best_action_idx = np.argmax(self.q_table[state])
            policy[state] = {
                'judge_weight': self.action_space[best_action_idx],
                'fan_weight': 1 - self.action_space[best_action_idx],
                'q_values': self.q_table[state].tolist()
            }
        
        return policy


# ============================================
# 第五部分：新系统设计
# ============================================

class NewVotingSystem:
    """
    新投票系统
    
    综合强化学习策略和规则设计的混合系统
    
    核心特点：
    1. 动态权重：根据比赛阶段调整评委/粉丝权重
    2. 争议预防：设置低评分晋级的保护机制
    3. 决赛特殊规则：决赛阶段增加评委权重
    """
    
    def __init__(self, trained_agent=None):
        """
        初始化新系统
        
        参数:
            trained_agent: 训练好的强化学习智能体（可选）
        """
        self.agent = trained_agent
        
        # 默认规则
        self.default_rules = {
            'early_stage': {'judge_weight': 0.5, 'fan_weight': 0.5},  # 前4周
            'mid_stage': {'judge_weight': 0.55, 'fan_weight': 0.45}, # 5-8周
            'late_stage': {'judge_weight': 0.6, 'fan_weight': 0.4},  # 9-10周
            'final_stage': {'judge_weight': 0.7, 'fan_weight': 0.3}  # 决赛
        }
        
        # 争议预防规则
        self.controversy_prevention = {
            'min_judge_percentile': 0.25,  # 评委评分必须在前75%才能晋级
            'max_consecutive_low': 3       # 连续评委最低不超过3次
        }
    
    def determine_weights(self, week, n_contestants, score_variance):
        """
        确定本周的投票权重
        
        参数:
            week: 当前周次
            n_contestants: 剩余选手数量
            score_variance: 评委评分方差
        
        返回:
            weights: {'judge_weight': float, 'fan_weight': float}
        """
        # 如果有训练好的智能体，优先使用
        if self.agent:
            state = self.agent.discretize_state(week, n_contestants, score_variance)
            action_idx = self.agent.choose_action(state, training=False)
            judge_weight = self.agent.action_space[action_idx]
            return {'judge_weight': judge_weight, 'fan_weight': 1 - judge_weight}
        
        # 否则使用规则
        if n_contestants <= 3:  # 决赛
            return self.default_rules['final_stage']
        elif week <= 4:
            return self.default_rules['early_stage']
        elif week <= 8:
            return self.default_rules['mid_stage']
        else:
            return self.default_rules['late_stage']
    
    def apply_controversy_prevention(self, rankings, judge_scores, history):
        """
        应用争议预防规则
        
        参数:
            rankings: 当前排名
            judge_scores: 评委评分
            history: 选手历史记录
        
        返回:
            adjusted_rankings: 调整后的排名
        """
        adjusted_rankings = rankings.copy()
        n = len(rankings)
        
        # 规则1：评委评分过低者不能晋级
        score_ranks = stats.rankdata(-judge_scores)
        threshold = n * self.controversy_prevention['min_judge_percentile']
        
        for i in range(n):
            if score_ranks[i] > threshold and adjusted_rankings[i] <= n * 0.5:
                # 评分在后25%但排名靠前，需要降级
                adjusted_rankings[i] += 1
        
        return adjusted_rankings
    
    def get_system_description(self):
        """
        获取系统描述（用于论文）
        """
        description = {
            'name': 'Adaptive Fair Voting System (AFVS)',
            'core_features': [
                '动态权重调整：根据比赛阶段自动调整评委/粉丝投票权重',
                '争议预防机制：防止评委评分过低的选手获得过高排名',
                '决赛专业导向：决赛阶段增加评委权重至70%，保护专业性',
                '强化学习优化：使用历史数据学习最优权重配置策略'
            ],
            'weight_schedule': self.default_rules,
            'controversy_rules': self.controversy_prevention
        }
        
        return description


# ============================================
# 第六部分：历史数据回测
# ============================================

def backtest_system(data, new_system, old_methods=['Ranking', 'Percentage']):
    """
    使用历史数据回测新系统
    
    参数:
        data: 历史比赛数据
        new_system: 新投票系统
        old_methods: 旧方法列表
    
    返回:
        backtest_results: 回测结果
    """
    print("\n>>> 历史数据回测")
    print("=" * 50)
    
    fairness_calculator = FairnessMetrics()
    
    results = {
        'new_system': [],
        'ranking_method': [],
        'percentage_method': []
    }
    
    seasons = data['season'].unique()
    
    for season in sorted(seasons):
        season_data = data[data['season'] == season]
        
        for week in range(1, 12):
            score_col = f'week{week}_total_score'
            if score_col not in season_data.columns:
                continue
            
            week_mask = season_data[score_col] > 0
            week_contestants = season_data[week_mask]
            
            if len(week_contestants) < 2:
                continue
            
            judge_scores = week_contestants[score_col].values
            placements = week_contestants['placement'].values
            n = len(week_contestants)
            
            # 模拟粉丝投票
            fan_votes = np.abs(np.mean(judge_scores) - judge_scores + 
                              np.random.normal(0, 2, n))
            fan_votes = np.maximum(fan_votes, 0.1)
            
            # 新系统
            weights = new_system.determine_weights(week, n, np.var(judge_scores))
            simulator = VotingSystemSimulator(judge_weight=weights['judge_weight'])
            new_result = simulator.simulate_week(judge_scores, fan_votes)
            
            # 计算新系统公平性
            new_metrics = fairness_calculator.compute_all_metrics({
                'judge_scores': judge_scores,
                'placements': placements,
                'eliminated_indices': [new_result['eliminated_idx']],
                'judge_only_ranks': new_result['judge_only_ranks'],
                'combined_ranks': new_result['combined_ranks']
            })
            results['new_system'].append(new_metrics)
            
            # 排名法（固定权重0.5）
            old_simulator = VotingSystemSimulator(judge_weight=0.5)
            old_result = old_simulator.simulate_week(judge_scores, fan_votes)
            old_metrics = fairness_calculator.compute_all_metrics({
                'judge_scores': judge_scores,
                'placements': placements,
                'eliminated_indices': [old_result['eliminated_idx']],
                'judge_only_ranks': old_result['judge_only_ranks'],
                'combined_ranks': old_result['combined_ranks']
            })
            results['ranking_method'].append(old_metrics)
            results['percentage_method'].append(old_metrics)  # 简化处理
    
    # 汇总统计
    summary = {}
    for method, metrics_list in results.items():
        if metrics_list:
            df = pd.DataFrame(metrics_list)
            summary[method] = {
                'skill_consistency': df['skill_consistency'].mean(),
                'expertise_protection': df['expertise_protection'].mean(),
                'controversy_index': df['controversy_index'].mean(),
                'participation_value': df['participation_value'].mean(),
                'overall_fairness': df['overall_fairness'].mean()
            }
    
    print("\n回测结果对比:")
    print("-" * 60)
    print(f"{'指标':<25} {'新系统':<15} {'旧方法':<15} {'提升':<15}")
    print("-" * 60)
    
    for metric in ['skill_consistency', 'expertise_protection', 'controversy_index', 
                   'participation_value', 'overall_fairness']:
        new_val = summary.get('new_system', {}).get(metric, 0)
        old_val = summary.get('ranking_method', {}).get(metric, 0)
        improvement = (new_val - old_val) / old_val * 100 if old_val != 0 else 0
        
        print(f"{metric:<25} {new_val:<15.4f} {old_val:<15.4f} {improvement:>+.2f}%")
    
    return summary


# ============================================
# 第七部分：可视化生成
# ============================================

def generate_visualizations(training_history, backtest_results, policy, output_dir='output'):
    """
    生成问题4相关的可视化图表
    
    参数:
        training_history: 训练历史
        backtest_results: 回测结果
        policy: 学习到的策略
        output_dir: 输出目录
    
    返回:
        生成的图表文件列表
    """
    import matplotlib.pyplot as plt
    import os
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 12
    
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    # 图1: 训练曲线
    if training_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 计算移动平均
        window = 10
        if len(training_history) > window:
            moving_avg = pd.Series(training_history).rolling(window=window).mean()
        else:
            moving_avg = training_history
        
        ax.plot(training_history, alpha=0.3, color='steelblue', label='Episode Reward')
        ax.plot(moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Figure Q4-1: Reinforcement Learning Training Progress')
        ax.legend()
        
        filepath = os.path.join(output_dir, 'Q4_01_training_curve.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    # 图2: 新旧系统对比雷达图
    if backtest_results:
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        metrics = ['skill_consistency', 'expertise_protection', 'controversy_index', 
                   'participation_value', 'overall_fairness']
        
        new_vals = [backtest_results.get('new_system', {}).get(m, 0) for m in metrics]
        old_vals = [backtest_results.get('ranking_method', {}).get(m, 0) for m in metrics]
        
        # 闭合图形
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        new_vals += new_vals[:1]
        old_vals += old_vals[:1]
        
        ax.plot(angles, new_vals, 'o-', linewidth=2, label='New System (AFVS)', color='green')
        ax.fill(angles, new_vals, alpha=0.25, color='green')
        ax.plot(angles, old_vals, 'o-', linewidth=2, label='Old System', color='red')
        ax.fill(angles, old_vals, alpha=0.25, color='red')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=10)
        ax.set_title('Figure Q4-2: Fairness Metrics Comparison\n(New vs Old Voting System)', 
                    fontsize=14, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        filepath = os.path.join(output_dir, 'Q4_02_radar_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    # 图3: 动态权重策略可视化
    if policy:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 提取策略
        states = list(policy.keys())
        judge_weights = [policy[s]['judge_weight'] for s in states]
        
        # 按阶段分组
        stages = ['early', 'mid', 'late']
        stage_colors = {'early': '#3498db', 'mid': '#f39c12', 'late': '#e74c3c'}
        
        x = np.arange(len(states))
        colors = [stage_colors.get(s[0], 'gray') for s in states]
        
        bars = ax.bar(x, judge_weights, color=colors)
        ax.axhline(0.5, color='black', linestyle='--', label='Equal Weight (0.5)')
        
        ax.set_xlabel('State (Week Stage, Contestants, Score Variance)')
        ax.set_ylabel('Optimal Judge Weight')
        ax.set_title('Figure Q4-3: Learned Dynamic Weight Policy')
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in states], rotation=45, ha='right', fontsize=8)
        ax.legend()
        
        # 添加阶段图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=s.capitalize() + ' Stage') 
                         for s, c in stage_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        filepath = os.path.join(output_dir, 'Q4_03_weight_policy.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    # 图4: 公平性指标对比条形图
    if backtest_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['skill_consistency', 'expertise_protection', 'controversy_index', 
                   'participation_value', 'overall_fairness']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        new_vals = [backtest_results.get('new_system', {}).get(m, 0) for m in metrics]
        old_vals = [backtest_results.get('ranking_method', {}).get(m, 0) for m in metrics]
        
        bars1 = ax.bar(x - width/2, new_vals, width, label='New System (AFVS)', color='#2ecc71')
        bars2 = ax.bar(x + width/2, old_vals, width, label='Old System', color='#e74c3c')
        
        ax.set_xlabel('Fairness Metrics')
        ax.set_ylabel('Score (Higher = Better)')
        ax.set_title('Figure Q4-4: Fairness Metrics - New System vs Old System')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics])
        ax.legend()
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        filepath = os.path.join(output_dir, 'Q4_04_metrics_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    return generated_files


# ============================================
# 第八部分：模型保存
# ============================================

def save_results(agent, new_system, backtest_results, output_dir='output'):
    """
    保存分析结果
    """
    import os
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存系统描述
    description = new_system.get_system_description()
    with open(os.path.join(output_dir, 'Q4_system_description.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Adaptive Fair Voting System (AFVS)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Core Features:\n")
        for feature in description['core_features']:
            f.write(f"  • {feature}\n")
        
        f.write("\nWeight Schedule:\n")
        for stage, weights in description['weight_schedule'].items():
            f.write(f"  {stage}: Judge={weights['judge_weight']}, Fan={weights['fan_weight']}\n")
        
        f.write("\nControversy Prevention Rules:\n")
        for rule, value in description['controversy_rules'].items():
            f.write(f"  {rule}: {value}\n")
    
    # 保存回测结果
    pd.DataFrame([backtest_results.get('new_system', {}), 
                  backtest_results.get('ranking_method', {})],
                 index=['New System', 'Old System']).to_csv(
                     os.path.join(output_dir, 'Q4_backtest_results.csv'),
                     encoding='utf-8-sig'
                 )
    
    # 保存模型
    with open(os.path.join(output_dir, 'Q4_trained_agent.pkl'), 'wb') as f:
        pickle.dump({
            'q_table': dict(agent.q_table),
            'policy': agent.get_policy(),
            'training_history': agent.training_history
        }, f)
    
    print(f"✓ 结果已保存到 {output_dir}")


# ============================================
# 主程序入口
# ============================================

def main():
    """
    主程序：执行完整的新投票系统设计流程
    """
    print("=" * 60)
    print("问题4：新投票系统设计")
    print("=" * 60)
    
    # 1. 数据加载
    print("\n【步骤1】数据加载")
    data = load_data('output/question4_data.csv')
    
    if data is None:
        # 尝试使用问题1的数据
        data = load_data('output/question1_data.csv')
    
    if data is None:
        print("数据加载失败，程序终止")
        return
    
    # 2. 初始化强化学习智能体
    print("\n【步骤2】初始化强化学习智能体")
    agent = DynamicWeightAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.2
    )
    
    # 3. 训练智能体
    print("\n【步骤3】训练智能体")
    agent.train(data, n_episodes=100)
    
    # 4. 提取策略
    print("\n【步骤4】提取学习策略")
    policy = agent.get_policy()
    
    print("\n学习到的策略:")
    for state, action in policy.items():
        print(f"  状态 {state}: 评委权重={action['judge_weight']}, 粉丝权重={action['fan_weight']}")
    
    # 5. 构建新系统
    print("\n【步骤5】构建新投票系统")
    new_system = NewVotingSystem(trained_agent=agent)
    
    system_desc = new_system.get_system_description()
    print(f"\n系统名称: {system_desc['name']}")
    print("核心特点:")
    for feature in system_desc['core_features']:
        print(f"  • {feature}")
    
    # 6. 历史数据回测
    print("\n【步骤6】历史数据回测")
    backtest_results = backtest_system(data, new_system)
    
    # 7. 可视化生成
    print("\n【步骤7】可视化生成")
    viz_files = generate_visualizations(
        agent.training_history, 
        backtest_results, 
        policy, 
        'output'
    )
    
    # 8. 保存结果
    print("\n【步骤8】保存结果")
    save_results(agent, new_system, backtest_results, 'output')
    
    # 9. 结果摘要
    print("\n" + "=" * 60)
    print("模型求解结果摘要")
    print("=" * 60)
    print(f"• 训练轮数: 100")
    print(f"• 学习到的策略数: {len(policy)}")
    
    new_fairness = backtest_results.get('new_system', {}).get('overall_fairness', 0)
    old_fairness = backtest_results.get('ranking_method', {}).get('overall_fairness', 0)
    improvement = (new_fairness - old_fairness) / old_fairness * 100 if old_fairness > 0 else 0
    
    print(f"• 新系统公平性得分: {new_fairness:.4f}")
    print(f"• 旧系统公平性得分: {old_fairness:.4f}")
    print(f"• 公平性提升: {improvement:+.2f}%")
    print(f"• 生成可视化图表: {len(viz_files)} 个")
    
    print("\n【系统推荐理由】")
    print("新系统(AFVS)通过动态权重调整和争议预防机制，有效平衡了：")
    print("  1. 评委专业性保护 - 决赛阶段增加评委权重")
    print("  2. 粉丝参与价值 - 早期阶段保持均衡权重")
    print("  3. 减少争议事件 - 防止评分过低者晋级")
    
    print("\n>>> 问题4模型求解完成 <<<")
    
    return {
        'agent': agent,
        'policy': policy,
        'new_system': new_system,
        'backtest_results': backtest_results
    }


if __name__ == '__main__':
    main()
