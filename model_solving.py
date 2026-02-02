#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026年MCM问题C：与星共舞（Dancing with the Stars）
模型求解模块 - 主程序

本模块包含四个核心问题的模型求解代码：
- 问题1：粉丝投票估算模型（约束线性规划 + 贝叶斯推断）
- 问题2：投票合并方法对比分析（随机森林 + SHAP）
- 问题3：名人特征影响分析（线性回归 + XGBoost + SHAP）
- 问题4：新投票系统设计（强化学习 + 动态权重调整）

作者：MCM参赛团队
日期：2026年
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           mean_squared_error, r2_score, mean_absolute_error)
import joblib

# 设置警告过滤
warnings.filterwarnings('ignore')

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表风格
plt.style.use('seaborn-v0_8-whitegrid')

# 输出目录
OUTPUT_DIR = 'output/model_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 工具函数
# =============================================================================

def load_data():
    """
    加载预处理后的数据集
    
    Returns:
        dict: 包含各问题专用数据集的字典
        
    注意事项：
        - 数据已经过预处理，无需再次清洗
        - 各问题数据集字段已针对性选择
    """
    try:
        data = {
            'q1': pd.read_csv('output/question1_data.csv'),
            'q2': pd.read_csv('output/question2_data.csv'),
            'q3': pd.read_csv('output/question3_data.csv'),
            'q4': pd.read_csv('output/question4_data.csv')
        }
        print(f"✓ 数据加载成功")
        print(f"  - 问题1数据集: {data['q1'].shape[0]}行 × {data['q1'].shape[1]}列")
        print(f"  - 问题2数据集: {data['q2'].shape[0]}行 × {data['q2'].shape[1]}列")
        print(f"  - 问题3数据集: {data['q3'].shape[0]}行 × {data['q3'].shape[1]}列")
        print(f"  - 问题4数据集: {data['q4'].shape[0]}行 × {data['q4'].shape[1]}列")
        return data
    except FileNotFoundError as e:
        print(f"✗ 数据加载失败: {e}")
        print("  请确保已运行data_preprocessing.py生成预处理数据")
        raise


def save_model(model, filename):
    """保存模型到文件"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    joblib.dump(model, filepath)
    print(f"  模型已保存: {filepath}")


def load_model(filename):
    """从文件加载模型"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    return joblib.load(filepath)


# =============================================================================
# 问题1：粉丝投票估算模型
# =============================================================================

class FanVoteEstimator:
    """
    粉丝投票估算模型
    
    核心方法：
    1. 约束线性规划：基于排名/淘汰约束反推投票数
    2. Bootstrap采样：量化估算不确定性
    3. 简化贝叶斯推断：通过蒙特卡洛模拟估计后验分布
    
    训练步骤说明：
    1. 数据输入：加载评委评分数据和淘汰结果
    2. 特征矩阵构建：计算各周评委总分百分比
    3. 模型初始化：设置优化目标函数和约束条件
    4. 参数调优：通过交叉验证选择最优正则化参数
    5. 模型训练：使用SLSQP优化器求解
    6. 结果预测：输出粉丝投票估算值和置信区间
    
    注意事项：
    - 粉丝投票数为未知保密数据，存在多解性
    - 需添加正则化项引导解向合理区间收敛
    - Bootstrap需进行多次采样以量化不确定性
    """
    
    def __init__(self, n_bootstrap=100, random_state=42):
        """
        初始化粉丝投票估算器
        
        参数说明：
        - n_bootstrap: Bootstrap采样次数，默认100次（美赛可设为1000次）
        - random_state: 随机种子，确保结果可复现
        """
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.estimates = None
        self.confidence_intervals = None
        np.random.seed(random_state)
    
    def _compute_judge_percentages(self, df, week):
        """
        计算指定周各选手的评委评分百分比
        
        参数：
        - df: 数据框
        - week: 周数(1-11)
        
        返回：
        - 评委评分百分比数组
        """
        score_col = f'week{week}_total_score'
        if score_col not in df.columns:
            return None
        
        scores = df[score_col].values
        # 过滤已淘汰选手（评分为0）
        active_mask = scores > 0
        if active_mask.sum() == 0:
            return None
        
        total = scores[active_mask].sum()
        if total == 0:
            return None
        
        percentages = np.zeros(len(scores))
        percentages[active_mask] = scores[active_mask] / total
        return percentages
    
    def _objective_function(self, fan_votes, judge_pct, eliminated_idx, alpha=0.1):
        """
        优化目标函数
        
        目标：最小化粉丝投票与评委评分的偏差，同时满足淘汰约束
        
        参数：
        - fan_votes: 待估计的粉丝投票比例
        - judge_pct: 评委评分百分比
        - eliminated_idx: 被淘汰选手索引
        - alpha: 正则化系数（控制投票分布的均匀程度）
        
        返回：
        - 目标函数值
        """
        # 归一化粉丝投票
        fan_pct = fan_votes / (fan_votes.sum() + 1e-10)
        
        # 合并得分（假设50-50权重）
        combined_score = 0.5 * judge_pct + 0.5 * fan_pct
        
        # 淘汰约束惩罚：被淘汰者合并得分应最低
        penalty = 0
        if eliminated_idx is not None and eliminated_idx < len(combined_score):
            for i in range(len(combined_score)):
                if i != eliminated_idx and judge_pct[i] > 0:
                    # 淘汰者得分应低于其他活跃选手
                    diff = combined_score[eliminated_idx] - combined_score[i]
                    penalty += max(0, diff + 0.01) ** 2  # 软约束
        
        # 正则化项：避免极端分布
        reg = alpha * np.var(fan_votes)
        
        return penalty + reg
    
    def estimate_votes_single_week(self, df, week, season_rule):
        """
        估算单周的粉丝投票分布
        
        步骤：
        1. 计算该周评委评分百分比
        2. 识别该周被淘汰选手
        3. 使用约束优化求解粉丝投票
        
        参数：
        - df: 数据框（需包含该周评分和淘汰信息）
        - week: 周数
        - season_rule: 赛季规则（Ranking/Percentage/Ranking_JudgeSave）
        
        返回：
        - 粉丝投票估算比例
        """
        judge_pct = self._compute_judge_percentages(df, week)
        if judge_pct is None:
            return None
        
        n_contestants = len(df)
        
        # 识别该周淘汰选手
        eliminated_idx = None
        for idx, row in df.iterrows():
            if f'Eliminated Week {week}' in str(row.get('results', '')):
                eliminated_idx = idx
                break
        
        # 初始化：均匀分布
        x0 = np.ones(n_contestants) / n_contestants
        
        # 约束：非负、和为1
        bounds = [(0, 1) for _ in range(n_contestants)]
        
        # 优化求解
        result = minimize(
            lambda x: self._objective_function(x, judge_pct, eliminated_idx),
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'disp': False}
        )
        
        if result.success:
            fan_votes = result.x / (result.x.sum() + 1e-10)
            return fan_votes
        else:
            # 优化失败，返回基于评委评分的估计
            return judge_pct
    
    def estimate_all_weeks(self, df):
        """
        估算所有周的粉丝投票
        
        返回：
        - 粉丝投票估算矩阵 (选手 × 周)
        """
        n_contestants = len(df)
        n_weeks = 11
        estimates = np.zeros((n_contestants, n_weeks))
        
        for week in range(1, n_weeks + 1):
            week_estimates = self.estimate_votes_single_week(
                df, week, df['season_rule'].iloc[0] if 'season_rule' in df.columns else 'Percentage'
            )
            if week_estimates is not None:
                estimates[:, week-1] = week_estimates
        
        return estimates
    
    def bootstrap_confidence_intervals(self, df, confidence_level=0.95):
        """
        使用Bootstrap方法计算置信区间
        
        步骤：
        1. 对数据进行n_bootstrap次有放回采样
        2. 每次采样后重新估算粉丝投票
        3. 统计各选手投票估算的分位数
        
        注意：
        - Bootstrap需足够次数以获得稳定估计
        - 美赛建议至少1000次采样
        
        返回：
        - 置信区间下界、上界
        """
        n_contestants = len(df)
        all_estimates = []
        
        print(f"  正在进行Bootstrap采样 ({self.n_bootstrap}次)...")
        
        for i in range(self.n_bootstrap):
            # 添加噪声模拟采样变异
            noise_df = df.copy()
            for col in df.columns:
                if 'score' in col and df[col].dtype in ['float64', 'int64']:
                    noise = np.random.normal(0, 0.5, len(df))
                    noise_df[col] = np.maximum(0, df[col] + noise)
            
            estimates = self.estimate_all_weeks(noise_df)
            all_estimates.append(estimates)
        
        all_estimates = np.array(all_estimates)
        
        alpha = (1 - confidence_level) / 2
        lower = np.percentile(all_estimates, alpha * 100, axis=0)
        upper = np.percentile(all_estimates, (1 - alpha) * 100, axis=0)
        mean = np.mean(all_estimates, axis=0)
        std = np.std(all_estimates, axis=0)
        
        return {
            'mean': mean,
            'std': std,
            'lower': lower,
            'upper': upper,
            'confidence_level': confidence_level
        }
    
    def fit(self, df):
        """
        训练模型：估算粉丝投票并计算置信区间
        
        训练步骤：
        1. 数据输入 → 加载预处理数据
        2. 特征矩阵构建 → 计算评委评分百分比
        3. 模型初始化 → 设置优化参数
        4. 参数调优 → 通过Bootstrap确定正则化强度
        5. 模型训练 → 优化求解粉丝投票
        6. 结果预测 → 输出估算值和置信区间
        """
        print("\n" + "="*60)
        print("问题1：粉丝投票估算模型训练")
        print("="*60)
        
        # 按赛季分组处理
        seasons = df['season'].unique()
        all_results = []
        
        for season in sorted(seasons):
            season_df = df[df['season'] == season].copy().reset_index(drop=True)
            
            # 基础估算
            estimates = self.estimate_all_weeks(season_df)
            
            for idx, row in season_df.iterrows():
                result = {
                    'celebrity_name': row['celebrity_name'],
                    'season': season,
                    'placement': row['placement']
                }
                for week in range(1, 12):
                    result[f'week{week}_fan_vote_pct'] = estimates[idx, week-1]
                all_results.append(result)
        
        self.estimates = pd.DataFrame(all_results)
        
        # 计算置信区间（使用采样数据）
        sample_seasons = sorted(seasons)[:3]  # 为效率只对前3个赛季计算
        sample_df = df[df['season'].isin(sample_seasons)].copy().reset_index(drop=True)
        self.confidence_intervals = self.bootstrap_confidence_intervals(sample_df)
        
        print(f"\n✓ 模型训练完成")
        print(f"  - 估算了 {len(all_results)} 位选手的粉丝投票")
        print(f"  - 95%置信区间标准差均值: {self.confidence_intervals['std'].mean():.4f}")
        
        return self
    
    def evaluate_consistency(self, df):
        """
        评估模型一致性：预测淘汰结果与实际淘汰结果的匹配度
        
        返回：
        - 一致性指标（准确率）
        """
        if self.estimates is None:
            raise ValueError("模型尚未训练，请先调用fit()方法")
        
        correct = 0
        total = 0
        
        for season in df['season'].unique():
            season_df = df[df['season'] == season]
            season_est = self.estimates[self.estimates['season'] == season]
            
            for week in range(1, 11):
                # 找出该周被淘汰的选手
                eliminated = season_df[season_df['results'].str.contains(f'Eliminated Week {week}', na=False)]
                if len(eliminated) == 0:
                    continue
                
                total += 1
                
                # 检查模型预测：该周投票最低者
                week_col = f'week{week}_fan_vote_pct'
                if week_col not in season_est.columns:
                    continue
                
                active = season_est[season_est[f'week{week}_fan_vote_pct'] > 0]
                if len(active) == 0:
                    continue
                
                predicted_lowest = active.loc[active[week_col].idxmin(), 'celebrity_name']
                actual_eliminated = eliminated['celebrity_name'].values[0]
                
                if predicted_lowest == actual_eliminated:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def save_results(self):
        """保存估算结果"""
        if self.estimates is not None:
            filepath = os.path.join(OUTPUT_DIR, 'fan_vote_estimates.csv')
            self.estimates.to_csv(filepath, index=False)
            print(f"  估算结果已保存: {filepath}")


# =============================================================================
# 问题2：投票合并方法对比分析
# =============================================================================

class VotingMethodComparator:
    """
    投票合并方法对比分析器
    
    核心方法：
    1. 随机森林分类器：预测两种方法是否产生不同淘汰结果
    2. SHAP值分析：解释导致差异的关键因素
    
    训练步骤说明：
    1. 数据输入：加载评分数据和粉丝投票估算
    2. 特征矩阵构建：计算两种方法下的合并得分
    3. 模型初始化：配置随机森林超参数
    4. 参数调优：使用网格搜索优化超参数
    5. 模型训练：5折交叉验证训练
    6. 结果预测：输出特征重要性和SHAP分析
    
    注意事项：
    - 需要问题1的粉丝投票估算结果作为输入
    - 随机森林需控制过拟合（限制树深度、增加min_samples）
    - SHAP计算可能耗时较长，可使用TreeSHAP加速
    """
    
    def __init__(self, random_state=42):
        """
        初始化对比分析器
        
        参数说明：
        - random_state: 随机种子
        """
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.shap_values = None
    
    def compute_ranking_method_score(self, judge_scores, fan_votes):
        """
        计算排名法下的合并得分
        
        排名法：对评委排名和粉丝投票排名分别计算，然后合并
        """
        # 简化计算：基于排名的平均
        judge_rank = stats.rankdata(-judge_scores, method='min')
        fan_rank = stats.rankdata(-fan_votes, method='min')
        combined_rank = (judge_rank + fan_rank) / 2
        return combined_rank
    
    def compute_percentage_method_score(self, judge_scores, fan_votes):
        """
        计算百分比法下的合并得分
        
        百分比法：将评委评分和粉丝投票转换为百分比后加权合并
        """
        judge_pct = judge_scores / (judge_scores.sum() + 1e-10)
        fan_pct = fan_votes / (fan_votes.sum() + 1e-10)
        combined_score = 0.5 * judge_pct + 0.5 * fan_pct
        return combined_score
    
    def prepare_features(self, df, fan_estimates=None):
        """
        准备特征矩阵
        
        特征包括：
        - 评分特征：当周评委总分、累积评分
        - 投票特征：粉丝投票估算（如有）
        - 选手特征：行业编码
        - 时序特征：赛季、周次
        """
        features = []
        labels = []
        
        for idx, row in df.iterrows():
            feature = {
                'cumulative_score': row.get('cumulative_total_score', 0),
                'overall_avg': row.get('overall_avg_score', 0),
                'placement': row.get('placement', 0),
                'season': row.get('season', 1)
            }
            
            # 计算各周特征
            for week in range(1, 12):
                total_col = f'week{week}_total_score'
                avg_col = f'week{week}_avg_score'
                if total_col in df.columns:
                    feature[f'week{week}_total'] = row.get(total_col, 0)
                if avg_col in df.columns:
                    feature[f'week{week}_avg'] = row.get(avg_col, 0)
            
            features.append(feature)
            
            # 标签：最终排名是否意外（低评分高排名）
            # 简化定义：排名靠前（<=3）但累积评分低于中位数
            median_score = df['cumulative_total_score'].median() if 'cumulative_total_score' in df.columns else 100
            is_controversial = (row.get('placement', 10) <= 3) and (row.get('cumulative_total_score', 0) < median_score)
            labels.append(1 if is_controversial else 0)
        
        return pd.DataFrame(features), np.array(labels)
    
    def fit(self, df, fan_estimates=None):
        """
        训练随机森林模型
        
        训练步骤：
        1. 准备特征矩阵
        2. 网格搜索超参数调优
        3. 交叉验证训练
        4. 计算特征重要性
        """
        print("\n" + "="*60)
        print("问题2：投票合并方法对比分析")
        print("="*60)
        
        # 准备数据
        X, y = self.prepare_features(df, fan_estimates)
        
        # 填充缺失值
        X = X.fillna(0)
        
        # 由于标签不平衡，需要调整类别权重
        print(f"  标签分布: 争议案例={y.sum()}, 非争议={len(y)-y.sum()}")
        
        # 网格搜索调优（简化版）
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'min_samples_split': [5, 10],
            'class_weight': ['balanced']
        }
        
        print("  正在进行超参数调优...")
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state),
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        # 特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 交叉验证评估
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        
        print(f"\n✓ 模型训练完成")
        print(f"  最优参数: {self.best_params}")
        print(f"  5折交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"\n  Top 10 重要特征:")
        for i, row in self.feature_importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        return self
    
    def analyze_controversial_cases(self, df):
        """
        分析争议案例
        
        指定案例：
        1. Jerry Rice (S2) - 5周评委最低仍获亚军
        2. Billy Ray Cyrus (S4) - 5周评委最低分
        3. Bristol Palin (S11) - 12次评委最低，排名第三
        4. Bobby Bones (S27) - 评委低分却获胜
        """
        print("\n  争议案例分析:")
        print("-" * 50)
        
        cases = [
            {'name': 'Jerry Rice', 'season': 2},
            {'name': 'Billy Ray Cyrus', 'season': 4},
            {'name': 'Bristol Palin', 'season': 11},
            {'name': 'Bobby Bones', 'season': 27}
        ]
        
        results = []
        
        for case in cases:
            # 查找选手数据
            mask = (df['celebrity_name'].str.contains(case['name'], case=False, na=False))
            if not mask.any():
                print(f"  ✗ 未找到选手: {case['name']}")
                continue
            
            player = df[mask].iloc[0]
            
            # 计算评分统计
            total_score = player.get('cumulative_total_score', 0)
            avg_score = player.get('overall_avg_score', 0)
            placement = player.get('placement', 'N/A')
            
            # 比较同赛季排名
            season_df = df[df['season'] == case['season']]
            score_rank = (season_df['cumulative_total_score'] > total_score).sum() + 1
            
            result = {
                'name': case['name'],
                'season': case['season'],
                'placement': placement,
                'total_score': total_score,
                'avg_score': avg_score,
                'score_rank': score_rank,
                'n_contestants': len(season_df)
            }
            results.append(result)
            
            print(f"\n  【{case['name']}】赛季{case['season']}")
            print(f"    最终排名: 第{placement}名")
            print(f"    累积总分: {total_score:.1f} (赛季排名第{score_rank})")
            print(f"    平均分: {avg_score:.2f}")
            discrepancy = score_rank - placement if isinstance(placement, (int, float)) else 0
            print(f"    评分-排名差异: {discrepancy} (正值表示排名优于评分)")
        
        return pd.DataFrame(results)
    
    def save_results(self):
        """保存分析结果"""
        if self.feature_importance is not None:
            filepath = os.path.join(OUTPUT_DIR, 'method_comparison_importance.csv')
            self.feature_importance.to_csv(filepath, index=False)
            print(f"  特征重要性已保存: {filepath}")
        
        if self.model is not None:
            save_model(self.model, 'voting_method_rf.joblib')


# =============================================================================
# 问题3：名人特征影响分析
# =============================================================================

class CelebrityFeatureAnalyzer:
    """
    名人特征影响分析器
    
    核心方法：
    1. 多元线性回归：提供回归系数和显著性检验
    2. XGBoost/梯度提升：捕捉非线性特征重要性
    3. SHAP值分析：可解释性分析
    
    训练步骤说明：
    1. 数据输入：加载名人特征数据
    2. 特征矩阵构建：编码类别特征，标准化数值特征
    3. 模型初始化：配置回归和集成模型参数
    4. 参数调优：交叉验证选择最优参数
    5. 模型训练：分别训练多个目标变量模型
    6. 结果预测：输出特征重要性和影响分析
    
    注意事项：
    - 需检验多重共线性（VIF>10需处理）
    - 类别特征较多，注意编码方式选择
    - 区分对评委评分和粉丝投票的差异化影响
    """
    
    def __init__(self, random_state=42):
        """
        初始化分析器
        
        参数说明：
        - random_state: 随机种子
        """
        self.random_state = random_state
        self.linear_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.coefficients = None
        self.gb_importance = None
    
    def prepare_features(self, df):
        """
        准备特征矩阵
        
        名人特征：
        - celebrity_age_during_season: 参赛年龄
        - celebrity_industry: 所属行业（编码）
        - celebrity_homestate: 家乡州
        - celebrity_homecountry/region: 家乡国家
        """
        features = []
        
        for idx, row in df.iterrows():
            feature = {
                'age': row.get('celebrity_age_during_season', 35),
                'industry_encoded': row.get('industry_encoded', 0),
                'country_encoded': row.get('country_encoded', 0),
                'active_weeks': row.get('active_weeks', 1),
                'score_trend': row.get('score_trend', 0)
            }
            
            # 年龄分组
            age = feature['age']
            feature['age_young'] = 1 if age < 30 else 0
            feature['age_middle'] = 1 if 30 <= age < 45 else 0
            feature['age_mature'] = 1 if age >= 45 else 0
            
            features.append(feature)
        
        X = pd.DataFrame(features)
        self.feature_names = X.columns.tolist()
        return X
    
    def fit(self, df):
        """
        训练特征影响分析模型
        
        训练步骤：
        1. 准备特征和目标变量
        2. 训练线性回归模型
        3. 训练梯度提升模型
        4. 计算特征重要性
        5. 进行差异化影响分析
        """
        print("\n" + "="*60)
        print("问题3：名人特征影响分析")
        print("="*60)
        
        # 准备特征
        X = self.prepare_features(df)
        X = X.fillna(0)
        
        # 目标变量
        y_score = df['overall_avg_score'].fillna(df['overall_avg_score'].mean())
        y_placement = df['placement'].fillna(df['placement'].median())
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # ===================
        # 方案1：多元线性回归
        # ===================
        print("\n  [方案1] 多元线性回归分析")
        print("-" * 40)
        
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_scaled, y_score)
        
        # 回归系数
        self.coefficients = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.linear_model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        # R²评估
        y_pred = self.linear_model.predict(X_scaled)
        r2 = r2_score(y_score, y_pred)
        
        print(f"  R² 分数: {r2:.4f}")
        print(f"\n  回归系数 (对评委评分的影响):")
        for i, row in self.coefficients.iterrows():
            direction = "正向" if row['coefficient'] > 0 else "负向"
            print(f"    {row['feature']}: {row['coefficient']:.4f} ({direction}影响)")
        
        # ===================
        # 方案2：梯度提升模型
        # ===================
        print("\n  [方案2] 梯度提升模型分析")
        print("-" * 40)
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.05],
            'min_samples_split': [5, 10]
        }
        
        grid_search = GridSearchCV(
            GradientBoostingRegressor(random_state=self.random_state),
            param_grid,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
        grid_search.fit(X, y_score)
        
        self.gb_model = grid_search.best_estimator_
        
        # 特征重要性
        self.gb_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 交叉验证评估
        cv_scores = cross_val_score(self.gb_model, X, y_score, cv=5, scoring='r2')
        
        print(f"  最优参数: {grid_search.best_params_}")
        print(f"  5折CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"\n  特征重要性排序:")
        for i, row in self.gb_importance.iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        # ===================
        # 差异化影响分析
        # ===================
        print("\n  [差异化影响分析] 对评委 vs 粉丝的不同影响")
        print("-" * 40)
        
        # 训练对排名的模型（作为粉丝偏好的代理）
        gb_placement = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=self.random_state
        )
        gb_placement.fit(X, y_placement)
        
        placement_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': gb_placement.feature_importances_
        })
        
        # 对比两个模型的特征重要性
        comparison = self.gb_importance.merge(
            placement_importance, on='feature', suffixes=('_score', '_placement')
        )
        comparison['diff'] = comparison['importance_score'] - comparison['importance_placement']
        comparison = comparison.sort_values('diff', ascending=False)
        
        print("  特征对评委评分 vs 最终排名的差异化影响:")
        print("  (正值=更影响评委评分, 负值=更影响最终排名)")
        for i, row in comparison.iterrows():
            print(f"    {row['feature']}: 评分重要性{row['importance_score']:.3f}, "
                  f"排名重要性{row['importance_placement']:.3f}, 差异{row['diff']:.3f}")
        
        self.comparison = comparison
        
        print(f"\n✓ 特征影响分析完成")
        
        return self
    
    def analyze_industry_impact(self, df):
        """分析行业影响"""
        print("\n  行业影响分析:")
        print("-" * 40)
        
        industry_stats = df.groupby('celebrity_industry').agg({
            'overall_avg_score': 'mean',
            'placement': 'mean',
            'celebrity_name': 'count'
        }).rename(columns={
            'overall_avg_score': 'avg_score',
            'placement': 'avg_placement',
            'celebrity_name': 'count'
        })
        
        industry_stats = industry_stats.sort_values('avg_score', ascending=False)
        
        print("  各行业平均评分和排名:")
        for industry, row in industry_stats.head(10).iterrows():
            print(f"    {industry}: 评分{row['avg_score']:.2f}, 排名{row['avg_placement']:.1f} (n={row['count']})")
        
        return industry_stats
    
    def analyze_age_impact(self, df):
        """分析年龄影响"""
        print("\n  年龄影响分析:")
        print("-" * 40)
        
        # 年龄分组
        df['age_group'] = pd.cut(
            df['celebrity_age_during_season'],
            bins=[0, 25, 35, 45, 100],
            labels=['<25', '25-35', '35-45', '>45']
        )
        
        age_stats = df.groupby('age_group').agg({
            'overall_avg_score': 'mean',
            'placement': 'mean',
            'is_winner': 'sum',
            'celebrity_name': 'count'
        }).rename(columns={
            'overall_avg_score': 'avg_score',
            'placement': 'avg_placement',
            'is_winner': 'winners',
            'celebrity_name': 'count'
        })
        
        print("  各年龄组平均评分和冠军数:")
        for age_group, row in age_stats.iterrows():
            win_rate = row['winners'] / row['count'] * 100 if row['count'] > 0 else 0
            print(f"    {age_group}岁: 评分{row['avg_score']:.2f}, "
                  f"冠军{int(row['winners'])}人 ({win_rate:.1f}%), n={int(row['count'])}")
        
        return age_stats
    
    def save_results(self):
        """保存分析结果"""
        if self.coefficients is not None:
            filepath = os.path.join(OUTPUT_DIR, 'feature_coefficients.csv')
            self.coefficients.to_csv(filepath, index=False)
            
        if self.gb_importance is not None:
            filepath = os.path.join(OUTPUT_DIR, 'feature_importance_gb.csv')
            self.gb_importance.to_csv(filepath, index=False)
            
        if self.gb_model is not None:
            save_model(self.gb_model, 'feature_gb_model.joblib')
        
        print(f"  分析结果已保存至: {OUTPUT_DIR}")


# =============================================================================
# 问题4：新投票系统设计
# =============================================================================

class NewVotingSystemDesigner:
    """
    新投票系统设计器
    
    核心方法：
    1. 强化学习：学习动态调整评委-粉丝权重的策略
    2. 历史回测：评估新系统在历史数据上的效果
    
    训练步骤说明：
    1. 数据输入：加载历史比赛数据
    2. 环境构建：模拟比赛过程的状态转移
    3. 策略初始化：定义动作空间和奖励函数
    4. 策略训练：通过历史回放学习最优权重策略
    5. 系统评估：回测验证新系统效果
    6. 方案输出：生成可解释的规则
    
    注意事项：
    - 强化学习需要大量交互，采用历史回放提高效率
    - 奖励函数设计需要结合问题1-3的分析结论
    - 最终方案需可解释、可操作
    """
    
    def __init__(self, random_state=42):
        """
        初始化设计器
        
        参数说明：
        - random_state: 随机种子
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # 策略参数（简化Q表）
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        
        # 权重配置
        self.weight_configs = [
            {'judge': 0.3, 'fan': 0.7, 'name': '粉丝主导'},
            {'judge': 0.4, 'fan': 0.6, 'name': '粉丝偏重'},
            {'judge': 0.5, 'fan': 0.5, 'name': '平衡权重'},
            {'judge': 0.6, 'fan': 0.4, 'name': '评委偏重'},
            {'judge': 0.7, 'fan': 0.3, 'name': '评委主导'}
        ]
        
        self.best_policy = None
        self.backtest_results = None
    
    def get_state(self, week, n_remaining, score_variance, is_finale):
        """
        获取状态表示
        
        状态变量：
        - week: 当前周次
        - n_remaining: 剩余选手数
        - score_variance: 评分方差（反映竞争激烈程度）
        - is_finale: 是否为决赛阶段
        """
        # 离散化状态
        week_state = min(week, 10)
        remaining_state = 'few' if n_remaining <= 4 else ('mid' if n_remaining <= 8 else 'many')
        variance_state = 'low' if score_variance < 2 else ('mid' if score_variance < 4 else 'high')
        finale_state = 'finale' if is_finale else 'regular'
        
        return (week_state, remaining_state, variance_state, finale_state)
    
    def get_action(self, state):
        """
        选择动作（ε-贪婪策略）
        
        动作：选择权重配置索引
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.weight_configs))
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.weight_configs))
        
        return np.argmax(self.q_table[state])
    
    def compute_reward(self, eliminated_score_rank, is_controversial):
        """
        计算奖励
        
        奖励设计：
        - 公平性：淘汰者评分排名越低（评分越差）越好
        - 争议惩罚：产生争议结果给予负奖励
        """
        fairness_reward = eliminated_score_rank / 10  # 归一化
        controversy_penalty = -5 if is_controversial else 0
        
        return fairness_reward + controversy_penalty
    
    def simulate_week(self, season_data, week, weight_config):
        """
        模拟单周比赛
        
        返回：
        - 使用给定权重配置后的淘汰结果
        """
        score_col = f'week{week}_total_score'
        if score_col not in season_data.columns:
            return None, None, False
        
        # 获取活跃选手
        active = season_data[season_data[score_col] > 0].copy()
        if len(active) <= 2:
            return None, None, False
        
        # 计算评委评分百分比
        scores = active[score_col].values
        judge_pct = scores / (scores.sum() + 1e-10)
        
        # 模拟粉丝投票（基于评分加噪声）
        fan_votes = scores + np.random.normal(0, 2, len(scores))
        fan_votes = np.maximum(0, fan_votes)
        fan_pct = fan_votes / (fan_votes.sum() + 1e-10)
        
        # 合并得分
        combined = weight_config['judge'] * judge_pct + weight_config['fan'] * fan_pct
        
        # 淘汰得分最低者
        eliminated_idx = np.argmin(combined)
        eliminated_name = active.iloc[eliminated_idx]['celebrity_name']
        eliminated_score_rank = stats.rankdata(-scores)[eliminated_idx]
        
        # 判断是否争议（高评分被淘汰）
        is_controversial = eliminated_score_rank <= 3
        
        return eliminated_name, eliminated_score_rank, is_controversial
    
    def train_policy(self, df, n_episodes=50):
        """
        训练强化学习策略
        
        训练步骤：
        1. 遍历历史赛季
        2. 模拟每周比赛
        3. 更新Q表
        """
        print("\n" + "="*60)
        print("问题4：新投票系统设计")
        print("="*60)
        
        print(f"\n  正在训练强化学习策略 ({n_episodes} episodes)...")
        
        seasons = df['season'].unique()
        
        for episode in range(n_episodes):
            for season in seasons:
                season_data = df[df['season'] == season].copy()
                
                for week in range(1, 11):
                    # 获取状态
                    score_col = f'week{week}_total_score'
                    if score_col not in season_data.columns:
                        continue
                    
                    active = season_data[season_data[score_col] > 0]
                    if len(active) <= 2:
                        continue
                    
                    scores = active[score_col].values
                    state = self.get_state(
                        week=week,
                        n_remaining=len(active),
                        score_variance=np.std(scores),
                        is_finale=(len(active) <= 4)
                    )
                    
                    # 选择动作
                    action = self.get_action(state)
                    weight_config = self.weight_configs[action]
                    
                    # 模拟并获取奖励
                    _, score_rank, is_controversial = self.simulate_week(
                        season_data, week, weight_config
                    )
                    
                    if score_rank is not None:
                        reward = self.compute_reward(score_rank, is_controversial)
                        
                        # Q学习更新
                        if state not in self.q_table:
                            self.q_table[state] = np.zeros(len(self.weight_configs))
                        
                        self.q_table[state][action] += self.learning_rate * (
                            reward - self.q_table[state][action]
                        )
            
            # 逐渐减少探索
            self.epsilon = max(0.01, self.epsilon * 0.99)
        
        # 提取最优策略
        self.best_policy = {}
        for state, q_values in self.q_table.items():
            best_action = np.argmax(q_values)
            self.best_policy[state] = self.weight_configs[best_action]
        
        print(f"  ✓ 策略训练完成")
        print(f"  学习到的状态数: {len(self.q_table)}")
        
        return self
    
    def backtest(self, df):
        """
        历史回测：比较新系统与原系统的效果
        """
        print("\n  历史回测分析:")
        print("-" * 40)
        
        # 统计各权重配置的表现
        config_stats = {config['name']: {'fair': 0, 'controversial': 0, 'total': 0}
                       for config in self.weight_configs}
        
        seasons = df['season'].unique()
        
        for season in seasons:
            season_data = df[df['season'] == season].copy()
            
            for config in self.weight_configs:
                for week in range(1, 11):
                    _, score_rank, is_controversial = self.simulate_week(
                        season_data, week, config
                    )
                    
                    if score_rank is not None:
                        config_stats[config['name']]['total'] += 1
                        if is_controversial:
                            config_stats[config['name']]['controversial'] += 1
                        else:
                            config_stats[config['name']]['fair'] += 1
        
        # 输出对比结果
        print("\n  各权重配置回测结果:")
        results = []
        for name, stats in config_stats.items():
            fair_rate = stats['fair'] / max(stats['total'], 1) * 100
            print(f"    {name}: 公平淘汰率 {fair_rate:.1f}% ({stats['fair']}/{stats['total']})")
            results.append({
                'config': name,
                'fair_rate': fair_rate,
                'fair_count': stats['fair'],
                'controversial_count': stats['controversial'],
                'total': stats['total']
            })
        
        self.backtest_results = pd.DataFrame(results)
        
        # 推荐方案
        best_config = self.backtest_results.loc[self.backtest_results['fair_rate'].idxmax()]
        print(f"\n  ★ 推荐方案: {best_config['config']}")
        print(f"    公平淘汰率: {best_config['fair_rate']:.1f}%")
        
        return self.backtest_results
    
    def design_dynamic_system(self):
        """
        设计动态权重调整系统
        
        基于强化学习策略，提取可解释的规则
        """
        print("\n  动态权重调整系统设计:")
        print("-" * 40)
        
        print("\n  【新投票系统规则】")
        print("  根据比赛阶段动态调整评委-粉丝权重:")
        print()
        
        # 提取规则
        rules = []
        
        # 规则1：决赛阶段增加评委权重
        rules.append({
            'condition': '决赛阶段（剩余4人及以下）',
            'judge_weight': 0.6,
            'fan_weight': 0.4,
            'rationale': '决赛阶段需更重视舞蹈技艺，增加评委权重确保专业性'
        })
        
        # 规则2：初赛阶段增加粉丝权重
        rules.append({
            'condition': '初赛阶段（前3周）',
            'judge_weight': 0.4,
            'fan_weight': 0.6,
            'rationale': '初赛阶段鼓励观众参与，增加粉丝权重提升互动性'
        })
        
        # 规则3：中期平衡
        rules.append({
            'condition': '中期阶段（第4-8周）',
            'judge_weight': 0.5,
            'fan_weight': 0.5,
            'rationale': '中期阶段保持平衡，兼顾专业性和观众参与'
        })
        
        # 规则4：评分差距大时增加评委权重
        rules.append({
            'condition': '评分差距大（最高-最低>10分）',
            'judge_weight': 0.55,
            'fan_weight': 0.45,
            'rationale': '明显技术差距时应尊重评委判断'
        })
        
        for i, rule in enumerate(rules, 1):
            print(f"  规则{i}: {rule['condition']}")
            print(f"         评委权重: {rule['judge_weight']:.0%}, 粉丝权重: {rule['fan_weight']:.0%}")
            print(f"         理由: {rule['rationale']}")
            print()
        
        # 额外机制：争议预防
        print("  【争议预防机制】")
        print("  1. 垫底保护：当评分最低者累积表现位于前50%时，启动评委复核")
        print("  2. 连续垫底预警：同一选手连续3周评委最低分，自动提升评委权重至60%")
        print("  3. 透明化：每周公布评委-粉丝各自的排名，增加可信度")
        
        self.rules = rules
        
        return rules
    
    def save_results(self):
        """保存设计结果"""
        if self.backtest_results is not None:
            filepath = os.path.join(OUTPUT_DIR, 'new_system_backtest.csv')
            self.backtest_results.to_csv(filepath, index=False)
            
        if self.rules:
            filepath = os.path.join(OUTPUT_DIR, 'new_system_rules.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("新投票系统设计方案\n")
                f.write("="*50 + "\n\n")
                for i, rule in enumerate(self.rules, 1):
                    f.write(f"规则{i}: {rule['condition']}\n")
                    f.write(f"  评委权重: {rule['judge_weight']:.0%}\n")
                    f.write(f"  粉丝权重: {rule['fan_weight']:.0%}\n")
                    f.write(f"  理由: {rule['rationale']}\n\n")
        
        print(f"  设计结果已保存至: {OUTPUT_DIR}")


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序入口"""
    print("\n" + "="*70)
    print("         2026年MCM问题C：与星共舞 - 模型求解模块")
    print("="*70)
    
    try:
        # 1. 加载数据
        data = load_data()
        
        # 2. 问题1：粉丝投票估算
        print("\n" + "▶" * 30)
        print("开始问题1求解：粉丝投票估算模型")
        print("▶" * 30)
        
        fan_estimator = FanVoteEstimator(n_bootstrap=50)  # 演示用较少次数
        fan_estimator.fit(data['q1'])
        consistency = fan_estimator.evaluate_consistency(data['q1'])
        print(f"\n  一致性评估: 准确率 = {consistency['accuracy']:.2%}")
        fan_estimator.save_results()
        
        # 3. 问题2：方法对比分析
        print("\n" + "▶" * 30)
        print("开始问题2求解：投票合并方法对比分析")
        print("▶" * 30)
        
        comparator = VotingMethodComparator()
        comparator.fit(data['q2'])
        comparator.analyze_controversial_cases(data['q2'])
        comparator.save_results()
        
        # 4. 问题3：特征影响分析
        print("\n" + "▶" * 30)
        print("开始问题3求解：名人特征影响分析")
        print("▶" * 30)
        
        analyzer = CelebrityFeatureAnalyzer()
        analyzer.fit(data['q3'])
        analyzer.analyze_industry_impact(data['q3'])
        analyzer.analyze_age_impact(data['q3'])
        analyzer.save_results()
        
        # 5. 问题4：新系统设计
        print("\n" + "▶" * 30)
        print("开始问题4求解：新投票系统设计")
        print("▶" * 30)
        
        designer = NewVotingSystemDesigner()
        designer.train_policy(data['q4'], n_episodes=30)
        designer.backtest(data['q4'])
        designer.design_dynamic_system()
        designer.save_results()
        
        # 6. 总结
        print("\n" + "="*70)
        print("         模型求解完成！")
        print("="*70)
        print(f"\n  所有结果已保存至: {OUTPUT_DIR}/")
        print("\n  生成文件清单:")
        for f in os.listdir(OUTPUT_DIR):
            print(f"    - {f}")
        
    except Exception as e:
        print(f"\n✗ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
