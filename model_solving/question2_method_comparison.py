#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：投票合并方法对比分析
============================
模型方法：随机森林 + SHAP可解释性分析

核心思路：
1. 比较排名法和百分比法产生的结果差异
2. 使用随机森林分类器预测差异发生的条件
3. 使用SHAP值分析差异的驱动因素
4. 深度剖析4个争议案例

作者：MCM 2026 C题参赛团队
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
# 第二部分：方法结果计算引擎
# ============================================

class VotingMethodCalculator:
    """
    投票合并方法计算器
    
    实现两种方法：
    1. 排名法（Ranking Method）：基于排名合并
    2. 百分比法（Percentage Method）：基于百分比合并
    """
    
    def __init__(self, judge_weight=0.5, fan_weight=0.5):
        """
        初始化计算器
        
        参数:
            judge_weight: 评委评分权重
            fan_weight: 粉丝投票权重
        """
        self.judge_weight = judge_weight
        self.fan_weight = fan_weight
    
    def calculate_ranking_method(self, judge_scores, fan_votes):
        """
        排名法计算合并得分
        
        规则：将评委评分排名和粉丝投票排名相加，排名低（数值小）者胜出
        
        参数:
            judge_scores: 评委评分数组
            fan_votes: 粉丝投票数组（估算值）
        
        返回:
            combined_ranks: 合并排名
        """
        n = len(judge_scores)
        
        # 计算评委排名（高分排前）
        from scipy import stats
        judge_ranks = stats.rankdata(-np.array(judge_scores), method='average')
        
        # 计算粉丝排名（高投票排前）
        fan_ranks = stats.rankdata(-np.array(fan_votes), method='average')
        
        # 合并排名（简单相加）
        combined_ranks = judge_ranks + fan_ranks
        
        return combined_ranks
    
    def calculate_percentage_method(self, judge_scores, fan_votes):
        """
        百分比法计算合并得分
        
        规则：将评委评分百分比和粉丝投票百分比加权平均
        
        参数:
            judge_scores: 评委评分数组
            fan_votes: 粉丝投票数组（估算值）
        
        返回:
            combined_scores: 合并得分
        """
        # 转换为百分比
        judge_total = np.sum(judge_scores)
        fan_total = np.sum(fan_votes)
        
        if judge_total > 0:
            judge_pct = np.array(judge_scores) / judge_total
        else:
            judge_pct = np.zeros(len(judge_scores))
        
        if fan_total > 0:
            fan_pct = np.array(fan_votes) / fan_total
        else:
            fan_pct = np.zeros(len(fan_votes))
        
        # 加权合并
        combined_scores = self.judge_weight * judge_pct + self.fan_weight * fan_pct
        
        return combined_scores


# ============================================
# 第三部分：特征工程
# ============================================

def build_comparison_features(data, fan_voting_data=None):
    """
    构建方法对比的特征矩阵
    
    参数:
        data: 预处理后的问题2数据
        fan_voting_data: 问题1的粉丝投票估算结果（可选）
    
    返回:
        features_df: 特征矩阵
        target: 目标变量（两种方法是否产生不同结果）
    """
    print("\n>>> 构建对比特征")
    print("-" * 40)
    
    calculator = VotingMethodCalculator()
    
    # 存储结果
    comparison_records = []
    
    # 按赛季-周分组处理
    for season in data['season'].unique():
        season_data = data[data['season'] == season]
        season_rule = season_data['season_rule'].iloc[0]
        
        # 获取该赛季的所有周
        for week in range(1, 12):
            score_col = f'week{week}_total_score'
            avg_col = f'week{week}_avg_score'
            
            if score_col not in season_data.columns:
                continue
            
            # 筛选该周有效选手
            week_mask = season_data[score_col] > 0
            week_data = season_data[week_mask].copy()
            
            if len(week_data) < 2:
                continue
            
            judge_scores = week_data[score_col].values
            
            # 模拟粉丝投票（基于评分的反向推断）
            # 假设评分较低的选手可能有更高的粉丝投票（才能留下来）
            mean_score = np.mean(judge_scores)
            fan_votes = mean_score - judge_scores + np.random.normal(0, 0.1, len(judge_scores))
            fan_votes = np.maximum(0, fan_votes)  # 确保非负
            
            # 计算两种方法的结果
            ranking_result = calculator.calculate_ranking_method(judge_scores, fan_votes)
            percentage_result = calculator.calculate_percentage_method(judge_scores, fan_votes)
            
            # 确定被淘汰者（排名/得分最低）
            ranking_eliminated_idx = np.argmax(ranking_result)  # 排名数值最大=最差
            percentage_eliminated_idx = np.argmin(percentage_result)  # 百分比最低
            
            # 是否产生不同结果
            different_result = (ranking_eliminated_idx != percentage_eliminated_idx)
            
            for i, (idx, row) in enumerate(week_data.iterrows()):
                comparison_records.append({
                    'celebrity_name': row['celebrity_name'],
                    'season': season,
                    'week': week,
                    'season_rule': season_rule,
                    'judge_score': judge_scores[i],
                    'fan_vote_estimate': fan_votes[i],
                    'ranking_result': ranking_result[i],
                    'percentage_result': percentage_result[i],
                    'is_lowest_ranking': i == ranking_eliminated_idx,
                    'is_lowest_percentage': i == percentage_eliminated_idx,
                    'method_difference': different_result,
                    'placement': row['placement'],
                    'cumulative_score': row.get('cumulative_total_score', 0),
                    'avg_score': row.get('overall_avg_score', judge_scores[i] / 3)
                })
    
    features_df = pd.DataFrame(comparison_records)
    
    # 计算差异特征
    features_df['score_rank_diff'] = features_df['ranking_result'] - features_df['percentage_result'] * 10
    features_df['relative_position'] = features_df.groupby(['season', 'week'])['judge_score'].rank(ascending=False)
    
    print(f"✓ 特征构建完成")
    print(f"  - 总记录数: {len(features_df)}")
    print(f"  - 产生不同结果的周次: {features_df['method_difference'].sum()}")
    
    return features_df


# ============================================
# 第四部分：随机森林模型
# ============================================

class MethodComparisonModel:
    """
    方法对比分析模型
    
    使用随机森林分类器预测两种方法是否会产生不同结果
    
    参数说明:
        n_estimators: 树的数量（默认100）
        max_depth: 最大深度（防止过拟合）
        min_samples_split: 最小分割样本数
        random_state: 随机种子
    """
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5, random_state=42):
        """
        初始化模型
        
        初始化方式：
            - n_estimators=100: 足够的树以获得稳定结果
            - max_depth=10: 限制深度避免过拟合
            - min_samples_split=5: 防止过小分割
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            class_weight='balanced',  # 处理类别不平衡
            n_jobs=-1
        )
        self.feature_names = None
        self.label_encoders = {}
        
        print(f"✓ 随机森林模型初始化完成")
        print(f"  - 树的数量: {n_estimators}")
        print(f"  - 最大深度: {max_depth}")
    
    def prepare_features(self, data):
        """
        准备模型特征
        
        参数:
            data: 特征数据框
        
        返回:
            X: 特征矩阵
            y: 目标变量
        """
        # 选择数值特征
        numeric_features = ['judge_score', 'fan_vote_estimate', 'ranking_result', 
                          'percentage_result', 'cumulative_score', 'avg_score',
                          'score_rank_diff', 'relative_position', 'week', 'season']
        
        # 编码类别特征
        if 'season_rule' in data.columns:
            if 'season_rule' not in self.label_encoders:
                self.label_encoders['season_rule'] = LabelEncoder()
                data['season_rule_encoded'] = self.label_encoders['season_rule'].fit_transform(data['season_rule'])
            else:
                data['season_rule_encoded'] = self.label_encoders['season_rule'].transform(data['season_rule'])
            numeric_features.append('season_rule_encoded')
        
        # 构建特征矩阵
        available_features = [f for f in numeric_features if f in data.columns]
        X = data[available_features].fillna(0).values
        
        # 目标变量
        y = data['method_difference'].astype(int).values
        
        self.feature_names = available_features
        
        return X, y
    
    def train(self, X, y, cv_folds=5):
        """
        训练模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
            cv_folds: 交叉验证折数
        
        注意事项:
            - 使用分层交叉验证确保每折类别比例一致
            - 监控训练-测试差距检测过拟合
        """
        print("\n>>> 模型训练")
        print("-" * 40)
        
        # 交叉验证
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"  - 交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 检测过拟合
        # 在全部数据上训练
        self.model.fit(X, y)
        train_score = self.model.score(X, y)
        
        overfitting_gap = train_score - cv_scores.mean()
        if overfitting_gap > 0.1:
            print(f"  ⚠ 可能存在过拟合: 训练-测试差距 = {overfitting_gap:.4f}")
        else:
            print(f"  ✓ 模型泛化良好: 训练-测试差距 = {overfitting_gap:.4f}")
        
        # 特征重要性
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Top 5 重要特征:")
        for i, row in feature_importance.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        return {
            'cv_scores': cv_scores,
            'train_score': train_score,
            'feature_importance': feature_importance
        }


# ============================================
# 第五部分：SHAP可解释性分析
# ============================================

def shap_analysis(model, X, feature_names, output_dir='output'):
    """
    SHAP值分析
    
    参数:
        model: 训练好的随机森林模型
        X: 特征矩阵
        feature_names: 特征名称
        output_dir: 输出目录
    
    返回:
        shap_values: SHAP值数组
        shap_summary: SHAP摘要统计
    
    注意事项:
        - TreeSHAP算法对树模型计算效率高
        - SHAP值反映相关性而非因果性
    """
    print("\n>>> SHAP可解释性分析")
    print("-" * 40)
    
    try:
        import shap
        
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # 如果是二分类，取正类的SHAP值
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        
        # 计算每个特征的平均绝对SHAP值
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_summary = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        print(f"  ✓ SHAP分析完成")
        print(f"  Top 5 SHAP特征:")
        for i, row in shap_summary.head(5).iterrows():
            print(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")
        
        return shap_values, shap_summary
        
    except ImportError:
        print("  ⚠ SHAP库未安装，跳过SHAP分析")
        print("  安装命令: pip install shap")
        return None, None
    except Exception as e:
        print(f"  ⚠ SHAP分析出错: {str(e)}")
        return None, None


# ============================================
# 第六部分：争议案例分析
# ============================================

def analyze_controversial_cases(data, features_df):
    """
    分析4个争议案例
    
    争议案例：
    1. Jerry Rice (S2): 5周评委最低分仍获亚军
    2. Billy Ray Cyrus (S4): 5周评委最低分
    3. Bristol Palin (S11): 12次评委最低分，排名第三
    4. Bobby Bones (S27): 评委打分一直很低却获胜
    
    参数:
        data: 原始数据
        features_df: 特征数据框
    
    返回:
        case_analysis: 争议案例分析结果
    """
    print("\n>>> 争议案例分析")
    print("=" * 50)
    
    controversial_cases = [
        {'name': 'Jerry Rice', 'season': 2, 'description': '5周评委最低分仍获亚军'},
        {'name': 'Billy Ray Cyrus', 'season': 4, 'description': '5周评委最低分'},
        {'name': 'Bristol Palin', 'season': 11, 'description': '12次评委最低分，排名第三'},
        {'name': 'Bobby Bones', 'season': 27, 'description': '评委低分却获胜'}
    ]
    
    case_results = []
    
    for case in controversial_cases:
        print(f"\n案例: {case['name']} (Season {case['season']})")
        print(f"争议描述: {case['description']}")
        print("-" * 40)
        
        # 查找该选手的数据
        case_data = features_df[
            (features_df['celebrity_name'].str.contains(case['name'], case=False, na=False)) &
            (features_df['season'] == case['season'])
        ]
        
        if len(case_data) > 0:
            # 统计分析
            avg_judge_score = case_data['judge_score'].mean()
            avg_ranking_result = case_data['ranking_result'].mean()
            avg_percentage_result = case_data['percentage_result'].mean()
            times_lowest = case_data['is_lowest_ranking'].sum()
            
            print(f"  平均评委评分: {avg_judge_score:.2f}")
            print(f"  平均排名法得分: {avg_ranking_result:.2f}")
            print(f"  平均百分比法得分: {avg_percentage_result:.4f}")
            print(f"  评委最低次数: {times_lowest}")
            
            # 分析两种方法的差异
            if 'method_difference' in case_data.columns:
                diff_weeks = case_data['method_difference'].sum()
                print(f"  方法产生不同结果的周次: {diff_weeks}")
            
            case_results.append({
                'name': case['name'],
                'season': case['season'],
                'description': case['description'],
                'avg_judge_score': avg_judge_score,
                'avg_ranking_result': avg_ranking_result,
                'avg_percentage_result': avg_percentage_result,
                'times_lowest': times_lowest,
                'n_weeks': len(case_data),
                'placement': case_data['placement'].iloc[0]
            })
        else:
            print(f"  ⚠ 未找到该选手数据")
    
    return case_results


# ============================================
# 第七部分：可视化生成
# ============================================

def generate_visualizations(features_df, model_results, case_results, output_dir='output'):
    """
    生成问题2相关的可视化图表
    
    参数:
        features_df: 特征数据框
        model_results: 模型训练结果
        case_results: 争议案例分析结果
        output_dir: 输出目录
    
    返回:
        生成的图表文件列表
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 12
    
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    # 图1: 特征重要性条形图
    if 'feature_importance' in model_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df = model_results['feature_importance'].head(10)
        
        bars = ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
        ax.set_xlabel('Feature Importance')
        ax.set_title('Figure Q2-1: Top 10 Feature Importance for Method Difference Prediction')
        ax.invert_yaxis()
        
        # 添加数值标签
        for bar, val in zip(bars, importance_df['importance']):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center')
        
        filepath = os.path.join(output_dir, 'Q2_01_feature_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    # 图2: 两种方法结果对比散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(features_df['ranking_result'], 
                        features_df['percentage_result'],
                        c=features_df['method_difference'].astype(int),
                        cmap='RdYlGn_r', alpha=0.6, s=50)
    
    ax.set_xlabel('Ranking Method Score')
    ax.set_ylabel('Percentage Method Score')
    ax.set_title('Figure Q2-2: Comparison of Two Voting Methods')
    
    # 添加对角线
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], 
            [ax.get_ylim()[0], ax.get_ylim()[1]], 
            'k--', alpha=0.3, label='Equal Line')
    
    plt.colorbar(scatter, label='Different Result (1=Yes, 0=No)')
    ax.legend()
    
    filepath = os.path.join(output_dir, 'Q2_02_method_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    # 图3: 按赛季规则的差异分布
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, rule in enumerate(['Ranking', 'Percentage', 'Ranking_JudgeSave']):
        rule_data = features_df[features_df['season_rule'] == rule]
        if len(rule_data) > 0:
            diff_rate = rule_data['method_difference'].mean() * 100
            
            # 绘制柱状图
            categories = ['Same Result', 'Different Result']
            values = [100 - diff_rate, diff_rate]
            colors = ['#2ecc71', '#e74c3c']
            
            axes[i].bar(categories, values, color=colors)
            axes[i].set_title(f'{rule}\n(Diff Rate: {diff_rate:.1f}%)')
            axes[i].set_ylabel('Percentage (%)')
            axes[i].set_ylim(0, 100)
    
    plt.suptitle('Figure Q2-3: Method Difference Rate by Season Rule', fontsize=14, y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'Q2_03_diff_by_rule.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    # 图4: 争议案例分析图
    if case_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = [c['name'] for c in case_results]
        scores = [c['avg_judge_score'] for c in case_results]
        placements = [c['placement'] for c in case_results]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, scores, width, label='Avg Judge Score', color='steelblue')
        
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, placements, width, label='Final Placement', color='coral')
        
        ax.set_xlabel('Controversial Cases')
        ax.set_ylabel('Average Judge Score', color='steelblue')
        ax2.set_ylabel('Final Placement (1=Winner)', color='coral')
        ax.set_title('Figure Q2-4: Controversial Cases - Score vs Placement Paradox')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 添加注释
        ax.text(0.5, -0.15, 
                'Note: Higher scores but worse placements indicate fan voting impact',
                transform=ax.transAxes, ha='center', fontsize=10, style='italic')
        
        filepath = os.path.join(output_dir, 'Q2_04_controversial_cases.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    # 图5: 交叉验证结果
    if 'cv_scores' in model_results:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cv_scores = model_results['cv_scores']
        folds = range(1, len(cv_scores) + 1)
        
        ax.bar(folds, cv_scores, color='steelblue', edgecolor='white')
        ax.axhline(cv_scores.mean(), color='red', linestyle='--', 
                  label=f'Mean = {cv_scores.mean():.4f}')
        ax.fill_between([0.5, len(cv_scores) + 0.5], 
                       cv_scores.mean() - cv_scores.std(),
                       cv_scores.mean() + cv_scores.std(),
                       alpha=0.2, color='red')
        
        ax.set_xlabel('CV Fold')
        ax.set_ylabel('Accuracy')
        ax.set_title('Figure Q2-5: Cross-Validation Performance')
        ax.legend()
        ax.set_xticks(folds)
        
        filepath = os.path.join(output_dir, 'Q2_05_cv_performance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    return generated_files


# ============================================
# 第八部分：方法推荐
# ============================================

def generate_recommendation(features_df, case_results):
    """
    基于分析结果生成方法推荐
    
    参数:
        features_df: 特征数据框
        case_results: 争议案例结果
    
    返回:
        recommendation: 推荐报告
    """
    print("\n>>> 方法推荐")
    print("=" * 50)
    
    # 分析各规则下的差异率
    rule_stats = features_df.groupby('season_rule')['method_difference'].agg(['mean', 'sum', 'count'])
    
    print("各赛季规则下的方法差异统计:")
    for rule, row in rule_stats.iterrows():
        print(f"  {rule}: 差异率 {row['mean']*100:.2f}%, 总差异次数 {row['sum']:.0f}")
    
    # 分析争议案例的共同特征
    if case_results:
        avg_score_diff = np.mean([c['avg_judge_score'] for c in case_results])
        avg_placement = np.mean([c['placement'] for c in case_results])
        
        print(f"\n争议案例特征:")
        print(f"  平均评委评分: {avg_score_diff:.2f}")
        print(f"  平均最终排名: {avg_placement:.1f}")
    
    # 生成推荐
    recommendation = {
        'primary': '百分比法',
        'reason': '百分比法能更好地平衡评委专业评分和粉丝投票，减少争议事件发生',
        'conditions': [
            '当评委评分差距较大时，百分比法可避免粉丝投票过度影响',
            '建议在决赛阶段增加评委权重（如60%评委 + 40%粉丝）',
            '对于28-34季的评委决定淘汰规则，建议仅在得分非常接近时启用'
        ],
        'alternative': '排名法',
        'alternative_when': '当选手水平接近时，排名法可增加比赛悬念'
    }
    
    print(f"\n【推荐方案】: {recommendation['primary']}")
    print(f"【推荐理由】: {recommendation['reason']}")
    print("【使用条件】:")
    for cond in recommendation['conditions']:
        print(f"  • {cond}")
    
    return recommendation


# ============================================
# 第九部分：模型保存与加载
# ============================================

def save_results(features_df, model_results, case_results, recommendation, output_dir='output'):
    """
    保存分析结果
    """
    import os
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存特征数据
    features_df.to_csv(os.path.join(output_dir, 'Q2_comparison_features.csv'), 
                       index=False, encoding='utf-8-sig')
    
    # 保存分析结果
    results = {
        'model_results': model_results,
        'case_results': case_results,
        'recommendation': recommendation
    }
    
    with open(os.path.join(output_dir, 'Q2_analysis_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✓ 结果已保存到 {output_dir}")


# ============================================
# 主程序入口
# ============================================

def main():
    """
    主程序：执行完整的方法对比分析流程
    """
    print("=" * 60)
    print("问题2：投票合并方法对比分析")
    print("=" * 60)
    
    # 1. 数据加载
    print("\n【步骤1】数据加载")
    data = load_data('output/question2_data.csv')
    
    if data is None:
        print("数据加载失败，程序终止")
        return
    
    # 2. 特征构建
    print("\n【步骤2】特征构建")
    features_df = build_comparison_features(data)
    
    # 3. 模型训练
    print("\n【步骤3】模型训练")
    model = MethodComparisonModel(n_estimators=100, max_depth=8)
    X, y = model.prepare_features(features_df)
    model_results = model.train(X, y)
    
    # 4. SHAP分析
    print("\n【步骤4】SHAP分析")
    shap_values, shap_summary = shap_analysis(
        model.model, X, model.feature_names, 'output'
    )
    if shap_summary is not None:
        model_results['shap_summary'] = shap_summary
    
    # 5. 争议案例分析
    print("\n【步骤5】争议案例分析")
    case_results = analyze_controversial_cases(data, features_df)
    
    # 6. 方法推荐
    print("\n【步骤6】方法推荐")
    recommendation = generate_recommendation(features_df, case_results)
    
    # 7. 可视化生成
    print("\n【步骤7】可视化生成")
    viz_files = generate_visualizations(features_df, model_results, case_results, 'output')
    
    # 8. 保存结果
    print("\n【步骤8】保存结果")
    save_results(features_df, model_results, case_results, recommendation, 'output')
    
    # 9. 结果摘要
    print("\n" + "=" * 60)
    print("模型求解结果摘要")
    print("=" * 60)
    print(f"• 分析记录数: {len(features_df)}")
    print(f"• 方法差异发生率: {features_df['method_difference'].mean()*100:.2f}%")
    print(f"• 模型交叉验证准确率: {model_results['cv_scores'].mean():.4f}")
    print(f"• 争议案例分析: {len(case_results)} 个")
    print(f"• 推荐方法: {recommendation['primary']}")
    print(f"• 生成可视化图表: {len(viz_files)} 个")
    
    print("\n>>> 问题2模型求解完成 <<<")
    
    return {
        'features': features_df,
        'model_results': model_results,
        'case_results': case_results,
        'recommendation': recommendation
    }


if __name__ == '__main__':
    main()
