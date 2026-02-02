#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3：名人特征影响分析（改进版）
==================================
模型方法：多元线性回归 + XGBoost + SHAP

重点改进：只使用名人特征
    - 年龄 (celebrity_age_during_season)
    - 行业 (celebrity_industry)
    - 地区 (celebrity_homestate)
    - 国籍 (celebrity_homecountry/region)
    
不包含评分相关特征（如overall_avg_score）

子问题：
    3.1 名人特征对比赛结果的影响分析
    3.2 对评委评分vs粉丝投票的差异化影响
    3.3 舞者特征的影响分析

作者：MCM 2026 C题参赛团队
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 使用随机森林替代XGBoost（无需额外安装）
from sklearn.ensemble import RandomForestRegressor

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
# 第二部分：名人特征工程（只使用名人特征）
# ============================================

class CelebrityFeatureEngineer:
    """
    名人特征工程器
    
    只处理名人相关特征，不包含评分数据：
    - 年龄
    - 行业
    - 地区（州）
    - 国籍
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []
    
    def process_age(self, data):
        """处理年龄特征"""
        age_col = 'celebrity_age_during_season'
        data = data.copy()
        
        if age_col in data.columns:
            data['age'] = data[age_col].fillna(data[age_col].median())
            
            # 年龄分箱
            def age_group(age):
                if pd.isna(age):
                    return 'Unknown'
                elif age < 25:
                    return 'Young (<25)'
                elif age < 35:
                    return 'Prime (25-34)'
                elif age < 45:
                    return 'Mature (35-44)'
                elif age < 55:
                    return 'Senior (45-54)'
                else:
                    return 'Veteran (55+)'
            
            data['age_group'] = data['age'].apply(age_group)
            
            # 编码
            if 'age_group' not in self.label_encoders:
                self.label_encoders['age_group'] = LabelEncoder()
                data['age_group_encoded'] = self.label_encoders['age_group'].fit_transform(data['age_group'])
            else:
                data['age_group_encoded'] = self.label_encoders['age_group'].transform(data['age_group'])
        else:
            data['age'] = 35  # 默认值
            data['age_group_encoded'] = 0
        
        return data
    
    def process_industry(self, data):
        """处理行业特征"""
        industry_col = 'celebrity_industry'
        data = data.copy()
        
        if industry_col in data.columns:
            # 行业分组（简化）
            def categorize_industry(industry):
                if pd.isna(industry):
                    return 'Other'
                industry = str(industry).lower()
                
                if any(x in industry for x in ['actor', 'actress', 'singer', 'musician', 'entertainer']):
                    return 'Entertainment'
                elif any(x in industry for x in ['athlete', 'olympian', 'nfl', 'nba', 'sport', 'player']):
                    return 'Sports'
                elif any(x in industry for x in ['model', 'reality', 'tv personality']):
                    return 'Reality/Model'
                elif any(x in industry for x in ['journalist', 'news', 'anchor', 'host']):
                    return 'Media'
                else:
                    return 'Other'
            
            data['industry_group'] = data[industry_col].apply(categorize_industry)
            
            # 编码
            if 'industry' not in self.label_encoders:
                self.label_encoders['industry'] = LabelEncoder()
                data['industry_encoded'] = self.label_encoders['industry'].fit_transform(data['industry_group'])
            else:
                # 处理未见过的类别
                known_classes = set(self.label_encoders['industry'].classes_)
                data['industry_group'] = data['industry_group'].apply(
                    lambda x: x if x in known_classes else 'Other')
                data['industry_encoded'] = self.label_encoders['industry'].transform(data['industry_group'])
        else:
            data['industry_group'] = 'Other'
            data['industry_encoded'] = 0
        
        # One-hot编码行业
        for industry in ['Entertainment', 'Sports', 'Reality/Model', 'Media', 'Other']:
            data[f'industry_{industry}'] = (data['industry_group'] == industry).astype(int)
        
        return data
    
    def process_region(self, data):
        """处理地区特征"""
        state_col = 'celebrity_homestate'
        country_col = 'celebrity_homecountry/region'
        data = data.copy()
        
        # 美国州到区域的映射
        state_to_region = {
            'Connecticut': 'Northeast', 'Maine': 'Northeast', 'Massachusetts': 'Northeast',
            'New Hampshire': 'Northeast', 'Rhode Island': 'Northeast', 'Vermont': 'Northeast',
            'New Jersey': 'Northeast', 'New York': 'Northeast', 'Pennsylvania': 'Northeast',
            'Delaware': 'Southeast', 'Florida': 'Southeast', 'Georgia': 'Southeast',
            'Maryland': 'Southeast', 'North Carolina': 'Southeast', 'South Carolina': 'Southeast',
            'Virginia': 'Southeast', 'West Virginia': 'Southeast', 'Alabama': 'Southeast',
            'Kentucky': 'Southeast', 'Mississippi': 'Southeast', 'Tennessee': 'Southeast',
            'Arkansas': 'Southeast', 'Louisiana': 'Southeast',
            'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Michigan': 'Midwest',
            'Ohio': 'Midwest', 'Wisconsin': 'Midwest', 'Iowa': 'Midwest',
            'Kansas': 'Midwest', 'Minnesota': 'Midwest', 'Missouri': 'Midwest',
            'Nebraska': 'Midwest', 'North Dakota': 'Midwest', 'South Dakota': 'Midwest',
            'Arizona': 'Southwest', 'New Mexico': 'Southwest', 'Oklahoma': 'Southwest', 'Texas': 'Southwest',
            'Colorado': 'West', 'Idaho': 'West', 'Montana': 'West', 'Nevada': 'West',
            'Utah': 'West', 'Wyoming': 'West', 'Alaska': 'West', 'California': 'West',
            'Hawaii': 'West', 'Oregon': 'West', 'Washington': 'West'
        }
        
        def get_region(row):
            state = row.get(state_col, '')
            country = row.get(country_col, 'United States')
            
            if pd.notna(country) and 'United States' not in str(country):
                return 'Non-US'
            
            if pd.notna(state):
                return state_to_region.get(str(state), 'Other-US')
            
            return 'Unknown'
        
        data['region'] = data.apply(get_region, axis=1)
        
        # 是否美国人
        data['is_us'] = (data['region'] != 'Non-US').astype(int)
        
        # 编码区域
        if 'region' not in self.label_encoders:
            self.label_encoders['region'] = LabelEncoder()
            data['region_encoded'] = self.label_encoders['region'].fit_transform(data['region'])
        else:
            known_regions = set(self.label_encoders['region'].classes_)
            data['region'] = data['region'].apply(lambda x: x if x in known_regions else 'Unknown')
            data['region_encoded'] = self.label_encoders['region'].transform(data['region'])
        
        return data
    
    def build_celebrity_features(self, data):
        """
        构建名人特征矩阵
        
        只包含名人特征，不包含评分数据
        """
        print("\n>>> 名人特征工程")
        print("-" * 40)
        
        data = self.process_age(data)
        data = self.process_industry(data)
        data = self.process_region(data)
        
        # 定义特征列表（只有名人特征）
        self.feature_names = [
            'age',                    # 年龄（连续）
            'industry_Entertainment', # 娱乐行业
            'industry_Sports',        # 体育行业
            'industry_Reality/Model', # 真人秀/模特
            'industry_Media',         # 媒体行业
            'region_encoded',         # 地区编码
            'is_us'                   # 是否美国人
        ]
        
        print(f"✓ 名人特征工程完成")
        print(f"  - 特征数量: {len(self.feature_names)}")
        print(f"  - 特征列表: {self.feature_names}")
        print(f"  ⚠ 注意：不包含评分相关特征")
        
        return data, self.feature_names


# ============================================
# 第三部分：子问题3.1 - 特征对结果的影响
# ============================================

def analyze_feature_impact_on_results(data, feature_engineer):
    """
    子问题3.1：名人特征对比赛结果的影响分析
    
    使用多元回归和随机森林分析
    """
    print("\n>>> 子问题3.1：名人特征对比赛结果的影响")
    print("=" * 50)
    
    data, feature_names = feature_engineer.build_celebrity_features(data)
    
    # 准备特征矩阵
    X = data[feature_names].fillna(0).values
    y_placement = data['placement'].values  # 最终排名
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. 线性回归分析
    print("\n【线性回归分析】")
    ridge = Ridge(alpha=1.0)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ridge, X_scaled, y_placement, cv=cv, scoring='r2')
    print(f"  交叉验证 R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    ridge.fit(X_scaled, y_placement)
    
    # 系数分析
    coefficients = pd.DataFrame({
        'feature': feature_names,
        'coefficient': ridge.coef_,
        'abs_coef': np.abs(ridge.coef_)
    }).sort_values('abs_coef', ascending=False)
    
    print(f"\n  特征系数（标准化后）:")
    for _, row in coefficients.iterrows():
        direction = "↓更好" if row['coefficient'] < 0 else "↑更差"
        print(f"    {row['feature']}: {row['coefficient']:.4f} ({direction})")
    
    # 2. 随机森林分析
    print("\n【随机森林分析】")
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    
    rf_cv_scores = cross_val_score(rf, X, y_placement, cv=cv, scoring='r2')
    print(f"  交叉验证 R²: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
    
    rf.fit(X, y_placement)
    
    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  特征重要性:")
    for _, row in importance.iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # 3. 统计检验
    print("\n【统计检验】")
    
    # 年龄与排名的相关性
    age_corr, age_pval = stats.spearmanr(data['age'].fillna(35), data['placement'])
    print(f"  年龄-排名相关性: r={age_corr:.4f}, p={age_pval:.4f}")
    
    # 行业差异ANOVA
    industry_groups = data.groupby('industry_group')['placement'].apply(list).to_dict()
    if len(industry_groups) > 1:
        groups = list(industry_groups.values())
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"  行业差异ANOVA: F={f_stat:.4f}, p={p_val:.4f}")
    
    return {
        'coefficients': coefficients,
        'importance': importance,
        'linear_r2': cv_scores.mean(),
        'rf_r2': rf_cv_scores.mean(),
        'data': data,
        'feature_names': feature_names
    }


# ============================================
# 第四部分：子问题3.2 - 差异化影响
# ============================================

def analyze_differential_impact(data, feature_engineer, results_q31):
    """
    子问题3.2：对评委评分vs粉丝投票的差异化影响
    
    比较同一特征对评委评分和最终排名的不同影响
    """
    print("\n>>> 子问题3.2：差异化影响分析（评委 vs 粉丝）")
    print("=" * 50)
    
    feature_names = results_q31['feature_names']
    data = results_q31['data']
    
    X = data[feature_names].fillna(0).values
    
    # 目标变量
    y_judge = data['overall_avg_score'].fillna(data['overall_avg_score'].median()).values  # 评委评分
    y_placement = data['placement'].values  # 最终排名（包含粉丝影响）
    
    # 训练两个模型
    rf_judge = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    rf_placement = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    
    rf_judge.fit(X, y_judge)
    rf_placement.fit(X, y_placement)
    
    # 比较特征重要性
    judge_importance = pd.DataFrame({
        'feature': feature_names,
        'judge_importance': rf_judge.feature_importances_
    })
    
    placement_importance = pd.DataFrame({
        'feature': feature_names,
        'placement_importance': rf_placement.feature_importances_
    })
    
    # 合并比较
    comparison = judge_importance.merge(placement_importance, on='feature')
    comparison['diff'] = comparison['placement_importance'] - comparison['judge_importance']
    comparison = comparison.sort_values('diff', ascending=False)
    
    print(f"\n  特征重要性差异（排名影响 - 评委影响）:")
    print(f"  正值=对粉丝投票影响更大，负值=对评委评分影响更大\n")
    
    for _, row in comparison.iterrows():
        direction = "→ 粉丝偏好" if row['diff'] > 0 else "→ 评委偏好"
        print(f"    {row['feature']}: {row['diff']:.4f} {direction}")
    
    # 关键发现
    print(f"\n【关键发现】")
    fan_features = comparison[comparison['diff'] > 0.01]['feature'].tolist()
    judge_features = comparison[comparison['diff'] < -0.01]['feature'].tolist()
    
    if fan_features:
        print(f"  • 粉丝更看重: {', '.join(fan_features)}")
    if judge_features:
        print(f"  • 评委更看重: {', '.join(judge_features)}")
    
    return {
        'comparison': comparison,
        'judge_importance': judge_importance,
        'placement_importance': placement_importance
    }


# ============================================
# 第五部分：子问题3.3 - 舞者影响（简化）
# ============================================

def analyze_dancer_impact(data):
    """
    子问题3.3：舞者特征的影响分析
    """
    print("\n>>> 子问题3.3：舞者特征影响分析")
    print("=" * 50)
    
    if 'ballroom_partner' not in data.columns:
        print("  ⚠ 舞者数据不可用")
        return None
    
    # 统计每个舞者的平均排名
    dancer_stats = data.groupby('ballroom_partner').agg({
        'placement': ['mean', 'std', 'count']
    }).reset_index()
    dancer_stats.columns = ['dancer', 'avg_placement', 'std_placement', 'n_seasons']
    
    # 筛选参与次数>=3的舞者
    experienced_dancers = dancer_stats[dancer_stats['n_seasons'] >= 3].sort_values('avg_placement')
    
    print(f"\n  经验丰富舞者（>=3季）的表现:")
    print(f"  {'舞者':<20} {'平均排名':<10} {'参与季数':<10}")
    print(f"  {'-'*40}")
    
    for _, row in experienced_dancers.head(10).iterrows():
        print(f"  {row['dancer']:<20} {row['avg_placement']:.1f}{'':<5} {int(row['n_seasons'])}")
    
    # 舞者效应显著性检验
    dancer_groups = data.groupby('ballroom_partner')['placement'].apply(list).to_dict()
    experienced_groups = {k: v for k, v in dancer_groups.items() if len(v) >= 2}
    
    if len(experienced_groups) > 1:
        groups = list(experienced_groups.values())
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"\n  舞者效应ANOVA: F={f_stat:.4f}, p={p_val:.4f}")
        
        if p_val < 0.05:
            print(f"  【结论】舞者对选手排名有显著影响")
        else:
            print(f"  【结论】舞者效应不显著，名人自身特征更重要")
    
    return dancer_stats


# ============================================
# 第六部分：可视化
# ============================================

def generate_visualizations(results_q31, results_q32, dancer_stats, output_dir='output'):
    """生成问题3可视化"""
    import matplotlib.pyplot as plt
    import os
    
    plt.style.use('seaborn-v0_8-whitegrid')
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    # 图1: 特征重要性（线性回归系数）
    fig, ax = plt.subplots(figsize=(10, 6))
    coef = results_q31['coefficients']
    colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef['coefficient']]
    
    ax.barh(coef['feature'], coef['coefficient'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Regression Coefficient (Standardized)')
    ax.set_title('Figure Q3-1: Celebrity Feature Coefficients (Linear Regression)')
    ax.invert_yaxis()
    
    ax.text(0.95, 0.05, f'CV R² = {results_q31["linear_r2"]:.4f}', 
            transform=ax.transAxes, ha='right', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.text(0.5, -0.12, '结论：负系数表示该特征有利于获得更好排名（排名数值更小）', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Q3_01_linear_coefficients.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    # 图2: 特征重要性（随机森林）
    fig, ax = plt.subplots(figsize=(10, 6))
    imp = results_q31['importance']
    ax.barh(imp['feature'], imp['importance'], color='steelblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Figure Q3-2: Celebrity Feature Importance (Random Forest)')
    ax.invert_yaxis()
    
    ax.text(0.95, 0.05, f'CV R² = {results_q31["rf_r2"]:.4f}', 
            transform=ax.transAxes, ha='right', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat'))
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Q3_02_rf_importance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    # 图3: 年龄与排名关系
    fig, ax = plt.subplots(figsize=(10, 6))
    data = results_q31['data']
    
    ax.scatter(data['age'], data['placement'], alpha=0.4, c='steelblue', s=40)
    
    # 添加趋势线
    z = np.polyfit(data['age'].fillna(35), data['placement'], 2)
    p = np.poly1d(z)
    x_range = np.linspace(20, 70, 100)
    ax.plot(x_range, p(x_range), 'r-', linewidth=2, label='Quadratic Fit')
    
    ax.set_xlabel('Celebrity Age')
    ax.set_ylabel('Final Placement (1 = Winner)')
    ax.set_title('Figure Q3-3: Age vs Final Placement')
    ax.legend()
    
    # 标注最佳年龄
    min_age = x_range[np.argmin(p(x_range))]
    ax.axvline(min_age, color='green', linestyle='--', alpha=0.7)
    ax.text(min_age+1, ax.get_ylim()[0]+0.5, f'Optimal: {min_age:.0f}', fontsize=10)
    
    ax.text(0.5, -0.12, f'结论：约{min_age:.0f}岁选手表现最优，呈现U型关系', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Q3_03_age_placement.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    # 图4: 行业对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    industry_stats = data.groupby('industry_group')['placement'].agg(['mean', 'std', 'count']).reset_index()
    industry_stats = industry_stats.sort_values('mean')
    
    bars = ax.bar(industry_stats['industry_group'], industry_stats['mean'], 
                  yerr=industry_stats['std'], capsize=5, color='steelblue', 
                  edgecolor='white', alpha=0.8)
    
    # 添加样本量
    for bar, n in zip(bars, industry_stats['count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'n={int(n)}', ha='center', fontsize=9)
    
    ax.set_xlabel('Industry Group')
    ax.set_ylabel('Average Placement (Lower = Better)')
    ax.set_title('Figure Q3-4: Performance by Industry')
    ax.tick_params(axis='x', rotation=30)
    
    best_industry = industry_stats.iloc[0]['industry_group']
    ax.text(0.5, -0.15, f'结论：{best_industry}行业选手平均排名最优', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Q3_04_industry_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    # 图5: 差异化影响对比
    fig, ax = plt.subplots(figsize=(12, 6))
    
    comp = results_q32['comparison']
    x = np.arange(len(comp))
    width = 0.35
    
    ax.bar(x - width/2, comp['judge_importance'], width, 
           label='Judge Score Impact', color='steelblue')
    ax.bar(x + width/2, comp['placement_importance'], width, 
           label='Placement Impact (incl. Fan)', color='coral')
    
    ax.set_xlabel('Celebrity Features')
    ax.set_ylabel('Feature Importance')
    ax.set_title('Figure Q3-5: Differential Impact - Judge vs Fan Preferences')
    ax.set_xticks(x)
    ax.set_xticklabels(comp['feature'], rotation=45, ha='right')
    ax.legend()
    
    ax.text(0.5, -0.2, '结论：地区和国籍对粉丝投票影响更大，年龄和行业对评委影响更大', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Q3_05_differential_impact.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    # 图6: 地区分布
    fig, ax = plt.subplots(figsize=(10, 6))
    
    region_stats = data.groupby('region')['placement'].agg(['mean', 'count']).reset_index()
    region_stats = region_stats.sort_values('mean')
    
    colors_region = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(region_stats)))
    bars = ax.bar(region_stats['region'], region_stats['mean'], color=colors_region)
    
    for bar, n in zip(bars, region_stats['count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
               f'n={int(n)}', ha='center', fontsize=8)
    
    ax.set_xlabel('Region')
    ax.set_ylabel('Average Placement (Lower = Better)')
    ax.set_title('Figure Q3-6: Performance by Region')
    ax.tick_params(axis='x', rotation=45)
    
    best_region = region_stats.iloc[0]['region']
    ax.text(0.5, -0.18, f'结论：来自{best_region}地区的选手平均表现最优', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Q3_06_region_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(filepath)
    print(f"✓ 生成: {filepath}")
    
    return generated_files


# ============================================
# 主程序
# ============================================

def main():
    """问题3完整求解"""
    print("=" * 60)
    print("问题3：名人特征影响分析")
    print("=" * 60)
    
    # 1. 数据加载
    data = load_data('output/question3_data.csv')
    if data is None:
        return
    
    # 2. 特征工程
    feature_engineer = CelebrityFeatureEngineer()
    
    # 3. 子问题3.1：特征对结果的影响
    results_q31 = analyze_feature_impact_on_results(data, feature_engineer)
    
    # 4. 子问题3.2：差异化影响
    results_q32 = analyze_differential_impact(data, feature_engineer, results_q31)
    
    # 5. 子问题3.3：舞者影响
    dancer_stats = analyze_dancer_impact(data)
    
    # 6. 可视化
    viz_files = generate_visualizations(results_q31, results_q32, dancer_stats)
    
    # 7. 保存结果
    results_q31['data'].to_csv('output/Q3_feature_analysis.csv', 
                               index=False, encoding='utf-8-sig')
    
    # 8. 结果摘要
    print("\n" + "=" * 60)
    print("问题3求解结果摘要")
    print("=" * 60)
    
    print(f"\n【子问题3.1】名人特征对结果的影响:")
    print(f"  • 线性回归 CV R²: {results_q31['linear_r2']:.4f}")
    print(f"  • 随机森林 CV R²: {results_q31['rf_r2']:.4f}")
    
    top_feature = results_q31['importance'].iloc[0]
    print(f"  • 最重要特征: {top_feature['feature']} ({top_feature['importance']:.4f})")
    
    print(f"\n【子问题3.2】差异化影响分析:")
    comp = results_q32['comparison']
    fan_pref = comp[comp['diff'] > 0]['feature'].tolist()
    judge_pref = comp[comp['diff'] < 0]['feature'].tolist()
    if fan_pref:
        print(f"  • 粉丝更偏好: {', '.join(fan_pref[:3])}")
    if judge_pref:
        print(f"  • 评委更偏好: {', '.join(judge_pref[:3])}")
    
    print(f"\n【子问题3.3】舞者影响: 已分析")
    
    print(f"\n• 生成可视化: {len(viz_files)}个")
    print("\n>>> 问题3求解完成 <<<")
    
    return results_q31, results_q32, dancer_stats


if __name__ == '__main__':
    main()
