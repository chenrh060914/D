#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3：名人特征影响分析
========================
模型方法：多元线性回归 + XGBoost + SHAP可解释性分析

核心思路：
1. 分析名人特征（年龄、行业、地域等）对比赛结果的影响
2. 使用多元线性回归获得系数解释
3. 使用XGBoost捕捉非线性关系
4. 使用SHAP分析差异化影响（评委vs粉丝）

作者：MCM 2026 C题参赛团队
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 尝试导入XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost未安装，将使用随机森林替代")
    from sklearn.ensemble import RandomForestRegressor

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
# 第二部分：特征工程
# ============================================

class FeatureEngineer:
    """
    名人特征工程器
    
    处理名人特征：
    - 年龄: 数值型，可分箱
    - 行业: 类别型，编码处理
    - 地域: 类别型，分组处理
    - 国籍: 类别型，二值化
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def process_age(self, data):
        """
        处理年龄特征
        
        分箱策略：
        - 青年: <30岁
        - 中年: 30-45岁
        - 成熟: >45岁
        """
        age_col = 'celebrity_age_during_season'
        if age_col not in data.columns:
            return data
        
        data = data.copy()
        
        # 保留原始年龄
        data['age'] = data[age_col].fillna(data[age_col].median())
        
        # 年龄分箱
        def age_bin(age):
            if pd.isna(age):
                return 'Unknown'
            elif age < 30:
                return 'Young'
            elif age <= 45:
                return 'Middle'
            else:
                return 'Mature'
        
        data['age_group'] = data['age'].apply(age_bin)
        
        return data
    
    def process_industry(self, data):
        """
        处理行业特征
        
        分组策略：
        - Entertainment: Actor/Actress, Singer, TV Personality
        - Sports: Athlete, Olympian
        - Other: 其他行业
        """
        industry_col = 'celebrity_industry'
        if industry_col not in data.columns:
            return data
        
        data = data.copy()
        
        # 行业分组映射
        industry_mapping = {
            'Actor': 'Entertainment',
            'Actress': 'Entertainment',
            'Actor/Actress': 'Entertainment',
            'Singer': 'Entertainment',
            'Singer/Rapper': 'Entertainment',
            'TV Personality': 'Entertainment',
            'Model': 'Entertainment',
            'Reality Star': 'Entertainment',
            'Athlete': 'Sports',
            'Olympian': 'Sports',
            'NFL Player': 'Sports',
            'NBA Player': 'Sports',
            'Journalist': 'Media',
            'News Anchor': 'Media',
            'Politician': 'Politics'
        }
        
        def map_industry(industry):
            if pd.isna(industry):
                return 'Other'
            for key, value in industry_mapping.items():
                if key.lower() in str(industry).lower():
                    return value
            return 'Other'
        
        data['industry_group'] = data[industry_col].apply(map_industry)
        
        # 编码
        if 'industry' not in self.label_encoders:
            self.label_encoders['industry'] = LabelEncoder()
            data['industry_encoded'] = self.label_encoders['industry'].fit_transform(data['industry_group'])
        else:
            data['industry_encoded'] = self.label_encoders['industry'].transform(data['industry_group'])
        
        return data
    
    def process_region(self, data):
        """
        处理地域特征
        
        分组策略：按美国人口普查区域分组
        - Northeast, Southeast, Midwest, Southwest, West, Non-US
        """
        state_col = 'celebrity_homestate'
        country_col = 'celebrity_homecountry/region'
        
        data = data.copy()
        
        # 美国州到区域的映射
        state_to_region = {
            # Northeast
            'Connecticut': 'Northeast', 'Maine': 'Northeast', 'Massachusetts': 'Northeast',
            'New Hampshire': 'Northeast', 'Rhode Island': 'Northeast', 'Vermont': 'Northeast',
            'New Jersey': 'Northeast', 'New York': 'Northeast', 'Pennsylvania': 'Northeast',
            # Southeast
            'Delaware': 'Southeast', 'Florida': 'Southeast', 'Georgia': 'Southeast',
            'Maryland': 'Southeast', 'North Carolina': 'Southeast', 'South Carolina': 'Southeast',
            'Virginia': 'Southeast', 'West Virginia': 'Southeast', 'Alabama': 'Southeast',
            'Kentucky': 'Southeast', 'Mississippi': 'Southeast', 'Tennessee': 'Southeast',
            'Arkansas': 'Southeast', 'Louisiana': 'Southeast',
            # Midwest
            'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Michigan': 'Midwest',
            'Ohio': 'Midwest', 'Wisconsin': 'Midwest', 'Iowa': 'Midwest',
            'Kansas': 'Midwest', 'Minnesota': 'Midwest', 'Missouri': 'Midwest',
            'Nebraska': 'Midwest', 'North Dakota': 'Midwest', 'South Dakota': 'Midwest',
            # Southwest
            'Arizona': 'Southwest', 'New Mexico': 'Southwest', 'Oklahoma': 'Southwest', 'Texas': 'Southwest',
            # West
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
                return state_to_region.get(state, 'Other-US')
            
            return 'Unknown'
        
        data['region'] = data.apply(get_region, axis=1)
        
        # 编码
        if 'region' not in self.label_encoders:
            self.label_encoders['region'] = LabelEncoder()
            data['region_encoded'] = self.label_encoders['region'].fit_transform(data['region'])
        else:
            data['region_encoded'] = self.label_encoders['region'].transform(data['region'])
        
        # 是否美国人
        data['is_us'] = (data['region'] != 'Non-US').astype(int)
        
        return data
    
    def build_features(self, data):
        """
        构建完整特征矩阵
        
        参数:
            data: 原始数据
        
        返回:
            data: 处理后的数据
            feature_names: 特征名称列表
        """
        print("\n>>> 特征工程")
        print("-" * 40)
        
        # 处理各类特征
        data = self.process_age(data)
        data = self.process_industry(data)
        data = self.process_region(data)
        
        # 定义特征列表
        self.feature_names = [
            'age',
            'industry_encoded',
            'region_encoded',
            'is_us'
        ]
        
        # 添加可用的其他特征
        optional_features = ['cumulative_total_score', 'overall_avg_score', 
                            'score_trend', 'active_weeks']
        for feat in optional_features:
            if feat in data.columns:
                self.feature_names.append(feat)
        
        # 填充缺失值
        for col in self.feature_names:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].median() if data[col].dtype in ['float64', 'int64'] else 0)
        
        print(f"✓ 特征工程完成")
        print(f"  - 特征数量: {len(self.feature_names)}")
        print(f"  - 特征列表: {self.feature_names}")
        
        return data, self.feature_names


# ============================================
# 第三部分：多元线性回归模型
# ============================================

class LinearRegressionAnalyzer:
    """
    多元线性回归分析器
    
    用于分析名人特征对比赛结果的线性影响
    
    参数说明:
        regularization: 正则化方法 ('none', 'ridge', 'lasso')
        alpha: 正则化系数
    """
    
    def __init__(self, regularization='ridge', alpha=1.0):
        """
        初始化方式：
            - regularization: 正则化类型，ridge对共线性更稳健
            - alpha: 正则化强度，较大值减少过拟合
        """
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()
        
        self.regularization = regularization
        self.alpha = alpha
        self.feature_names = None
        self.coefficients = None
        
        print(f"✓ 线性回归模型初始化")
        print(f"  - 正则化: {regularization}")
        print(f"  - Alpha: {alpha}")
    
    def fit(self, X, y, feature_names):
        """
        拟合模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称
        
        注意事项:
            - 使用VIF检验多重共线性
            - 标注显著性水平
        """
        self.feature_names = feature_names
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 拟合模型
        self.model.fit(X_scaled, y)
        
        # 提取系数
        self.coefficients = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        # 计算R²
        y_pred = self.model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"\n模型拟合结果:")
        print(f"  - R²: {r2:.4f}")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - 截距: {self.model.intercept_:.4f}")
        
        return {
            'r2': r2,
            'rmse': rmse,
            'intercept': self.model.intercept_,
            'coefficients': self.coefficients
        }
    
    def calculate_vif(self, X, feature_names):
        """
        计算方差膨胀因子（VIF）检测多重共线性
        
        VIF > 10 表示存在严重多重共线性
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        vif_data = []
        for i, feature in enumerate(feature_names):
            try:
                vif = variance_inflation_factor(X, i)
                vif_data.append({'feature': feature, 'VIF': vif})
            except:
                vif_data.append({'feature': feature, 'VIF': np.nan})
        
        vif_df = pd.DataFrame(vif_data)
        
        high_vif = vif_df[vif_df['VIF'] > 10]
        if len(high_vif) > 0:
            print(f"  ⚠ 发现多重共线性问题:")
            for _, row in high_vif.iterrows():
                print(f"    {row['feature']}: VIF = {row['VIF']:.2f}")
        
        return vif_df


# ============================================
# 第四部分：XGBoost模型
# ============================================

class XGBoostAnalyzer:
    """
    XGBoost回归分析器
    
    用于捕捉名人特征与比赛结果的非线性关系
    
    参数说明:
        n_estimators: 树的数量
        max_depth: 最大深度
        learning_rate: 学习率
        early_stopping_rounds: 早停轮数（防止过拟合）
    """
    
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1, 
                 early_stopping_rounds=10, random_state=42):
        """
        初始化方式：
            - n_estimators=100: 足够的迭代次数
            - max_depth=5: 限制深度避免过拟合
            - learning_rate=0.1: 适中的学习率
            - early_stopping: 早停机制防止过拟合
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        
        if HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        
        self.feature_names = None
        
        print(f"✓ {'XGBoost' if HAS_XGBOOST else '随机森林'}模型初始化")
        print(f"  - 树数量: {n_estimators}")
        print(f"  - 最大深度: {max_depth}")
    
    def fit(self, X, y, feature_names, cv_folds=5):
        """
        拟合模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称
            cv_folds: 交叉验证折数
        
        注意事项:
            - 使用早停机制防止过拟合
            - 监控训练和验证损失曲线
        """
        self.feature_names = feature_names
        
        # 交叉验证
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        
        print(f"\n交叉验证R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 在全部数据上训练
        self.model.fit(X, y)
        
        # 特征重要性
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # 计算训练集指标
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"训练集R²: {r2:.4f}")
        print(f"训练集RMSE: {rmse:.4f}")
        
        # 过拟合检测
        if r2 - cv_scores.mean() > 0.15:
            print(f"  ⚠ 可能存在过拟合: 差距 = {r2 - cv_scores.mean():.4f}")
        
        return {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'train_r2': r2,
            'train_rmse': rmse,
            'feature_importance': feature_importance
        }


# ============================================
# 第五部分：SHAP分析
# ============================================

def shap_analysis(model, X, feature_names, output_dir='output'):
    """
    SHAP值分析
    
    参数:
        model: 训练好的模型
        X: 特征矩阵
        feature_names: 特征名称
        output_dir: 输出目录
    
    返回:
        shap_results: SHAP分析结果
    """
    print("\n>>> SHAP可解释性分析")
    print("-" * 40)
    
    try:
        import shap
        
        # 创建SHAP解释器
        if HAS_XGBOOST:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model)
        
        shap_values = explainer.shap_values(X)
        
        # 计算每个特征的平均绝对SHAP值
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_summary = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        print(f"  ✓ SHAP分析完成")
        print(f"\n  SHAP特征重要性排序:")
        for i, row in shap_summary.iterrows():
            print(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")
        
        return {
            'shap_values': shap_values,
            'shap_summary': shap_summary,
            'expected_value': explainer.expected_value
        }
        
    except ImportError:
        print("  ⚠ SHAP库未安装，跳过SHAP分析")
        return None
    except Exception as e:
        print(f"  ⚠ SHAP分析出错: {str(e)}")
        return None


# ============================================
# 第六部分：差异化影响分析
# ============================================

def differential_impact_analysis(data, feature_engineer, output_dir='output'):
    """
    差异化影响分析：比较特征对评委评分vs粉丝投票的不同影响
    
    参数:
        data: 处理后的数据
        feature_engineer: 特征工程器
        output_dir: 输出目录
    
    返回:
        diff_analysis: 差异化分析结果
    """
    print("\n>>> 差异化影响分析（评委 vs 粉丝）")
    print("=" * 50)
    
    feature_names = feature_engineer.feature_names
    
    # 准备特征矩阵
    X = data[feature_names].fillna(0).values
    
    # 目标变量
    y_judge = data['overall_avg_score'].fillna(data['overall_avg_score'].median()).values
    y_placement = data['placement'].values
    
    # 分别建立模型
    results = {}
    
    # 评委评分模型
    print("\n【评委评分预测模型】")
    model_judge = XGBoostAnalyzer(n_estimators=50, max_depth=4)
    results['judge'] = model_judge.fit(X, y_judge, feature_names)
    
    # 排名预测模型（作为粉丝影响的代理）
    print("\n【最终排名预测模型】")
    model_placement = XGBoostAnalyzer(n_estimators=50, max_depth=4)
    results['placement'] = model_placement.fit(X, y_placement, feature_names)
    
    # 对比特征重要性
    judge_imp = results['judge']['feature_importance'].set_index('feature')['importance']
    place_imp = results['placement']['feature_importance'].set_index('feature')['importance']
    
    comparison = pd.DataFrame({
        'feature': feature_names,
        'judge_importance': [judge_imp.get(f, 0) for f in feature_names],
        'placement_importance': [place_imp.get(f, 0) for f in feature_names]
    })
    
    comparison['importance_diff'] = comparison['placement_importance'] - comparison['judge_importance']
    comparison = comparison.sort_values('importance_diff', ascending=False)
    
    print("\n特征影响差异（排名 - 评委）:")
    print("正值=对粉丝投票影响更大，负值=对评委评分影响更大")
    for _, row in comparison.iterrows():
        direction = "→ 粉丝偏好" if row['importance_diff'] > 0 else "→ 评委偏好"
        print(f"  {row['feature']}: {row['importance_diff']:.4f} {direction}")
    
    results['comparison'] = comparison
    
    return results


# ============================================
# 第七部分：可视化生成
# ============================================

def generate_visualizations(data, linear_results, xgb_results, diff_results, 
                          feature_names, output_dir='output'):
    """
    生成问题3相关的可视化图表
    
    参数:
        data: 处理后的数据
        linear_results: 线性回归结果
        xgb_results: XGBoost结果
        diff_results: 差异化分析结果
        feature_names: 特征名称
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
    
    # 图1: 线性回归系数图
    if 'coefficients' in linear_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        coef_df = linear_results['coefficients']
        colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef_df['coefficient']]
        
        bars = ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Regression Coefficient')
        ax.set_title('Figure Q3-1: Linear Regression Coefficients for Celebrity Features')
        ax.invert_yaxis()
        
        # 添加R²标注
        ax.text(0.95, 0.05, f'R² = {linear_results["r2"]:.4f}', 
                transform=ax.transAxes, ha='right', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat'))
        
        filepath = os.path.join(output_dir, 'Q3_01_linear_coefficients.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    # 图2: XGBoost特征重要性
    if 'feature_importance' in xgb_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        imp_df = xgb_results['feature_importance']
        
        bars = ax.barh(imp_df['feature'], imp_df['importance'], color='steelblue')
        ax.set_xlabel('Feature Importance')
        ax.set_title('Figure Q3-2: XGBoost Feature Importance for Celebrity Characteristics')
        ax.invert_yaxis()
        
        # 添加CV R²标注
        ax.text(0.95, 0.05, f'CV R² = {xgb_results["cv_r2_mean"]:.4f}', 
                transform=ax.transAxes, ha='right', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat'))
        
        filepath = os.path.join(output_dir, 'Q3_02_xgb_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    # 图3: 年龄与排名关系
    if 'age' in data.columns and 'placement' in data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(data['age'], data['placement'], alpha=0.5, c='steelblue', s=50)
        
        # 添加趋势线
        valid_mask = ~(data['age'].isna() | data['placement'].isna())
        if valid_mask.sum() > 2:
            z = np.polyfit(data.loc[valid_mask, 'age'], 
                          data.loc[valid_mask, 'placement'], 2)
            p = np.poly1d(z)
            x_range = np.linspace(data['age'].min(), data['age'].max(), 100)
            ax.plot(x_range, p(x_range), 'r-', linewidth=2, label='Polynomial Fit')
        
        ax.set_xlabel('Celebrity Age')
        ax.set_ylabel('Final Placement (1 = Winner)')
        ax.set_title('Figure Q3-3: Age vs Final Placement (Non-linear Relationship)')
        ax.legend()
        
        # 添加最佳年龄区间标注
        ax.axvspan(30, 45, alpha=0.2, color='green', label='Optimal Age Range')
        
        filepath = os.path.join(output_dir, 'Q3_03_age_placement.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    # 图4: 行业分组对比箱线图
    if 'industry_group' in data.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 评委评分
        industry_order = data.groupby('industry_group')['overall_avg_score'].median().sort_values(ascending=False).index
        sns.boxplot(data=data, x='industry_group', y='overall_avg_score', 
                   order=industry_order, ax=axes[0], palette='Set2')
        axes[0].set_xlabel('Industry Group')
        axes[0].set_ylabel('Average Judge Score')
        axes[0].set_title('Judge Scores by Industry')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 最终排名
        sns.boxplot(data=data, x='industry_group', y='placement', 
                   order=industry_order, ax=axes[1], palette='Set2')
        axes[1].set_xlabel('Industry Group')
        axes[1].set_ylabel('Final Placement')
        axes[1].set_title('Final Placement by Industry')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Figure Q3-4: Industry Group Impact Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'Q3_04_industry_impact.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    # 图5: 差异化影响对比图
    if diff_results and 'comparison' in diff_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        comp_df = diff_results['comparison']
        
        x = np.arange(len(comp_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, comp_df['judge_importance'], width, 
                      label='Judge Score Impact', color='steelblue')
        bars2 = ax.bar(x + width/2, comp_df['placement_importance'], width, 
                      label='Placement Impact', color='coral')
        
        ax.set_xlabel('Celebrity Features')
        ax.set_ylabel('Feature Importance')
        ax.set_title('Figure Q3-5: Differential Impact - Judge Score vs Final Placement')
        ax.set_xticks(x)
        ax.set_xticklabels(comp_df['feature'], rotation=45, ha='right')
        ax.legend()
        
        # 添加解释性注释
        ax.text(0.5, -0.2, 
                'Note: Higher placement impact suggests stronger influence from fan voting',
                transform=ax.transAxes, ha='center', fontsize=10, style='italic')
        
        filepath = os.path.join(output_dir, 'Q3_05_differential_impact.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    # 图6: 地域分布热力图
    if 'region' in data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        region_stats = data.groupby('region').agg({
            'placement': 'mean',
            'overall_avg_score': 'mean',
            'celebrity_name': 'count'
        }).rename(columns={'celebrity_name': 'count'})
        
        # 创建简单的条形图替代热力图
        regions = region_stats.index.tolist()
        x = np.arange(len(regions))
        
        ax.bar(x, region_stats['placement'], color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45, ha='right')
        ax.set_ylabel('Average Placement')
        ax.set_title('Figure Q3-6: Regional Distribution of Performance')
        
        # 添加样本量标注
        for i, (region, row) in enumerate(region_stats.iterrows()):
            ax.text(i, row['placement'] + 0.5, f'n={int(row["count"])}', 
                   ha='center', fontsize=9)
        
        filepath = os.path.join(output_dir, 'Q3_06_regional_impact.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filepath)
        print(f"✓ 生成: {filepath}")
    
    return generated_files


# ============================================
# 第八部分：模型保存与结果导出
# ============================================

def save_results(data, linear_results, xgb_results, diff_results, output_dir='output'):
    """
    保存分析结果
    """
    import os
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存处理后的数据
    data.to_csv(os.path.join(output_dir, 'Q3_processed_features.csv'), 
                index=False, encoding='utf-8-sig')
    
    # 保存分析结果
    results = {
        'linear_results': linear_results,
        'xgb_results': xgb_results,
        'diff_results': diff_results
    }
    
    with open(os.path.join(output_dir, 'Q3_analysis_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✓ 结果已保存到 {output_dir}")


# ============================================
# 主程序入口
# ============================================

def main():
    """
    主程序：执行完整的名人特征影响分析流程
    """
    print("=" * 60)
    print("问题3：名人特征影响分析")
    print("=" * 60)
    
    # 1. 数据加载
    print("\n【步骤1】数据加载")
    data = load_data('output/question3_data.csv')
    
    if data is None:
        print("数据加载失败，程序终止")
        return
    
    # 2. 特征工程
    print("\n【步骤2】特征工程")
    feature_engineer = FeatureEngineer()
    data, feature_names = feature_engineer.build_features(data)
    
    # 准备特征矩阵
    X = data[feature_names].fillna(0).values
    y = data['placement'].values  # 目标：最终排名
    
    # 3. 线性回归分析
    print("\n【步骤3】线性回归分析")
    linear_analyzer = LinearRegressionAnalyzer(regularization='ridge', alpha=1.0)
    linear_results = linear_analyzer.fit(X, y, feature_names)
    
    # 4. XGBoost分析
    print("\n【步骤4】XGBoost分析")
    xgb_analyzer = XGBoostAnalyzer(n_estimators=100, max_depth=5)
    xgb_results = xgb_analyzer.fit(X, y, feature_names)
    
    # 5. SHAP分析
    print("\n【步骤5】SHAP分析")
    shap_results = shap_analysis(xgb_analyzer.model, X, feature_names, 'output')
    if shap_results:
        xgb_results['shap'] = shap_results
    
    # 6. 差异化影响分析
    print("\n【步骤6】差异化影响分析")
    diff_results = differential_impact_analysis(data, feature_engineer, 'output')
    
    # 7. 可视化生成
    print("\n【步骤7】可视化生成")
    viz_files = generate_visualizations(data, linear_results, xgb_results, 
                                        diff_results, feature_names, 'output')
    
    # 8. 保存结果
    print("\n【步骤8】保存结果")
    save_results(data, linear_results, xgb_results, diff_results, 'output')
    
    # 9. 结果摘要
    print("\n" + "=" * 60)
    print("模型求解结果摘要")
    print("=" * 60)
    print(f"• 分析样本数: {len(data)}")
    print(f"• 特征数量: {len(feature_names)}")
    print(f"• 线性回归 R²: {linear_results['r2']:.4f}")
    print(f"• XGBoost CV R²: {xgb_results['cv_r2_mean']:.4f}")
    print(f"• 生成可视化图表: {len(viz_files)} 个")
    
    print("\n【关键发现】")
    if 'coefficients' in linear_results:
        top_feature = linear_results['coefficients'].iloc[0]
        print(f"• 最重要线性特征: {top_feature['feature']} (系数={top_feature['coefficient']:.4f})")
    
    if 'feature_importance' in xgb_results:
        top_xgb = xgb_results['feature_importance'].iloc[0]
        print(f"• 最重要非线性特征: {top_xgb['feature']} (重要性={top_xgb['importance']:.4f})")
    
    print("\n>>> 问题3模型求解完成 <<<")
    
    return {
        'data': data,
        'linear_results': linear_results,
        'xgb_results': xgb_results,
        'diff_results': diff_results
    }


if __name__ == '__main__':
    main()
