"""
波士顿房价预测 - 线性回归入门项目
作者：东南大学吴健雄学院 夏滔
日期：2026-02-09
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# 设置中文字体和图形显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class HousePricePredictor:
    """房价预测器 - 使用线性回归模型"""
    
    def __init__(self):
        """初始化预测器"""
        self.model = LinearRegression()
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """
        加载加州房价数据集
        这是波士顿房价数据集的替代（波士顿数据集因伦理问题已弃用）
        包含8个特征，目标为房价中位数
        """
        print("=" * 50)
        print("步骤1: 加载数据集")
        print("=" * 50)
        
        housing = fetch_california_housing()
        self.data = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.data['PRICE'] = housing.target
        
        print(f"数据集形状: {self.data.shape}")
        print(f"特征数量: {len(housing.feature_names)}")
        print("\n前5行数据:")
        print(self.data.head())
        print("\n数据描述:")
        print(self.data.describe())
        
        return self.data
    
    def explore_data(self):
        """数据探索和可视化"""
        print("\n" + "=" * 50)
        print("步骤2: 数据探索")
        print("=" * 50)
        
        # 检查缺失值
        print("缺失值统计:")
        print(self.data.isnull().sum())
        
        # 可视化特征与目标的关系
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        features = self.data.columns[:-1]  # 所有特征列
        target = self.data.columns[-1]     # 目标列（房价）
        
        for i, feature in enumerate(features):
            axes[i].scatter(self.data[feature], self.data[target], alpha=0.5)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('房价')
            axes[i].set_title(f'{feature} vs 房价')
        
        plt.tight_layout()
        plt.savefig('特征可视化.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 计算相关系数
        correlation = self.data.corr()['PRICE'].sort_values(ascending=False)
        print("\n特征与房价的相关系数:")
        print(correlation)
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """准备训练和测试数据"""
        print("\n" + "=" * 50)
        print("步骤3: 准备训练/测试数据")
        print("=" * 50)
        
        X = self.data.drop('PRICE', axis=1)
        y = self.data['PRICE']
        
        # 分割数据集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        print(f"特征示例: {list(X.columns[:3])}...")
    
    def train_model(self):
        """训练线性回归模型"""
        print("\n" + "=" * 50)
        print("步骤4: 训练线性回归模型")
        print("=" * 50)
        
        # 训练模型
        self.model.fit(self.X_train, self.y_train)
        
        # 获取模型参数
        print(f"模型截距 (θ₀): {self.model.intercept_:.4f}")
        print("\n模型系数 (θ₁到θ₈):")
        for i, (coef, feature) in enumerate(zip(self.model.coef_, self.X_train.columns)):
            print(f"  {feature}: {coef:.4f}")
        
        # 数学解释
        print("\n📊 数学解释:")
        print("线性回归模型公式: y = θ₀ + θ₁x₁ + θ₂x₂ + ... + θ₈x₈")
        print("其中 θ₀ 是截距，θ₁-θ₈ 是特征系数")
        print("通过最小化均方误差(MSE)来找到最佳参数")
        
        return self.model
    
    def evaluate_model(self):
        """评估模型性能"""
        print("\n" + "=" * 50)
        print("步骤5: 模型评估")
        print("=" * 50)
        
        # 预测
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # 计算指标
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print("训练集评估:")
        print(f"  均方误差 (MSE): {train_mse:.4f}")
        print(f"  R²分数: {train_r2:.4f}")
        print("\n测试集评估:")
        print(f"  均方误差 (MSE): {test_mse:.4f}")
        print(f"  R²分数: {test_r2:.4f}")
        
        # 可视化预测结果
        self._visualize_predictions(y_test_pred)
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def _visualize_predictions(self, y_test_pred):
        """可视化预测结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 真实值 vs 预测值
        axes[0].scatter(self.y_test, y_test_pred, alpha=0.6, edgecolors='k')
        axes[0].plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 
                    'r--', lw=2)
        axes[0].set_xlabel('真实房价')
        axes[0].set_ylabel('预测房价')
        axes[0].set_title('真实值 vs 预测值')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 残差图
        residuals = self.y_test - y_test_pred
        axes[1].scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('预测房价')
        axes[1].set_ylabel('残差 (真实-预测)')
        axes[1].set_title('残差图')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('模型评估.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def make_prediction(self, sample_house=None):
        """使用模型进行预测"""
        print("\n" + "=" * 50)
        print("步骤6: 预测示例")
        print("=" * 50)
        
        if sample_house is None:
            # 从测试集取一个样本
            sample_idx = 0
            sample_house = self.X_test.iloc[sample_idx:sample_idx+1]
            actual_price = self.y_test.iloc[sample_idx]
        
        predicted_price = self.model.predict(sample_house)[0]
        
        print("房屋特征值:")
        for feature, value in zip(sample_house.columns, sample_house.values[0]):
            print(f"  {feature}: {value:.2f}")
        
        print(f"\n预测房价: ${predicted_price*100000:.2f}")
        
        if sample_house is self.X_test.iloc[sample_idx:sample_idx+1]:
            print(f"实际房价: ${actual_price*100000:.2f}")
            print(f"预测误差: ${abs(actual_price - predicted_price)*100000:.2f}")
    
    def run_complete_pipeline(self):
        """运行完整的机器学习流程"""
        print("🏠 波士顿房价预测 - 线性回归模型")
        print("✨ 东南大学吴健雄学院 夏滔 机器学习入门项目")
        print("=" * 50)
        
        # 完整流程
        self.load_data()
        self.explore_data()
        self.prepare_data()
        self.train_model()
        metrics = self.evaluate_model()
        self.make_prediction()
        
        print("\n" + "=" * 50)
        print("🎉 项目完成!")
        print("=" * 50)
        print("总结:")
        print(f"1. 使用了 {self.X_train.shape[1]} 个特征")
        print(f"2. 训练了线性回归模型")
        print(f"3. 测试集 R² 分数: {metrics['test_r2']:.4f}")
        print(f"4. 生成了可视化图表: 特征可视化.png, 模型评估.png")
        
        return metrics


def main():
    """主函数"""
    try:
        # 创建预测器实例
        predictor = HousePricePredictor()
        
        # 运行完整流程
        predictor.run_complete_pipeline()
        
        # 学习建议
        print("\n" + "=" * 50)
        print("📚 下一步学习建议:")
        print("=" * 50)
        print("1. 尝试修改 test_size 参数，观察模型性能变化")
        print("2. 实现自己的线性回归（用NumPy，不用scikit-learn）")
        print("3. 添加新的特征工程（如特征组合、多项式特征）")
        print("4. 尝试其他回归模型（决策树、随机森林）")
        print("5. 用梯度下降手动优化模型参数")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请确保已安装所需库: pip install numpy pandas matplotlib scikit-learn")


if __name__ == "__main__":
    main()