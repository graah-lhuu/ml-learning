# linear_regression.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import os

class LinearRegressionManual:
    """手动实现的线性回归"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        """使用梯度下降训练模型"""
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for _ in range(self.n_iterations):
            # 预测
            y_pred = np.dot(X, self.weights) + self.bias
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 记录损失
            loss = mean_squared_error(y, y_pred)
            self.losses.append(loss)
        
        return self
    
    def predict(self, X):
        """预测"""
        return np.dot(X, self.weights) + self.bias
    
    def get_coefficients(self):
        """获取系数"""
        return self.weights, self.bias

def analyze_importance(weights, feature_names):
    """分析特征重要性"""
    print("\n" + "="*50)
    print("特征重要性分析")
    print("="*50)
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': weights,
        'abs_coefficient': np.abs(weights)
    })
    
    # 按绝对值排序
    importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
    
    print("特征系数（正表示正相关，负表示负相关）:")
    for idx, row in importance_df.iterrows():
        direction = "↑ 正相关" if row['coefficient'] > 0 else "↓ 负相关"
        print(f"{row['feature']:15s}: {row['coefficient']:8.4f} ({direction})")
    
    print(f"\n最重要的特征: {importance_df.iloc[0]['feature']}")
    print(f"影响最小的特征: {importance_df.iloc[-1]['feature']}")
    
    return importance_df

def visualize_results(model, X_test, y_test, feature_names, importance_df):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 损失曲线
    axes[0, 0].plot(model.losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 特征重要性
    colors = plt.cm.Set3(np.linspace(0, 1, len(importance_df)))
    bars = axes[0, 1].barh(importance_df['feature'], importance_df['abs_coefficient'], color=colors)
    axes[0, 1].set_title('Feature Importance (Absolute Coefficients)')
    axes[0, 1].set_xlabel('Absolute Coefficient Value')
    axes[0, 1].invert_yaxis()
    
    # 3. 预测 vs 实际
    y_pred = model.predict(X_test)
    axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Scores')
    axes[1, 0].set_ylabel('Predicted Scores')
    axes[1, 0].set_title(f'Predictions vs Actual (R² = {r2_score(y_test, y_pred):.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 残差图
    residuals = y_test - y_pred
    axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Scores')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residual Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    # 1. 加载数据
    # 使用绝对路径
    data_path = r'F:\courses\python\ml-learning\data\students_score_prediction\student_scores.csv'
    
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        print("请先运行 generate_data.py 生成数据")
        return
    
    data = pd.read_csv(data_path)
    print("数据加载成功!")
    print(f"数据形状: {data.shape}")
    print("\n前5行数据:")
    print(data.head())
    
    # 2. 准备数据
    feature_names = ['study_hours', 'sleep_hours', 'mygo_hours', 'chess_hours', 'french_hours']
    X = data[feature_names].values
    y = data['exam_score'].values
    
    # 3. 数据标准化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )
    
    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 5. 训练模型
    print("\n训练线性回归模型...")
    model = LinearRegressionManual(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # 6. 评估模型
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n模型性能:")
    print(f"MSE (均方误差): {mse:.2f}")
    print(f"R² Score: {r2:.3f}")
    
    # 7. 特征重要性分析
    weights, bias = model.get_coefficients()
    importance_df = analyze_importance(weights, feature_names)
    
    # 8. 预测新学生
    print("\n" + "="*50)
    print("新学生成绩预测")
    print("="*50)
    
    # 示例：预测一个新学生
    new_student = np.array([[30, 56, 2, 3, 2]])  # 学习30h, 睡眠56h, MYGO 2h, 下棋3h, 法语2h
    new_student_normalized = (new_student - X_mean) / X_std
    predicted_score = model.predict(new_student_normalized)[0]
    
    print(f"\n新学生数据:")
    for name, value in zip(feature_names, new_student[0]):
        print(f"  {name}: {value}小时")
    print(f"预测成绩: {predicted_score:.1f}分")
    
    # 9. 可视化
    print("\n生成可视化图表...")
    visualize_results(model, X_test, y_test, feature_names, importance_df)
    
    # 10. 模型系数解释
    print("\n" + "="*50)
    print("模型系数解释")
    print("="*50)
    print("标准化后的系数意味着每个特征增加1个标准差对成绩的影响:")
    
    for i, (name, weight) in enumerate(zip(feature_names, weights)):
        effect = weight * X_std[i]  # 转换为原始尺度的影响
        direction = "提高" if weight > 0 else "降低"
        print(f"{name:15s}: 每增加1小时，成绩{direction}{abs(effect):.2f}分")
    
    print(f"偏置项: {bias:.2f} (基准分数)")

if __name__ == "__main__":
    main()