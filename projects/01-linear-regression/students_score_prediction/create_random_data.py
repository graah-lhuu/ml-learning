# generate_data.py
import numpy as np
import pandas as pd
import os

def generate_student_data(n_samples=100, random_seed=42):
    """
    生成模拟的学生数据
    假设：
    - 学习时间：主要影响因素，正相关
    - 睡眠时间：适度睡眠最好，倒U型关系
    - 看MYGO!!!!!时间：适度放松可能有益，过多有害
    - 下国际象棋：锻炼思维，正相关
    - 学法语：可能分散精力，轻度负相关
    """
    np.random.seed(random_seed)
    
    # 生成特征数据
    study_hours = np.random.normal(25, 8, n_samples)  # 平均25小时，标准差8
    study_hours = np.clip(study_hours, 5, 40)  # 限制在5-40小时
    
    sleep_hours = np.random.normal(7.5, 1.5, n_samples) * 7  # 每周睡眠
    sleep_hours = np.clip(sleep_hours, 35, 70)
    
    mygo_hours = np.random.exponential(3, n_samples)  # 指数分布，大多人看得少
    mygo_hours = np.clip(mygo_hours, 0, 15)
    
    chess_hours = np.random.normal(2, 1.5, n_samples)
    chess_hours = np.clip(chess_hours, 0, 8)
    
    french_hours = np.random.normal(4, 2, n_samples)
    french_hours = np.clip(french_hours, 0, 10)
    
    # 生成目标变量（考试成绩 0-100分）
    # 使用线性组合 + 非线性效应 + 随机噪声
    
    # 基础分数
    base_score = 50
    
    # 线性效应
    score = (base_score +
             1.8 * (study_hours - 20) +  # 每多学1小时+1.8分
             0.5 * (sleep_hours - 52.5) - 0.01 * (sleep_hours - 52.5)**2 +  # 倒U型
             -0.8 * mygo_hours + 0.1 * mygo_hours**2 +  # 适量有益，过多有害
             1.2 * chess_hours +  # 国际象棋有助思维
             -0.3 * french_hours)  # 学法语可能分散精力
    
    # 添加交互效应
    score += 0.05 * study_hours * chess_hours  # 学习+下棋有协同效应
    
    # 添加随机噪声
    score += np.random.normal(0, 8, n_samples)
    
    # 确保分数在合理范围
    score = np.clip(score, 0, 100)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'study_hours': study_hours,
        'sleep_hours': sleep_hours,
        'mygo_hours': mygo_hours,
        'chess_hours': chess_hours,
        'french_hours': french_hours,
        'exam_score': score
    })
    
    return data

def save_data_to_csv(data, filename='student_scores.csv'):
    """保存数据到指定目录"""
    # 创建目录（如果不存在）
    data_dir = r'F:\courses\python\ml-learning\data\students_score_prediction'
    os.makedirs(data_dir, exist_ok=True)
    
    # 保存文件
    filepath = os.path.join(data_dir, filename)
    data.to_csv(filepath, index=False)
    print(f"数据已保存到: {filepath}")
    print(f"数据形状: {data.shape}")
    print(f"前5行数据:")
    print(data.head())
    
    return filepath

if __name__ == "__main__":
    # 生成数据
    student_data = generate_student_data(n_samples=100)
    
    # 保存数据
    filepath = save_data_to_csv(student_data)
    
    # 打印数据统计信息
    print("\n数据统计信息:")
    print(student_data.describe())