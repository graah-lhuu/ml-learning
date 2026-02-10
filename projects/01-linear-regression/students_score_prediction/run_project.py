# run_project.py
"""
项目运行脚本
使用方法：
1. 运行 python generate_data.py 生成数据
2. 运行 python linear_regression.py 训练模型并分析
"""

import subprocess
import sys
import os

def main():
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("机器学习项目: 学生成绩预测")
    print("="*50)
    
    # 检查并运行数据生成
    data_file = r'F:\courses\python\ml-learning\data\students_score_prediction\student_scores.csv'
    
    if not os.path.exists(data_file):
        print("数据文件不存在，正在生成模拟数据...")
        try:
            # 运行数据生成脚本
            from generate_data import generate_student_data, save_data_to_csv
            data = generate_student_data()
            save_data_to_csv(data)
            print("✓ 数据生成完成!")
        except Exception as e:
            print(f"✗ 数据生成失败: {e}")
            return
    else:
        print("✓ 数据文件已存在")
    
    # 运行线性回归分析
    print("\n开始线性回归分析...")
    try:
        from linear_regression import main as run_linear_regression
        run_linear_regression()
    except Exception as e:
        print(f"✗ 分析过程出错: {e}")

if __name__ == "__main__":
    main()