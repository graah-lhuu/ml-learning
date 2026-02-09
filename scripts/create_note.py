#!/usr/bin/env python3
"""
自动创建机器学习学习笔记模板
位置：scripts/create_note.py
用法：python scripts/create_note.py
"""

import os
import sys
from datetime import datetime

def get_project_root():
    """获取项目根目录"""
    # 当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录（scripts的父目录）
    project_root = os.path.dirname(script_dir)
    return project_root

def create_ml_note(topic, note_type="theory", title=None, tags=None):
    """
    创建机器学习学习笔记
    
    Args:
        topic: 主题名称，如 "linear-regression", "logistic-regression"
        note_type: 笔记类型，可选 "theory"|"code"|"math"|"experiment"|"summary"
        title: 笔记标题（可选，默认为主题名）
        tags: 标签列表（可选）
    """
    
    # 映射笔记类型到文件名前缀
    type_prefixes = {
        "theory": "01-理论基础",
        "code": "02-代码实现", 
        "math": "03-数学推导",
        "experiment": "04-实验记录",
        "summary": "05-学习总结"
    }
    
    if note_type not in type_prefixes:
        print(f"❌ 错误：不支持的笔记类型 '{note_type}'")
        print(f"   支持的笔记类型: {', '.join(type_prefixes.keys())}")
        return None
    
    # 项目根目录
    project_root = get_project_root()
    
    # 主题目录路径
    topic_dir = os.path.join(project_root, "docs", topic)
    
    # 确保目录存在
    os.makedirs(topic_dir, exist_ok=True)
    
    # 生成文件名
    prefix = type_prefixes[note_type]
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{prefix}_{timestamp}.md"
    filepath = os.path.join(topic_dir, filename)
    
    # 如果文件已存在，询问是否覆盖
    if os.path.exists(filepath):
        response = input(f"⚠️  文件已存在: {filename}\n   是否覆盖？(y/n): ")
        if response.lower() != 'y':
            # 创建带时间戳的新文件
            timestamp_detail = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp_detail}.md"
            filepath = os.path.join(topic_dir, filename)
    
    # 生成笔记标题
    if title is None:
        # 从topic生成标题，如 "linear-regression" -> "线性回归"
        title_map = {
            "linear-regression": "线性回归",
            "logistic-regression": "逻辑回归", 
            "decision-trees": "决策树",
            "neural-networks": "神经网络"
        }
        title = title_map.get(topic, topic.replace("-", " ").title())
    
    # 笔记模板
    template = generate_note_template(topic, title, note_type, tags)
    
    # 写入文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"✅ 笔记已创建: {filepath}")
    print(f"   主题: {title}")
    print(f"   类型: {note_type}")
    print(f"   标签: {', '.join(tags) if tags else '无'}")
    
    return filepath

def generate_note_template(topic, title, note_type, tags=None):
    """生成笔记模板内容"""
    
    # 基础模板
    template = f"""# {title} - {get_note_type_cn(note_type)}

> 创建时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
> 主题: {topic}  
> 标签: {', '.join(tags) if tags else '机器学习'}

## 学习目标

## 核心概念

## 关键知识点

## 难点与解决方案

## 实践应用

## 学习心得

## 参考资料

1. 
2. 
3. 

---

**[返回docs目录](../README.md)** | **[返回项目主页](../../README.md)**
"""
    
    # 根据笔记类型添加特定部分
    if note_type == "theory":
        template = template.replace("## 核心概念", """## 核心概念

### 算法原理

### 数学基础

### 应用场景

## 优缺点分析""")
    
    elif note_type == "code":
        template = template.replace("## 核心概念", """## 核心概念

### 代码结构

### 关键函数

### 参数说明

## 实现步骤""")
    
    elif note_type == "math":
        template = template.replace("## 核心概念", """## 核心概念

### 数学模型

### 公式推导

### 收敛性分析

## 证明过程""")
    
    elif note_type == "experiment":
        template = template.replace("## 核心概念", """## 核心概念

### 实验目的

### 实验设计

### 实验数据

## 实验结果""")
    
    return template

def get_note_type_cn(note_type):
    """获取笔记类型的中文名称"""
    cn_map = {
        "theory": "理论基础",
        "code": "代码实现", 
        "math": "数学推导",
        "experiment": "实验记录",
        "summary": "学习总结"
    }
    return cn_map.get(note_type, note_type)

def interactive_create():
    """交互式创建笔记"""
    print("📝 机器学习学习笔记生成工具")
    print("=" * 40)
    
    # 选择主题
    topics = ["linear-regression", "logistic-regression", "decision-trees", "neural-networks", "other"]
    print("可用的主题:")
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic}")
    
    topic_idx = int(input("请选择主题编号: ")) - 1
    if topic_idx < 0 or topic_idx >= len(topics):
        print("❌ 无效的选择")
        return
    
    topic = topics[topic_idx]
    if topic == "other":
        topic = input("请输入自定义主题名称: ").strip()
    
    # 选择笔记类型
    note_types = ["theory", "code", "math", "experiment", "summary"]
    print("\n笔记类型:")
    for i, nt in enumerate(note_types, 1):
        print(f"  {i}. {get_note_type_cn(nt)} ({nt})")
    
    type_idx = int(input("请选择笔记类型编号: ")) - 1
    if type_idx < 0 or type_idx >= len(note_types):
        print("❌ 无效的选择")
        return
    
    note_type = note_types[type_idx]
    
    # 自定义标题
    custom_title = input("自定义标题（直接回车使用默认）: ").strip()
    title = custom_title if custom_title else None
    
    # 标签
    tags_input = input("标签（用逗号分隔，如：机器学习,线性代数）: ").strip()
    tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else None
    
    # 创建笔记
    print("\n" + "=" * 40)
    create_ml_note(topic, note_type, title, tags)

def main():
    """主函数"""
    try:
        # 如果有命令行参数，使用参数模式
        if len(sys.argv) > 1:
            # 命令行参数模式
            import argparse
            parser = argparse.ArgumentParser(description="创建机器学习学习笔记")
            parser.add_argument("topic", help="主题名称")
            parser.add_argument("-t", "--type", default="theory", 
                              choices=["theory", "code", "math", "experiment", "summary"],
                              help="笔记类型")
            parser.add_argument("--title", help="自定义标题")
            parser.add_argument("--tags", help="标签，用逗号分隔")
            
            args = parser.parse_args()
            
            tags = args.tags.split(",") if args.tags else None
            create_ml_note(args.topic, args.type, args.title, tags)
        else:
            # 交互式模式
            interactive_create()
            
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()