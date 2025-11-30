#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回测结果查看工具
帮助用户查看和分析回测结果
"""

import os
import subprocess
from datetime import datetime

def list_backtest_results():
    """列出所有回测结果文件"""
    backtests_dir = "backtests"
    if not os.path.exists(backtests_dir):
        print("回测目录不存在，请先运行回测。")
        return
    
    files = [f for f in os.listdir(backtests_dir) if f.endswith('.png')]
    if not files:
        print("没有找到回测结果图表。")
        return
    
    print("\n可用的回测结果图表:")
    for i, file in enumerate(files, 1):
        file_path = os.path.join(backtests_dir, file)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
        size = os.path.getsize(file_path) / 1024  # KB
        print(f"{i}. {file} (修改时间: {mod_time}, 大小: {size:.1f}KB)")
    
    return files

def view_chart(file_index):
    """查看指定的回测图表"""
    backtests_dir = "backtests"
    files = [f for f in os.listdir(backtests_dir) if f.endswith('.png')]
    
    if not files or file_index < 1 or file_index > len(files):
        print("无效的文件索引。")
        return
    
    file_path = os.path.join(backtests_dir, files[file_index - 1])
    
    # 在macOS上使用open命令打开图片
    try:
        subprocess.run(["open", file_path], check=True)
        print(f"已打开图表: {files[file_index - 1]}")
    except subprocess.CalledProcessError:
        print(f"无法打开图表: {file_path}")
    except FileNotFoundError:
        print("系统不支持自动打开图片，请手动查看文件:", file_path)

def run_quick_backtest():
    """运行快速回测示例"""
    print("\n运行快速回测示例...")
    cmd = [
        "python", "quant_framework/scripts/run_backtest.py",
        "--symbol-slug", "btc-usdt-usdt",
        "--timeframe", "5m",
        "--cash", "10000",
        "--commission", "0.0005",
        "--plot"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n回测完成! 使用选项1查看新生成的图表。")
    except subprocess.CalledProcessError as e:
        print(f"回测失败: {e}")

def main():
    """主菜单"""
    while True:
        print("\n" + "="*50)
        print("量化交易回测工具")
        print("="*50)
        print("1. 查看回测结果列表")
        print("2. 查看回测图表")
        print("3. 运行快速回测示例")
        print("4. 退出")
        
        choice = input("\n请选择操作 (1-4): ").strip()
        
        if choice == "1":
            list_backtest_results()
        elif choice == "2":
            files = list_backtest_results()
            if files:
                try:
                    file_index = int(input(f"\n请输入要查看的图表编号 (1-{len(files)}): "))
                    view_chart(file_index)
                except ValueError:
                    print("请输入有效的数字。")
        elif choice == "3":
            run_quick_backtest()
        elif choice == "4":
            print("退出程序。")
            break
        else:
            print("无效的选择，请重新输入。")

if __name__ == "__main__":
    main()