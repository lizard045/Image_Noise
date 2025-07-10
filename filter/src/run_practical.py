#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
實用影像去噪工具執行腳本
提供友好的使用者介面來執行去噪處理
"""

import subprocess
import sys
import os
import argparse
import glob

def install_requirements():
    """
    安裝必要的套件
    """
    print("[INSTALL] 檢查並安裝依賴套件...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, encoding='utf-8', errors='ignore')
        print("[SUCCESS] 依賴套件已安裝")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 安裝依賴套件失敗: {e}")
        return False

def find_image_files(directory="."):
    """
    在指定目錄找尋影像檔案
    """
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
        image_files.extend(glob.glob(os.path.join(directory, ext.upper())))
    
    return sorted(image_files)

def interactive_mode():
    """
    互動模式
    """
    print("[INTERACTIVE] 實用影像去噪工具 - 互動模式")
    print("=" * 50)
    
    # 找尋影像檔案
    image_files = find_image_files()
    
    if not image_files:
        print("[ERROR] 當前目錄中沒有找到影像檔案")
        print("[INFO] 支援的格式: PNG, JPG, JPEG, BMP, TIFF")
        return
    
    print(f"[INFO] 找到 {len(image_files)} 個影像檔案：")
    for i, file in enumerate(image_files, 1):
        print(f"  {i:2d}. {file}")
    
    # 選擇影像
    while True:
        try:
            choice = input(f"\n[INPUT] 請選擇影像檔案 (1-{len(image_files)}): ")
            if choice.lower() in ['q', 'quit', 'exit']:
                print("[INFO] 再見！")
                return
            
            index = int(choice) - 1
            if 0 <= index < len(image_files):
                selected_image = image_files[index]
                break
            else:
                print(f"[ERROR] 請輸入 1-{len(image_files)} 之間的數字")
        except ValueError:
            print("[ERROR] 請輸入有效的數字")
    
    # 選擇濾波器
    filters = {
        '1': ('auto', '自動推薦'),
        '2': ('all', '所有濾波器'),
        '3': ('bilateral', '雙邊濾波器'),
        '4': ('non_local_means', '非局部均值濾波器'),
        '5': ('median', '中值濾波器'),
        '6': ('gaussian', '高斯濾波器'),
        '7': ('wiener', '維納濾波器'),
        '8': ('morphological', '形態學濾波器'),
        '9': ('adaptive', '自適應濾波器')
    }
    
    print("\n[SELECT] 選擇濾波器類型：")
    for key, (_, name) in filters.items():
        print(f"  {key}. {name}")
    
    while True:
        choice = input("\n[INPUT] 請選擇濾波器 (1-9): ")
        if choice.lower() in ['q', 'quit', 'exit']:
            print("[INFO] 再見！")
            return
        
        if choice in filters:
            selected_filter, filter_name = filters[choice]
            break
        else:
            print("[ERROR] 請輸入 1-9 之間的數字")
    
    # 執行處理
    print(f"\n[PROCESSING] 開始處理影像: {selected_image}")
    print(f"[PROCESSING] 使用濾波器: {filter_name}")
    
    cmd = [sys.executable, "practical_denoising.py", selected_image, "--filter", selected_filter]
    
    try:
        # 使用系統預設編碼處理中文輸出
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("[WARNING] 警告訊息:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("[SUCCESS] 處理完成！")
            print("[INFO] 提示：你可以在 'practical_results' 資料夾中查看結果")
        else:
            print("[ERROR] 處理失敗")
    
    except Exception as e:
        print(f"[ERROR] 執行失敗: {e}")

def batch_mode(image_path, filter_type='auto'):
    """
    批次處理模式
    """
    print(f"[BATCH] 批次處理模式")
    print(f"[BATCH] 影像: {image_path}")
    print(f"[BATCH] 濾波器: {filter_type}")
    print("=" * 50)
    
    if not os.path.exists(image_path):
        print(f"[ERROR] 找不到影像檔案: {image_path}")
        return
    
    cmd = [sys.executable, "practical_denoising.py", image_path, "--filter", filter_type]
    
    try:
        # 使用系統預設編碼處理中文輸出
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("[WARNING] 警告訊息:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("[SUCCESS] 處理完成！")
        else:
            print("[ERROR] 處理失敗")
    
    except Exception as e:
        print(f"[ERROR] 執行失敗: {e}")

def main():
    # 確保Windows系統能正確顯示中文
    if sys.platform == 'win32':
        os.system('chcp 65001 > nul')
    
    parser = argparse.ArgumentParser(description='實用影像去噪工具執行腳本')
    parser.add_argument('--image', help='影像檔案路徑 (批次模式)')
    parser.add_argument('--filter', choices=['auto', 'all', 'median', 'gaussian', 'bilateral', 'wiener', 'non_local_means', 'morphological', 'adaptive'], 
                       default='auto', help='濾波器類型')
    parser.add_argument('--no-install', action='store_true', help='跳過依賴套件安裝')
    
    args = parser.parse_args()
    
    # 檢查並安裝依賴套件
    if not args.no_install:
        if not install_requirements():
            return
    
    # 檢查主程式是否存在
    if not os.path.exists('practical_denoising.py'):
        print("[ERROR] 找不到 practical_denoising.py 檔案")
        return
    
    if args.image:
        # 批次模式
        batch_mode(args.image, args.filter)
    else:
        # 互動模式
        interactive_mode()

if __name__ == "__main__":
    main() 