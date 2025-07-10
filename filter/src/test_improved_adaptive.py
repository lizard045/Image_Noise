#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試改良版自適應濾波器
"""

import cv2
import numpy as np
import os
import sys
from improved_adaptive_filter import improved_adaptive_filter, brightness_aware_adaptive_filter, edge_preserving_adaptive_filter, calculate_psnr, calculate_cw_ssim

def test_improved_adaptive():
    """
    測試改良版自適應濾波器
    """
    # 使用現有的taipei101.png
    input_path = "taipei101.png"
    
    # 讀取影像
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ 無法讀取影像: {input_path}")
        return
    
    print(f"📸 開始測試改良版自適應濾波器")
    print(f"📏 影像尺寸: {image.shape}")
    
    # 創建輸出資料夾
    output_dir = "improved_adaptive_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原始影像
    cv2.imwrite(os.path.join(output_dir, '00_original.png'), image)
    
    # 1. 原始自適應濾波器
    print("\n🔄 測試原始自適應濾波器...")
    original_result = original_adaptive_filter(image)
    cv2.imwrite(os.path.join(output_dir, '01_original_adaptive.png'), original_result)
    
    # 2. 改良版自適應濾波器
    print("\n🔄 測試改良版自適應濾波器...")
    improved_result = improved_adaptive_filter(image)
    cv2.imwrite(os.path.join(output_dir, '02_improved_adaptive.png'), improved_result)
    
    # 3. 亮度感知自適應濾波器
    print("\n🔄 測試亮度感知自適應濾波器...")
    brightness_result = brightness_aware_adaptive_filter(image)
    cv2.imwrite(os.path.join(output_dir, '03_brightness_aware_adaptive.png'), brightness_result)
    
    # 4. 保邊緣自適應濾波器
    print("\n🔄 測試保邊緣自適應濾波器...")
    edge_result = edge_preserving_adaptive_filter(image)
    cv2.imwrite(os.path.join(output_dir, '04_edge_preserving_adaptive.png'), edge_result)
    
    # 品質評估
    print("\n📊 品質評估結果：")
    print("=" * 50)
    
    results = {
        '原始自適應': original_result,
        '改良自適應': improved_result,
        '亮度感知': brightness_result,
        '保邊緣': edge_result
    }
    
    for name, result in results.items():
        psnr = calculate_psnr(image, result)
        ssim = calculate_cw_ssim(image, result)
        print(f"{name:>8}: PSNR={psnr:6.2f}dB, SSIM={ssim:.4f}")
    
    print("\n🎉 測試完成！")
    print(f"📁 結果保存在: {output_dir}")
    
    # 分析亮部表現
    analyze_brightness_performance(image, results, output_dir)

def original_adaptive_filter(image, max_filter_size=5):
    """
    原始自適應濾波器（來自practical_denoising.py）
    """
    # 計算局部方差
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
    local_var = local_sqr_mean - local_mean**2
    
    # 基於局部方差選擇濾波器
    result = image.copy()
    
    # 高方差區域（邊緣）：使用較弱的濾波
    high_var_mask = local_var > np.percentile(local_var, 70)
    result[high_var_mask] = cv2.bilateralFilter(image, 5, 50, 50)[high_var_mask]
    
    # 低方差區域（平滑區域）：使用較強的濾波
    low_var_mask = local_var < np.percentile(local_var, 30)
    result[low_var_mask] = cv2.GaussianBlur(image, (5, 5), 1.5)[low_var_mask]
    
    return result

def analyze_brightness_performance(original, results, output_dir):
    """
    分析亮部表現
    """
    print("\n🔍 分析亮部表現...")
    
    # 轉換為灰階
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
    
    # 找出亮部區域
    bright_threshold = np.percentile(original_gray, 80)
    bright_mask = original_gray > bright_threshold
    
    print(f"📊 亮部區域佔比: {np.sum(bright_mask)/(bright_mask.shape[0]*bright_mask.shape[1])*100:.1f}%")
    print(f"📊 亮部閾值: {bright_threshold:.1f}")
    
    # 計算亮部區域的品質指標
    print("\n🌟 亮部區域品質評估：")
    print("=" * 40)
    
    for name, result in results.items():
        if len(result.shape) == 3:
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            result_gray = result
        
        # 只計算亮部區域的MSE
        bright_original = original_gray[bright_mask]
        bright_result = result_gray[bright_mask]
        
        mse = np.mean((bright_original.astype(np.float64) - bright_result.astype(np.float64)) ** 2)
        
        # 計算亮部區域的標準差（細節保持度）
        original_std = np.std(bright_original)
        result_std = np.std(bright_result)
        detail_preservation = result_std / original_std if original_std > 0 else 0
        
        print(f"{name:>8}: MSE={mse:6.2f}, 細節保持={detail_preservation:.3f}")
    
    # 創建亮部遮罩可視化
    bright_mask_vis = np.zeros_like(original_gray)
    bright_mask_vis[bright_mask] = 255
    cv2.imwrite(os.path.join(output_dir, '05_bright_mask.png'), bright_mask_vis)
    
    print(f"\n💾 亮部遮罩已保存: {os.path.join(output_dir, '05_bright_mask.png')}")

if __name__ == "__main__":
    test_improved_adaptive() 