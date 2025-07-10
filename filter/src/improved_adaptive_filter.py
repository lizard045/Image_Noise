#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改良版自適應濾波器
特別針對亮部表現問題進行優化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, metrics, filters
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means
import os
import sys
import math

def calculate_psnr(original, processed):
    """計算PSNR指標"""
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        processed_gray = processed
    
    mse = np.mean((original_gray.astype(np.float64) - processed_gray.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_cw_ssim(original, processed):
    """計算CW-SSIM指標"""
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        processed_gray = processed
    
    ssim_score = metrics.structural_similarity(original_gray, processed_gray, 
                                              data_range=processed_gray.max() - processed_gray.min())
    return ssim_score

def improved_adaptive_filter(image, preserve_brightness=True, detail_enhancement=True):
    """
    改良版自適應濾波器
    
    主要改進：
    1. 考慮亮度信息，避免過度平滑亮部
    2. 使用多層次判斷而非單一方差判斷
    3. 針對亮部區域使用保細節的濾波方法
    4. 可選的細節增強後處理
    """
    print("🔧 開始改良版自適應濾波處理...")
    
    # 轉換為灰階計算局部統計
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 計算局部統計
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
    local_var = local_sqr_mean - local_mean**2
    
    # 計算梯度資訊
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    result = image.copy()
    
    # 分析影像特性
    high_var_threshold = np.percentile(local_var, 75)
    low_var_threshold = np.percentile(local_var, 25)
    high_brightness_threshold = np.percentile(local_mean, 80)
    high_gradient_threshold = np.percentile(gradient_magnitude, 80)
    
    print(f"📊 影像分析：")
    print(f"   高方差閾值: {high_var_threshold:.2f}")
    print(f"   低方差閾值: {low_var_threshold:.2f}")
    print(f"   高亮度閾值: {high_brightness_threshold:.2f}")
    print(f"   高梯度閾值: {high_gradient_threshold:.2f}")
    
    # 建立不同的區域遮罩
    edge_mask = (local_var > high_var_threshold) & (gradient_magnitude > high_gradient_threshold)
    bright_smooth_mask = (local_mean > high_brightness_threshold) & (local_var < low_var_threshold)
    dark_smooth_mask = (local_mean <= high_brightness_threshold) & (local_var < low_var_threshold)
    medium_mask = ~(edge_mask | bright_smooth_mask | dark_smooth_mask)
    
    print(f"🎯 區域分析：")
    print(f"   邊緣區域: {np.sum(edge_mask)/(edge_mask.shape[0]*edge_mask.shape[1])*100:.1f}%")
    print(f"   亮部平滑: {np.sum(bright_smooth_mask)/(bright_smooth_mask.shape[0]*bright_smooth_mask.shape[1])*100:.1f}%")
    print(f"   暗部平滑: {np.sum(dark_smooth_mask)/(dark_smooth_mask.shape[0]*dark_smooth_mask.shape[1])*100:.1f}%")
    print(f"   中間區域: {np.sum(medium_mask)/(medium_mask.shape[0]*medium_mask.shape[1])*100:.1f}%")
    
    # 1. 邊緣區域：使用輕微的雙邊濾波保持邊緣
    if np.sum(edge_mask) > 0:
        print("🔄 處理邊緣區域...")
        result[edge_mask] = cv2.bilateralFilter(image, 5, 30, 30)[edge_mask]
    
    # 2. 亮部平滑區域：使用保細節的非局部均值濾波
    if np.sum(bright_smooth_mask) > 0:
        print("✨ 處理亮部平滑區域...")
        if preserve_brightness:
            # 使用較溫和的非局部均值濾波，保持亮部細節
            if len(image.shape) == 3:
                bright_filtered = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 15)
            else:
                bright_filtered = cv2.fastNlMeansDenoising(image, None, 5, 7, 15)
            result[bright_smooth_mask] = bright_filtered[bright_smooth_mask]
        else:
            # 使用輕微的高斯濾波
            result[bright_smooth_mask] = cv2.GaussianBlur(image, (3, 3), 0.8)[bright_smooth_mask]
    
    # 3. 暗部平滑區域：使用較強的去噪
    if np.sum(dark_smooth_mask) > 0:
        print("🌙 處理暗部平滑區域...")
        result[dark_smooth_mask] = cv2.GaussianBlur(image, (5, 5), 1.2)[dark_smooth_mask]
    
    # 4. 中間區域：使用平衡的濾波
    if np.sum(medium_mask) > 0:
        print("⚖️ 處理中間區域...")
        result[medium_mask] = cv2.bilateralFilter(image, 7, 50, 50)[medium_mask]
    
    # 細節增強後處理
    if detail_enhancement:
        print("🎨 進行細節增強...")
        result = enhance_details(result, image)
    
    print("✅ 改良版自適應濾波完成！")
    return result

def enhance_details(filtered_image, original_image):
    """
    細節增強後處理
    恢復一些在濾波過程中丟失的細節
    """
    # 計算細節資訊
    if len(original_image.shape) == 3:
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        filtered_gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original_image
        filtered_gray = filtered_image
    
    # 使用高通濾波器提取細節
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    original_details = cv2.filter2D(original_gray.astype(np.float32), -1, kernel)
    filtered_details = cv2.filter2D(filtered_gray.astype(np.float32), -1, kernel)
    
    # 計算細節增強權重
    detail_weight = 0.3
    enhanced_details = original_details * detail_weight + filtered_details * (1 - detail_weight)
    
    # 將細節加回濾波後的影像
    if len(filtered_image.shape) == 3:
        result = filtered_image.copy().astype(np.float32)
        for i in range(3):
            result[:, :, i] += enhanced_details * 0.1
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = filtered_image.astype(np.float32) + enhanced_details * 0.1
        return np.clip(result, 0, 255).astype(np.uint8)

def brightness_aware_adaptive_filter(image, brightness_preservation=0.8):
    """
    考慮亮度的自適應濾波器
    特別針對亮部區域進行優化
    """
    print("🌟 開始亮度感知自適應濾波...")
    
    # 轉換為LAB色彩空間進行亮度分析
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
    else:
        l_channel = image
    
    # 計算亮度分佈
    brightness_hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
    brightness_percentiles = np.percentile(l_channel, [20, 50, 80])
    
    print(f"📊 亮度分析：")
    print(f"   20%亮度: {brightness_percentiles[0]:.1f}")
    print(f"   50%亮度: {brightness_percentiles[1]:.1f}")
    print(f"   80%亮度: {brightness_percentiles[2]:.1f}")
    
    # 建立亮度遮罩
    dark_mask = l_channel < brightness_percentiles[0]
    medium_mask = (l_channel >= brightness_percentiles[0]) & (l_channel < brightness_percentiles[2])
    bright_mask = l_channel >= brightness_percentiles[2]
    
    result = image.copy()
    
    # 暗部：使用較強的去噪
    if np.sum(dark_mask) > 0:
        print("🌑 處理暗部區域...")
        result[dark_mask] = cv2.bilateralFilter(image, 9, 80, 80)[dark_mask]
    
    # 中間亮度：使用平衡的濾波
    if np.sum(medium_mask) > 0:
        print("🌗 處理中間亮度區域...")
        result[medium_mask] = cv2.bilateralFilter(image, 7, 60, 60)[medium_mask]
    
    # 亮部：使用保細節的濾波
    if np.sum(bright_mask) > 0:
        print("🌕 處理亮部區域...")
        # 使用非局部均值濾波保持亮部細節
        if len(image.shape) == 3:
            bright_filtered = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 15)
        else:
            bright_filtered = cv2.fastNlMeansDenoising(image, None, 6, 7, 15)
        
        # 根據亮度保持係數混合原始和濾波後的影像
        result[bright_mask] = (brightness_preservation * image[bright_mask] + 
                              (1 - brightness_preservation) * bright_filtered[bright_mask]).astype(np.uint8)
    
    print("✅ 亮度感知自適應濾波完成！")
    return result

def edge_preserving_adaptive_filter(image, edge_threshold=50):
    """
    保邊緣的自適應濾波器
    使用更精確的邊緣檢測來保護細節
    """
    print("🔍 開始保邊緣自適應濾波...")
    
    # 轉換為灰階
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 使用Canny邊緣檢測
    edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
    
    # 膨脹邊緣遮罩以保護邊緣附近區域
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # 建立遮罩
    edge_mask = edges_dilated > 0
    smooth_mask = ~edge_mask
    
    print(f"🎯 邊緣分析：")
    print(f"   邊緣區域: {np.sum(edge_mask)/(edge_mask.shape[0]*edge_mask.shape[1])*100:.1f}%")
    print(f"   平滑區域: {np.sum(smooth_mask)/(smooth_mask.shape[0]*smooth_mask.shape[1])*100:.1f}%")
    
    result = image.copy()
    
    # 邊緣區域：保持原始或使用非常輕微的濾波
    if np.sum(edge_mask) > 0:
        print("🏔️ 保護邊緣區域...")
        result[edge_mask] = (0.8 * image[edge_mask] + 
                           0.2 * cv2.bilateralFilter(image, 3, 20, 20)[edge_mask]).astype(np.uint8)
    
    # 平滑區域：使用適度的濾波
    if np.sum(smooth_mask) > 0:
        print("🌊 處理平滑區域...")
        if len(image.shape) == 3:
            smooth_filtered = cv2.fastNlMeansDenoisingColored(image, None, 8, 8, 7, 21)
        else:
            smooth_filtered = cv2.fastNlMeansDenoising(image, None, 8, 7, 21)
        result[smooth_mask] = smooth_filtered[smooth_mask]
    
    print("✅ 保邊緣自適應濾波完成！")
    return result

def compare_adaptive_filters(image):
    """
    比較不同的自適應濾波器
    """
    print("🔄 開始比較不同的自適應濾波器...")
    
    # 原始自適應濾波器（來自practical_denoising.py）
    original_adaptive = original_adaptive_filter(image)
    
    # 改良版自適應濾波器
    improved_adaptive = improved_adaptive_filter(image)
    
    # 亮度感知自適應濾波器
    brightness_aware = brightness_aware_adaptive_filter(image)
    
    # 保邊緣自適應濾波器
    edge_preserving = edge_preserving_adaptive_filter(image)
    
    return {
        'original': original_adaptive,
        'improved': improved_adaptive,
        'brightness_aware': brightness_aware,
        'edge_preserving': edge_preserving
    }

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

def main():
    """
    主程式
    """
    if len(sys.argv) < 2:
        print("使用方法: python improved_adaptive_filter.py <input_image>")
        print("範例: python improved_adaptive_filter.py taipei101.png")
        return
    
    input_path = sys.argv[1]
    
    # 讀取影像
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ 無法讀取影像: {input_path}")
        return
    
    print(f"📸 開始處理影像: {input_path}")
    print(f"📏 影像尺寸: {image.shape}")
    
    # 創建輸出資料夾
    output_dir = "improved_adaptive_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 比較不同的自適應濾波器
    results = compare_adaptive_filters(image)
    
    # 保存結果
    cv2.imwrite(os.path.join(output_dir, '00_original.png'), image)
    
    for name, result in results.items():
        filename = os.path.join(output_dir, f'{name}_adaptive.png')
        cv2.imwrite(filename, result)
        print(f"💾 已保存: {filename}")
        
        # 計算品質指標
        psnr = calculate_psnr(image, result)
        ssim = calculate_cw_ssim(image, result)
        print(f"📊 {name} 品質指標: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")
    
    print("🎉 處理完成！")
    print(f"📁 結果保存在: {output_dir}")

if __name__ == "__main__":
    main() 