#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
實用影像去噪工具
直接對照片進行去噪處理，不需要先加入噪點
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, metrics, filters
import os
import argparse
from PIL import Image
import sys
import math

def calculate_cw_ssim(original, processed):
    """
    計算CW-SSIM指標
    """
    if len(original.shape) == 3:
        # 彩色影像，轉換為灰階
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        processed_gray = processed
    
    # 使用skimage的SSIM
    ssim_score = metrics.structural_similarity(original_gray, processed_gray, 
                                              data_range=processed_gray.max() - processed_gray.min())
    return ssim_score

def calculate_psnr(original, processed):
    """
    計算PSNR指標
    """
    if len(original.shape) == 3:
        # 彩色影像，轉換為灰階
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        processed_gray = processed
    
    # 計算MSE（均方誤差）
    mse = np.mean((original_gray.astype(np.float64) - processed_gray.astype(np.float64)) ** 2)
    
    # 如果MSE為0，代表影像完全相同
    if mse == 0:
        return float('inf')
    
    # 計算PSNR
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    
    return psnr

def evaluate_image_quality(original, processed, filter_name):
    """
    評估影像品質並顯示結果
    """
    ssim_score = calculate_cw_ssim(original, processed)
    psnr_score = calculate_psnr(original, processed)
    
    print(f"[METRICS] {filter_name} 品質評估:")
    print(f"  CW-SSIM: {ssim_score:.6f} (越接近1.0越好)")
    print(f"  PSNR: {psnr_score:.2f} dB (越高越好)")
    
    # 品質解釋
    if ssim_score > 0.9:
        ssim_quality = "優秀"
    elif ssim_score > 0.8:
        ssim_quality = "良好"
    elif ssim_score > 0.7:
        ssim_quality = "一般"
    else:
        ssim_quality = "較差"
    
    if psnr_score > 30:
        psnr_quality = "優秀"
    elif psnr_score > 25:
        psnr_quality = "良好"
    elif psnr_score > 20:
        psnr_quality = "一般"
    else:
        psnr_quality = "較差"
    
    print(f"  結構相似度: {ssim_quality}, 噪聲比: {psnr_quality}")
    
    return ssim_score, psnr_score

def average_filter(image, kernel_size=5):
    """
    平均濾波器 - 適合處理隨機噪聲
    """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)

def median_filter(image, kernel_size=5):
    """
    中值濾波器 - 適合處理椒鹽噪聲和脈衝噪聲
    """
    return cv2.medianBlur(image, kernel_size)

def gaussian_filter(image, kernel_size=5, sigma=1.0):
    """
    高斯濾波器 - 適合處理高斯噪聲
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    雙邊濾波器 - 保邊緣去噪，適合自然影像
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def wiener_filter(image, noise_var=0.01):
    """
    維納濾波器 - 適合處理已知的噪聲特性
    """
    if len(image.shape) == 3:
        # 彩色影像，對每個通道分別處理
        result = np.zeros_like(image, dtype=np.float32)
        # 創建一個簡單的點擴散函數（PSF）
        psf = np.ones((5, 5)) / 25
        for i in range(3):
            channel = image[:, :, i].astype(np.float32) / 255.0
            result[:, :, i] = restoration.wiener(channel, psf, noise_var)
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
    else:
        # 灰階影像
        psf = np.ones((5, 5)) / 25
        normalized = image.astype(np.float32) / 255.0
        result = restoration.wiener(normalized, psf, noise_var)
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

def non_local_means_filter(image, h=10, template_window_size=7, search_window_size=21):
    """
    非局部均值濾波器 - 適合保持紋理細節
    """
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)
    else:
        return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)

def morphological_filter(image, kernel_size=3):
    """
    形態學濾波器 - 適合處理結構化噪聲
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def adaptive_filter(image, max_filter_size=5):
    """
    自適應濾波器 - 根據局部統計自動調整
    """
    # 計算局部方差
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
    local_sqr_mean = cv2.filter2D((image.astype(np.float32))**2, -1, kernel)
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

def calculate_noise_metrics(image):
    """
    計算影像的噪聲度量指標
    """
    # 轉換為灰階
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 計算局部方差
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
    local_var = local_sqr_mean - local_mean**2
    
    # 計算梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 噪聲指標
    noise_level = np.std(local_var)
    edge_density = np.mean(gradient_magnitude > np.percentile(gradient_magnitude, 90))
    
    return {
        'noise_level': noise_level,
        'edge_density': edge_density,
        'mean_variance': np.mean(local_var),
        'gradient_strength': np.mean(gradient_magnitude)
    }

def recommend_filter(image):
    """
    根據影像特徵推薦最佳濾波器
    """
    metrics = calculate_noise_metrics(image)
    
    print(f"[ANALYSIS] 影像分析結果：")
    print(f"   噪聲等級: {metrics['noise_level']:.2f}")
    print(f"   邊緣密度: {metrics['edge_density']:.3f}")
    print(f"   平均方差: {metrics['mean_variance']:.2f}")
    print(f"   梯度強度: {metrics['gradient_strength']:.2f}")
    
    # 根據指標推薦濾波器
    if metrics['noise_level'] > 1000:
        return "median", "檢測到脈衝噪聲，推薦中值濾波器"
    elif metrics['edge_density'] > 0.1:
        return "bilateral", "檢測到豐富的邊緣細節，推薦雙邊濾波器"
    elif metrics['noise_level'] > 500:
        return "non_local_means", "檢測到中等噪聲，推薦非局部均值濾波器"
    elif metrics['gradient_strength'] < 20:
        return "gaussian", "檢測到平滑影像，推薦高斯濾波器"
    else:
        return "adaptive", "使用自適應濾波器處理複雜噪聲"

def save_image(image, filename):
    """
    保存影像
    """
    cv2.imwrite(filename, image)
    print(f"[SAVED] 已保存: {filename}")

def main():
    # 確保Windows系統能正確顯示中文
    if sys.platform == 'win32':
        os.system('chcp 65001 > nul')
    
    parser = argparse.ArgumentParser(description='實用影像去噪工具')
    parser.add_argument('input_image', help='輸入影像路徑')
    parser.add_argument('--filter', choices=['all', 'median', 'gaussian', 'bilateral', 'wiener', 'non_local_means', 'morphological', 'adaptive', 'auto'], 
                       default='auto', help='選擇濾波器類型')
    parser.add_argument('--output_dir', default='practical_results', help='輸出資料夾')
    parser.add_argument('--kernel_size', type=int, default=5, help='濾波器核心大小')
    parser.add_argument('--no-metrics', action='store_true', help='跳過品質評估')
    
    args = parser.parse_args()
    
    # 讀取影像
    image = cv2.imread(args.input_image)
    if image is None:
        print(f"[ERROR] 無法讀取影像: {args.input_image}")
        return
    
    print(f"[INFO] 開始處理影像: {args.input_image}")
    print(f"[INFO] 影像尺寸: {image.shape}")
    
    # 創建輸出資料夾
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存原始影像
    original_filename = os.path.join(args.output_dir, '00_original.png')
    save_image(image, original_filename)
    
    # 定義濾波器
    filters_dict = {
        'median': lambda img: median_filter(img, args.kernel_size),
        'gaussian': lambda img: gaussian_filter(img, args.kernel_size, 1.0),
        'bilateral': lambda img: bilateral_filter(img, 9, 75, 75),
        'wiener': lambda img: wiener_filter(img, 0.01),
        'non_local_means': lambda img: non_local_means_filter(img, 10, 7, 21),
        'morphological': lambda img: morphological_filter(img, 3),
        'adaptive': lambda img: adaptive_filter(img, args.kernel_size)
    }
    
    filter_names = {
        'median': '中值濾波器',
        'gaussian': '高斯濾波器', 
        'bilateral': '雙邊濾波器',
        'wiener': '維納濾波器',
        'non_local_means': '非局部均值濾波器',
        'morphological': '形態學濾波器',
        'adaptive': '自適應濾波器'
    }
    
    # 用於儲存評估結果
    evaluation_results = []
    
    if args.filter == 'auto':
        # 自動推薦濾波器
        recommended_filter, reason = recommend_filter(image)
        print(f"\n[AUTO] 自動推薦: {filter_names[recommended_filter]}")
        print(f"[AUTO] 推薦理由: {reason}")
        
        # 應用推薦的濾波器
        print(f"\n[PROCESSING] 正在應用 {filter_names[recommended_filter]}...")
        filtered_image = filters_dict[recommended_filter](image)
        output_filename = os.path.join(args.output_dir, f'01_{recommended_filter}_filtered.png')
        save_image(filtered_image, output_filename)
        
        # 評估品質
        if not args.no_metrics:
            ssim_score, psnr_score = evaluate_image_quality(image, filtered_image, filter_names[recommended_filter])
            evaluation_results.append((filter_names[recommended_filter], ssim_score, psnr_score))
        
        print(f"\n[SUCCESS] 處理完成！")
        print(f"[SUCCESS] 結果保存在: {args.output_dir}")
        
    elif args.filter == 'all':
        # 應用所有濾波器
        print("\n[PROCESSING] 正在應用所有濾波器...")
        for i, (filter_key, filter_func) in enumerate(filters_dict.items(), 1):
            print(f"[PROCESSING] 處理 {filter_names[filter_key]}...")
            try:
                filtered_image = filter_func(image)
                output_filename = os.path.join(args.output_dir, f'{i:02d}_{filter_key}_filtered.png')
                save_image(filtered_image, output_filename)
                
                # 評估品質
                if not args.no_metrics:
                    ssim_score, psnr_score = evaluate_image_quality(image, filtered_image, filter_names[filter_key])
                    evaluation_results.append((filter_names[filter_key], ssim_score, psnr_score))
                
            except Exception as e:
                print(f"[WARNING] {filter_names[filter_key]} 處理失敗: {e}")
        
        print(f"\n[SUCCESS] 所有濾波器處理完成！")
        print(f"[SUCCESS] 結果保存在: {args.output_dir}")
        
    else:
        # 應用特定濾波器
        if args.filter in filters_dict:
            print(f"\n[PROCESSING] 正在應用 {filter_names[args.filter]}...")
            filtered_image = filters_dict[args.filter](image)
            output_filename = os.path.join(args.output_dir, f'01_{args.filter}_filtered.png')
            save_image(filtered_image, output_filename)
            
            # 評估品質
            if not args.no_metrics:
                ssim_score, psnr_score = evaluate_image_quality(image, filtered_image, filter_names[args.filter])
                evaluation_results.append((filter_names[args.filter], ssim_score, psnr_score))
            
            print(f"\n[SUCCESS] 處理完成！")
            print(f"[SUCCESS] 結果保存在: {args.output_dir}")
        else:
            print(f"[ERROR] 未知的濾波器: {args.filter}")
    
    # 顯示評估結果總結
    if evaluation_results and not args.no_metrics:
        print("\n" + "="*70)
        print(" 品質評估結果總結")
        print("="*70)
        print(f"{'濾波器':<15} {'CW-SSIM':<10} {'PSNR(dB)':<10} {'總體評價':<10}")
        print("-"*70)
        
        for filter_name, ssim, psnr in evaluation_results:
            # 綜合評價
            if ssim > 0.85 and psnr > 28:
                overall = "優秀"
            elif ssim > 0.75 and psnr > 25:
                overall = "良好"
            elif ssim > 0.65 and psnr > 22:
                overall = "一般"
            else:
                overall = "需改進"
            
            print(f"{filter_name:<15} {ssim:.4f}    {psnr:.2f}     {overall:<10}")
        
        # 推薦最佳濾波器
        if len(evaluation_results) > 1:
            # 綜合評分 = 0.6 * SSIM + 0.4 * (PSNR/40)
            best_filter = max(evaluation_results, key=lambda x: 0.6 * x[1] + 0.4 * (x[2]/40))
            print(f"\n[BEST] 推薦最佳濾波器: {best_filter[0]}")
            print(f"   CW-SSIM: {best_filter[1]:.4f}, PSNR: {best_filter[2]:.2f} dB")
        
        print("="*70)
        print("[INFO] 評估說明:")
        print("   CW-SSIM: 結構相似性指數，範圍0-1，越接近1越好")
        print("   PSNR: 峰值信噪比，單位dB，通常>25dB為良好品質")
        print("   評估基於原始影像與處理後影像的比較")

if __name__ == "__main__":
    main() 