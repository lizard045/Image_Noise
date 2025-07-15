#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import torch
import numpy as np
from collections import OrderedDict
import cv2

from utils import utils_logger
from utils import utils_image as util
from models.network_unet import UNetRes

"""
台北101影像去噪腳本
使用DRUNet模型進行彩色影像去噪處理

使用方法:
python taipei101_denoise.py --input_dir testsets/taipei101 --output_dir taipei101_color_denoised

"""

def advanced_detail_enhancement(denoised_img, original_img):
    """
    進階細節增強處理
    """
    # 轉換為灰階分析
    if len(original_img.shape) == 3:
        gray_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gray_denoised = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_orig = original_img
        gray_denoised = denoised_img
    
    # 計算局部統計
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray_orig.astype(np.float32), -1, kernel)
    local_sqr_mean = cv2.filter2D((gray_orig.astype(np.float32))**2, -1, kernel)
    local_var = local_sqr_mean - local_mean**2
    
    # 計算梯度資訊
    grad_x = cv2.Sobel(gray_orig, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_orig, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 建立不同區域的遮罩
    high_var_threshold = np.percentile(local_var, 75)
    high_brightness_threshold = np.percentile(local_mean, 80)
    high_gradient_threshold = np.percentile(gradient_magnitude, 80)
    
    # 邊緣區域：保持更多原始細節
    edge_mask = (local_var > high_var_threshold) & (gradient_magnitude > high_gradient_threshold)
    
    # 亮部區域：特別保護
    bright_mask = local_mean > high_brightness_threshold
    
    result = denoised_img.copy().astype(np.float32)
    
    # 對邊緣區域恢復更多細節
    if np.sum(edge_mask) > 0:
        edge_weight = 0.6  # 增加邊緣細節權重
        for i in range(denoised_img.shape[2] if len(denoised_img.shape) == 3 else 1):
            if len(denoised_img.shape) == 3:
                result[:, :, i][edge_mask] = (
                    edge_weight * original_img[:, :, i][edge_mask] + 
                    (1 - edge_weight) * denoised_img[:, :, i][edge_mask]
                )
            else:
                result[edge_mask] = (
                    edge_weight * original_img[edge_mask] + 
                    (1 - edge_weight) * denoised_img[edge_mask]
                )
    
    # 對亮部區域特別處理
    if np.sum(bright_mask) > 0:
        bright_weight = 0.4  # 亮部細節權重
        for i in range(denoised_img.shape[2] if len(denoised_img.shape) == 3 else 1):
            if len(denoised_img.shape) == 3:
                result[:, :, i][bright_mask] = (
                    bright_weight * original_img[:, :, i][bright_mask] + 
                    (1 - bright_weight) * denoised_img[:, :, i][bright_mask]
                )
            else:
                result[bright_mask] = (
                    bright_weight * original_img[bright_mask] + 
                    (1 - bright_weight) * denoised_img[bright_mask]
                )
    
    return np.clip(result, 0, 255).astype(np.uint8)

def adaptive_noise_level_processing(model, img_L, device, base_noise_level=10.0):
    """
    根據影像區域特性調整雜訊等級
    """
    # 轉換為numpy進行分析
    img_np = util.tensor2uint(img_L)
    
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_np
    
    # 計算局部方差
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
    local_var = local_sqr_mean - local_mean**2
    
    # 根據局部方差調整雜訊等級
    # 高方差區域（邊緣）使用較低的雜訊等級
    # 低方差區域（平滑區域）使用較高的雜訊等級
    var_percentiles = np.percentile(local_var, [25, 75])
    
    results = []
    noise_levels = []
    
    # 處理不同雜訊等級
    for noise_factor in [0.7, 1.0, 1.3]:  # 低、中、高雜訊等級
        current_noise_level = base_noise_level * noise_factor
        noise_level_normalized = current_noise_level / 255.0
        
        noise_level_tensor = torch.full((1, 1, img_L.shape[2], img_L.shape[3]), 
                                      noise_level_normalized).to(device)
        img_input = torch.cat([img_L, noise_level_tensor], dim=1)
        
        with torch.no_grad():
            img_E = model(img_input)
            results.append(util.tensor2uint(img_E))
            noise_levels.append(current_noise_level)
    
    # 根據局部特性混合結果
    final_result = np.zeros_like(results[0])
    
    # 低方差區域使用高雜訊等級處理結果
    low_var_mask = local_var < var_percentiles[0]
    final_result[low_var_mask] = results[2][low_var_mask]  # 高雜訊等級
    
    # 高方差區域使用低雜訊等級處理結果
    high_var_mask = local_var > var_percentiles[1]
    final_result[high_var_mask] = results[0][high_var_mask]  # 低雜訊等級
    
    # 中間區域使用中等雜訊等級
    medium_mask = ~(low_var_mask | high_var_mask)
    final_result[medium_mask] = results[1][medium_mask]  # 中等雜訊等級
    
    return final_result

def multi_scale_denoising(model, img_L, device, noise_level_normalized):
    """
    多尺度去噪處理
    """
    results = []
    scales = [1.0, 0.8, 0.6]  # 不同縮放比例
    
    for scale in scales:
        if scale != 1.0:
            # 縮放影像
            h, w = img_L.shape[2], img_L.shape[3]
            new_h, new_w = int(h * scale), int(w * scale)
            img_scaled = util.imresize_np(img_L, [new_h, new_w], True) # 使用util.imresize_np
        else:
            img_scaled = img_L
        
        # 添加雜訊等級資訊
        noise_level_tensor = torch.full((1, 1, img_scaled.shape[2], img_scaled.shape[3]), 
                                      noise_level_normalized).to(device)
        img_input = torch.cat([img_scaled, noise_level_tensor], dim=1)
        
        # 去噪處理
        with torch.no_grad():
            img_E = model(img_input)
        
        # 如果縮放過，需要恢復原始尺寸
        if scale != 1.0:
            img_E = util.imresize_np(img_E, [h, w], True) # 使用util.imresize_np
        
        results.append(util.tensor2uint(img_E))
    
    # 加權融合不同尺度的結果
    weights = [0.5, 0.3, 0.2]  # 原尺寸權重最高
    final_result = np.zeros_like(results[0]).astype(np.float32)
    
    for result, weight in zip(results, weights):
        final_result += result.astype(np.float32) * weight
    
    return np.clip(final_result, 0, 255).astype(np.uint8)

def frequency_domain_detail_preservation(denoised_img, original_img, detail_ratio=0.3):
    """
    在頻域中保護高頻細節
    """
    # 轉換為灰階進行頻域分析
    if len(original_img.shape) == 3:
        gray_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gray_denoised = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_orig = original_img
        gray_denoised = denoised_img
    
    # FFT變換
    f_orig = np.fft.fft2(gray_orig.astype(np.float32))
    f_denoised = np.fft.fft2(gray_denoised.astype(np.float32))
    
    # 創建高通濾波器
    rows, cols = gray_orig.shape
    center_row, center_col = rows // 2, cols // 2
    
    # 創建頻域遮罩
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - center_row)**2 + (x - center_col)**2)
    
    # 高頻遮罩（保留高頻細節）
    high_freq_mask = distance > min(rows, cols) * 0.1
    
    # 在高頻部分混合原始和去噪後的頻譜
    f_enhanced = f_denoised.copy()
    f_enhanced[high_freq_mask] = (
        (1 - detail_ratio) * f_denoised[high_freq_mask] + 
        detail_ratio * f_orig[high_freq_mask]
    )
    
    # 逆FFT變換
    enhanced_gray = np.real(np.fft.ifft2(f_enhanced))
    enhanced_gray = np.clip(enhanced_gray, 0, 255).astype(np.uint8)
    
    # 如果是彩色影像，需要將細節增強應用到彩色影像
    if len(denoised_img.shape) == 3:
        # 計算細節差異
        detail_diff = enhanced_gray.astype(np.float32) - gray_denoised.astype(np.float32)
        
        # 將細節差異應用到每個顏色通道
        result = denoised_img.astype(np.float32)
        for i in range(3):
            result[:, :, i] += detail_diff * 0.8  # 調整細節強度
        
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        return enhanced_gray

def main():
    # ----------------------------------------
    # 參數設定
    # ----------------------------------------
    parser = argparse.ArgumentParser(description='台北101影像去噪處理')
    parser.add_argument('--model_path', type=str, default='model_zoo/drunet_color.pth',
                       help='去噪模型路徑')
    parser.add_argument('--input_dir', type=str, default='testsets/taipei101',
                       help='輸入影像目錄')
    parser.add_argument('--output_dir', type=str, default='taipei101_color_denoised',
                       help='輸出結果目錄')
    parser.add_argument('--noise_level', type=float, default=10.0,
                       help='雜訊等級 (0-255)')
    parser.add_argument('--device', type=str, default='auto',
                       help='運算設備: auto, cpu, cuda')
    args = parser.parse_args()

    # ----------------------------------------
    # 建立輸出目錄和日誌
    # ----------------------------------------
    util.mkdir(args.output_dir)
    
    logger_name = 'taipei101_denoise'
    utils_logger.logger_info(logger_name, log_path=os.path.join(args.output_dir, 'denoise.log'))
    logger = logging.getLogger(logger_name)
    
    logger.info('=== 台北101影像去噪處理 ===')
    logger.info(f'模型路徑: {args.model_path}')
    logger.info(f'輸入目錄: {args.input_dir}')
    logger.info(f'輸出目錄: {args.output_dir}')
    logger.info(f'雜訊等級: {args.noise_level}')
    
    # ----------------------------------------
    # 設備設定
    # ----------------------------------------
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f'使用設備: {device}')
    
    # ----------------------------------------
    # 檢查路徑是否存在
    # ----------------------------------------
    if not os.path.exists(args.model_path):
        logger.error(f'模型檔案不存在: {args.model_path}')
        print(f" 錯誤: 找不到模型檔案 {args.model_path}")
        return
    
    if not os.path.exists(args.input_dir):
        logger.error(f'輸入目錄不存在: {args.input_dir}')
        print(f" 錯誤: 找不到輸入目錄 {args.input_dir}")
        return
    
    # ----------------------------------------
    # 載入模型
    # ----------------------------------------
    try:
        logger.info('載入DRUNet彩色去噪模型...')
        model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', 
                       downsample_mode="strideconv", upsample_mode="convtranspose", bias=False)
        
        # 載入預訓練權重
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=True)
        model.eval()
        
        # 禁用梯度計算
        for k, v in model.named_parameters():
            v.requires_grad = False
        
        model = model.to(device)
        
        # 計算模型參數數量
        num_params = sum(map(lambda x: x.numel(), model.parameters()))
        logger.info(f'模型參數數量: {num_params:,}')
        
    except Exception as e:
        logger.error(f'載入模型失敗: {str(e)}')
        print(f" 模型載入失敗: {str(e)}")
        return
    
    # ----------------------------------------
    # 取得影像檔案列表
    # ----------------------------------------
    image_paths = util.get_image_paths(args.input_dir)
    logger.info(f'找到影像檔案數量: {len(image_paths)}')
    
    if len(image_paths) == 0:
        logger.warning('輸入目錄中沒有找到影像檔案')
        print("  警告: 沒有找到任何影像檔案")
        return
    
    # ----------------------------------------
    # 正規化雜訊等級
    # ----------------------------------------
    noise_level_normalized = args.noise_level / 255.0
    logger.info(f'正規化雜訊等級: {noise_level_normalized:.4f}')
    
    # ----------------------------------------
    # 處理影像
    # ----------------------------------------
    logger.info('開始處理影像...')
    print(f"  開始處理 {len(image_paths)} 張影像")
    
    processed_count = 0
    failed_count = 0
    
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        
        try:
            logger.info(f'處理影像 {idx+1}/{len(image_paths)}: {img_name}')
            
            # 載入影像
            img_L = util.imread_uint(img_path, n_channels=3)
            original_shape = img_L.shape
            
            # 轉換為tensor
            img_tensor = util.uint2tensor4(img_L)
            img_tensor = img_tensor.to(device)
            
            # 使用自適應雜訊等級處理
            img_E = adaptive_noise_level_processing(model, img_tensor, device, args.noise_level)
            
            # 或者使用多尺度處理（根據需要選擇）
            # noise_level_normalized = args.noise_level / 255.0
            # img_E = multi_scale_denoising(model, img_tensor, device, noise_level_normalized)
            
            # 確保輸出影像尺寸正確
            if img_E.shape != original_shape:
                logger.warning(f'{img_name} - 輸出尺寸不匹配: {img_E.shape} vs {original_shape}')
            
            # 進階細節增強
            img_E = advanced_detail_enhancement(img_E, img_L)
            
            # 頻域細節保護
            img_E = frequency_domain_detail_preservation(img_E, img_L, detail_ratio=0.2)
            
            # 儲存結果
            output_path = os.path.join(args.output_dir, img_name)
            util.imsave(img_E, output_path)
            
            processed_count += 1
            print(f" ({idx+1}/{len(image_paths)}) {img_name} - 處理完成")
            
        except Exception as e:
            failed_count += 1
            logger.error(f'{img_name} - 處理失敗: {str(e)}')
            print(f" ({idx+1}/{len(image_paths)}) {img_name} - 處理失敗: {str(e)}")
            continue
    
    # ----------------------------------------
    # 處理結果統計
    # ----------------------------------------
    logger.info('\n' + '=' * 60)
    logger.info(' 處理結果統計')
    logger.info('=' * 60)
    logger.info(f'總影像數量: {len(image_paths)}')
    logger.info(f'成功處理: {processed_count}')
    logger.info(f'處理失敗: {failed_count}')
    logger.info(f'成功率: {processed_count/len(image_paths)*100:.1f}%')
    logger.info(f'結果儲存位置: {args.output_dir}')
    
    print(f"\n 台北101影像去噪處理完成！")
    print(f" 處理統計: {processed_count}/{len(image_paths)} 成功")
    print(f" 結果位置: {args.output_dir}")
    
    if processed_count > 0:
        print(f"\n 下一步: 使用評估腳本比較去噪效果")
        print(f"   python main_evaluate_taipei101.py")

if __name__ == "__main__":
    main()
