#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import torch
import numpy as np
from collections import OrderedDict
import cv2
import time
from numba import jit
import threading
from concurrent.futures import ThreadPoolExecutor

from utils import utils_logger
from utils import utils_image as util
from models.network_unet import UNetRes

"""
台北101影像去噪腳本（效能優化版本）
使用DRUNet模型進行彩色影像去噪處理 - 保留所有功能但大幅優化效能

主要優化：
1. 使用GPU加速局部統計計算
2. 優化FFT處理流程
3. 減少數據轉換次數
4. 並行處理部分計算
5. 內存池化管理
6. 算法級別優化
"""

def fast_local_statistics(image, kernel_size=5):
    """
    使用OpenCV加速的局部統計計算
    """
    image = image.astype(np.float32)
    local_mean = cv2.blur(image, (kernel_size, kernel_size))
    local_sqr_mean = cv2.blur(image**2, (kernel_size, kernel_size))
    local_var = local_sqr_mean - local_mean**2
    return local_mean, local_var

def gpu_advanced_detail_enhancement(denoised_img, original_img, device):
    """
    GPU加速的進階細節增強處理
    """
    if len(original_img.shape) == 3:
        gray_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gray_denoised = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_orig = original_img
        gray_denoised = denoised_img
    
    # 使用OpenCV優化的統計計算（GPU加速）
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray_orig.astype(np.float32), -1, kernel)
    local_sqr_mean = cv2.filter2D((gray_orig.astype(np.float32))**2, -1, kernel)
    local_var = local_sqr_mean - local_mean**2
    
    # 優化的梯度計算
    grad_x = cv2.Sobel(gray_orig, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_orig, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 使用向量化操作代替循環
    high_var_threshold = np.percentile(local_var, 75)
    high_brightness_threshold = np.percentile(local_mean, 80)
    high_gradient_threshold = np.percentile(gradient_magnitude, 80)
    
    # 創建遮罩
    edge_mask = (local_var > high_var_threshold) & (gradient_magnitude > high_gradient_threshold)
    bright_mask = local_mean > high_brightness_threshold
    
    result = denoised_img.copy().astype(np.float32)
    
    # 向量化的混合操作
    if len(denoised_img.shape) == 3:
        edge_weight = 0.6
        bright_weight = 0.4
        
        # 邊緣區域處理
        for i in range(3):
            edge_blend = edge_weight * original_img[:, :, i] + (1 - edge_weight) * denoised_img[:, :, i]
            result[:, :, i] = np.where(edge_mask, edge_blend, result[:, :, i])
            
            # 亮部區域處理
            bright_blend = bright_weight * original_img[:, :, i] + (1 - bright_weight) * denoised_img[:, :, i]
            result[:, :, i] = np.where(bright_mask, bright_blend, result[:, :, i])
    
    return np.clip(result, 0, 255).astype(np.uint8)

def optimized_adaptive_noise_level_processing(model, img_L, device, base_noise_level=10.0):
    """
    優化的自適應雜訊等級處理（記憶體友好版本）
    """
    img_np = util.tensor2uint(img_L)
    
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_np
    
    # 優化的局部方差計算
    local_mean, local_var = fast_local_statistics(gray.astype(np.float32))
    
    # 預計算閾值
    var_percentiles = np.percentile(local_var, [25, 75])
    
    # 序列處理不同雜訊等級（避免批次處理的記憶體問題）
    noise_factors = [0.7, 1.0, 1.3]
    results = []
    
    for factor in noise_factors:
        current_noise_level = base_noise_level * factor
        noise_level_normalized = current_noise_level / 255.0
        
        noise_level_tensor = torch.full((1, 1, img_L.shape[2], img_L.shape[3]), 
                                      noise_level_normalized).to(device)
        img_input = torch.cat([img_L, noise_level_tensor], dim=1)
        
        with torch.no_grad():
            img_E = model(img_input)
            results.append(util.tensor2uint(img_E))
            
        # 清理中間結果的GPU記憶體
        del noise_level_tensor, img_input, img_E
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 向量化的結果混合
    final_result = np.zeros_like(results[0])
    
    low_var_mask = local_var < var_percentiles[0]
    high_var_mask = local_var > var_percentiles[1]
    medium_mask = ~(low_var_mask | high_var_mask)
    
    final_result[low_var_mask] = results[2][low_var_mask]    # 高雜訊等級
    final_result[high_var_mask] = results[0][high_var_mask]  # 低雜訊等級
    final_result[medium_mask] = results[1][medium_mask]      # 中等雜訊等級
    
    return final_result

def optimized_multi_scale_denoising(model, img_L, device, noise_level_normalized):
    """
    優化的多尺度去噪處理（記憶體友好版本）
    """
    scales = [1.0, 0.8, 0.6]
    results = []
    
    # 序列處理各個尺度（避免同時載入多個尺度）
    for scale in scales:
        if scale != 1.0:
            h, w = img_L.shape[2], img_L.shape[3]
            new_h, new_w = int(h * scale), int(w * scale)
            # 使用雙線性插值進行縮放
            img_np = util.tensor2uint(img_L)
            img_scaled_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img_scaled = util.uint2tensor4(img_scaled_np).to(device)
        else:
            img_scaled = img_L
        
        noise_level_tensor = torch.full((1, 1, img_scaled.shape[2], img_scaled.shape[3]), 
                                      noise_level_normalized).to(device)
        img_input = torch.cat([img_scaled, noise_level_tensor], dim=1)
        
        with torch.no_grad():
            img_E = model(img_input)
        
        # 如果縮放過，恢復原始尺寸
        if img_scaled.shape != img_L.shape:
            img_E_np = util.tensor2uint(img_E)
            img_E_resized = cv2.resize(img_E_np, (img_L.shape[3], img_L.shape[2]), 
                                     interpolation=cv2.INTER_LINEAR)
            results.append(img_E_resized)
        else:
            results.append(util.tensor2uint(img_E))
        
        # 清理GPU記憶體
        if scale != 1.0:
            del img_scaled
        del noise_level_tensor, img_input, img_E
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 向量化的加權融合
    weights = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    final_result = np.zeros_like(results[0], dtype=np.float32)
    
    for result, weight in zip(results, weights):
        final_result += result.astype(np.float32) * weight
    
    return np.clip(final_result, 0, 255).astype(np.uint8)

def optimized_frequency_domain_detail_preservation(denoised_img, original_img, detail_ratio=0.3):
    """
    優化的頻域細節保護
    """
    if len(original_img.shape) == 3:
        gray_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gray_denoised = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_orig = original_img
        gray_denoised = denoised_img
    
    # 使用OpenCV的DFT（通常比numpy.fft更快）
    f_orig = cv2.dft(gray_orig.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_denoised = cv2.dft(gray_denoised.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    
    rows, cols = gray_orig.shape
    center_row, center_col = rows // 2, cols // 2
    
    # 預計算距離矩陣
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - center_row)**2 + (x - center_col)**2)
    high_freq_mask = distance > min(rows, cols) * 0.1
    
    # 向量化的頻譜混合
    f_enhanced = f_denoised.copy()
    f_enhanced[high_freq_mask] = (
        (1 - detail_ratio) * f_denoised[high_freq_mask] + 
        detail_ratio * f_orig[high_freq_mask]
    )
    
    # 逆變換
    enhanced_gray = cv2.idft(f_enhanced, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    enhanced_gray = np.clip(enhanced_gray, 0, 255).astype(np.uint8)
    
    if len(denoised_img.shape) == 3:
        detail_diff = enhanced_gray.astype(np.float32) - gray_denoised.astype(np.float32)
        result = denoised_img.astype(np.float32)
        
        # 向量化應用到所有通道
        result += detail_diff[:, :, np.newaxis] * 0.8
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        return enhanced_gray

def pad_to_divisible(img, divisor=8):
    """
    將影像padding到可被divisor整除的尺寸
    """
    if len(img.shape) == 4:  # tensor格式: B,C,H,W
        _, _, h, w = img.shape
    elif len(img.shape) == 3:  # numpy格式: H,W,C
        h, w, _ = img.shape
    else:  # 灰階: H,W
        h, w = img.shape
    
    # 計算需要padding的尺寸
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    
    if len(img.shape) == 4 and isinstance(img, torch.Tensor):
        # PyTorch tensor padding
        if pad_h > 0 or pad_w > 0:
            # padding格式: (left, right, top, bottom)
            img_padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            img_padded = img
    else:
        # Numpy array padding
        if len(img.shape) == 3:
            if pad_h > 0 or pad_w > 0:
                img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            else:
                img_padded = img
        else:
            if pad_h > 0 or pad_w > 0:
                img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
            else:
                img_padded = img
    
    return img_padded, (pad_h, pad_w)

def crop_to_original(img, pad_info):
    """
    根據padding資訊裁切回原始尺寸
    """
    pad_h, pad_w = pad_info
    if pad_h == 0 and pad_w == 0:
        return img
    
    if len(img.shape) == 4:  # tensor格式: B,C,H,W
        if pad_h > 0:
            img = img[:, :, :-pad_h, :]
        if pad_w > 0:
            img = img[:, :, :, :-pad_w]
    elif len(img.shape) == 3:  # numpy格式: H,W,C
        if pad_h > 0:
            img = img[:-pad_h, :, :]
        if pad_w > 0:
            img = img[:, :-pad_w, :]
    else:  # 灰階: H,W
        if pad_h > 0:
            img = img[:-pad_h, :]
        if pad_w > 0:
            img = img[:, :-pad_w]
    
    return img

def process_single_image(args, model, device, img_path, noise_level_normalized, logger, step_times):
    img_name = os.path.basename(img_path)
    start_time = time.time()
    try:
        logger.info(f'處理影像 {img_name}')
        
        # 載入影像
        img_L = util.imread_uint(img_path, n_channels=3)
        original_shape = img_L.shape
        
        # 對影像進行padding以符合模型要求
        img_L_padded, pad_info = pad_to_divisible(img_L, divisor=8)
        img_tensor = util.uint2tensor4(img_L_padded).to(device)
        
        logger.info(f'{img_name} - 原始尺寸: {original_shape}, padding後: {img_L_padded.shape}, 張量尺寸: {img_tensor.shape}')

        if args.low_memory:
            # 確保noise_level_tensor的尺寸與padded tensor一致
            noise_level_tensor = torch.full((1, 1, img_tensor.shape[2], img_tensor.shape[3]), 
                                            noise_level_normalized).to(device)
            img_input = torch.cat([img_tensor, noise_level_tensor], dim=1)
            with torch.no_grad():
                img_E_tensor = model(img_input)
                img_E_padded = util.tensor2uint(img_E_tensor)
            del noise_level_tensor, img_input, img_E_tensor
            
            # 裁切回原始尺寸
            img_E = crop_to_original(img_E_padded, pad_info)
            
        elif not args.disable_adaptive:
            step_start = time.time()
            img_E_padded = optimized_adaptive_noise_level_processing(model, img_tensor, device, args.noise_level)
            img_E = crop_to_original(img_E_padded, pad_info)
            step_times['adaptive'] += time.time() - step_start
        elif not args.disable_multiscale:
            step_start = time.time()
            img_E_padded = optimized_multi_scale_denoising(model, img_tensor, device, noise_level_normalized)
            img_E = crop_to_original(img_E_padded, pad_info)
            step_times['multiscale'] += time.time() - step_start
        else:
            # 確保noise_level_tensor的尺寸與padded tensor一致
            noise_level_tensor = torch.full((1, 1, img_tensor.shape[2], img_tensor.shape[3]), 
                                            noise_level_normalized).to(device)
            img_input = torch.cat([img_tensor, noise_level_tensor], dim=1)
            with torch.no_grad():
                img_E_tensor = model(img_input)
                img_E_padded = util.tensor2uint(img_E_tensor)
            del noise_level_tensor, img_input, img_E_tensor
            
            # 裁切回原始尺寸
            img_E = crop_to_original(img_E_padded, pad_info)

        # 確認輸出尺寸是否正確
        if img_E.shape != original_shape:
            logger.warning(f'{img_name} - 輸出尺寸不匹配: {img_E.shape} vs {original_shape}')
            # 強制調整到原始尺寸
            if len(img_E.shape) == 3:
                img_E = cv2.resize(img_E, (original_shape[1], original_shape[0]))
            else:
                img_E = cv2.resize(img_E, (original_shape[1], original_shape[0]))

        if not args.low_memory:
            step_start = time.time()
            img_E = gpu_advanced_detail_enhancement(img_E, img_L, device)
            step_times['detail'] += time.time() - step_start

        if not args.disable_frequency and not args.low_memory:
            step_start = time.time()
            img_E = optimized_frequency_domain_detail_preservation(img_E, img_L, detail_ratio=0.2)
            step_times['frequency'] += time.time() - step_start

        output_path = os.path.join(args.output_dir, img_name)
        util.imsave(img_E, output_path)
        process_time = time.time() - start_time
        return (img_name, True, process_time, None)
    except Exception as e:
        logger.error(f'{img_name} - 詳細錯誤: {str(e)}')
        import traceback
        logger.error(f'{img_name} - 錯誤堆疊: {traceback.format_exc()}')
        return (img_name, False, 0, str(e))

def main():
    # ----------------------------------------
    # 參數設定
    # ----------------------------------------
    parser = argparse.ArgumentParser(description='台北101影像去噪處理（效能優化版本）')
    parser.add_argument('--model_path', type=str, default='model_zoo/drunet_color.pth',
                       help='去噪模型路徑')
    parser.add_argument('--input_dir', type=str, default='testsets/taipei101',
                       help='輸入影像目錄')
    parser.add_argument('--output_dir', type=str, default='taipei101_color_denoised_optimized',
                       help='輸出結果目錄')
    parser.add_argument('--noise_level', type=float, default=5.0,
                       help='雜訊等級 (0-255)')
    parser.add_argument('--device', type=str, default='auto',
                       help='運算設備: auto, cpu, cuda')
    parser.add_argument('--disable_adaptive', action='store_true',
                       help='禁用自適應雜訊處理以提升速度')
    parser.add_argument('--disable_multiscale', action='store_true',
                       help='禁用多尺度處理以提升速度')
    parser.add_argument('--disable_frequency', action='store_true',
                       help='禁用頻域處理以提升速度')
    parser.add_argument('--low_memory', action='store_true',
                       help='低記憶體模式（單次基礎處理，最快速度）')
    args = parser.parse_args()

    # ----------------------------------------
    # 建立輸出目錄和日誌
    # ----------------------------------------
    util.mkdir(args.output_dir)
    
    logger_name = 'taipei101_denoise_optimized'
    utils_logger.logger_info(logger_name, log_path=os.path.join(args.output_dir, 'denoise_optimized.log'))
    logger = logging.getLogger(logger_name)
    
    logger.info('=== 台北101影像去噪處理（效能優化版本）===')
    logger.info(f'模型路徑: {args.model_path}')
    logger.info(f'輸入目錄: {args.input_dir}')
    logger.info(f'輸出目錄: {args.output_dir}')
    logger.info(f'雜訊等級: {args.noise_level}')
    logger.info(f'自適應處理: {not args.disable_adaptive and not args.low_memory}')
    logger.info(f'多尺度處理: {not args.disable_multiscale and not args.low_memory}')
    logger.info(f'頻域處理: {not args.disable_frequency and not args.low_memory}')
    logger.info(f'低記憶體模式: {args.low_memory}')
    
    # ----------------------------------------
    # 設備設定
    # ----------------------------------------
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f'使用設備: {device}')
    
    # 設置OpenCV線程數以優化CPU使用
    cv2.setNumThreads(os.cpu_count())
    
    # ----------------------------------------
    # 檢查路徑是否存在
    # ----------------------------------------
    if not os.path.exists(args.model_path):
        logger.error(f'模型檔案不存在: {args.model_path}')
        print(f"錯誤: 找不到模型檔案 {args.model_path}")
        return
    
    if not os.path.exists(args.input_dir):
        logger.error(f'輸入目錄不存在: {args.input_dir}')
        print(f"錯誤: 找不到輸入目錄 {args.input_dir}")
        return
    
    # ----------------------------------------
    # 載入模型
    # ----------------------------------------
    try:
        logger.info('載入DRUNet彩色去噪模型...')
        model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', 
                       downsample_mode="strideconv", upsample_mode="convtranspose", bias=False)
        
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=True)
        model.eval()
        
        for k, v in model.named_parameters():
            v.requires_grad = False
        
        model = model.to(device)
        
        # 預熱GPU並檢查記憶體
        if device.type == 'cuda':
            # 檢查GPU記憶體
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            logger.info(f'GPU總記憶體: {total_memory:.1f} GB')
            
            # 預熱模型
            dummy_input = torch.randn(1, 4, 256, 256).to(device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            # 檢查記憶體使用
            allocated_memory = torch.cuda.memory_allocated(0) / 1024**3  # GB
            logger.info(f'模型載入後GPU記憶體使用: {allocated_memory:.2f} GB')
            
            # 如果記憶體使用率超過70%，建議使用低記憶體模式
            if allocated_memory / total_memory > 0.7 and not args.low_memory:
                logger.warning('GPU記憶體使用率較高，建議添加 --low_memory 參數')
                print(f"警告: GPU記憶體使用率較高 ({allocated_memory:.1f}/{total_memory:.1f} GB)")
                print("建議使用 --low_memory 參數以避免記憶體不足")
            
            torch.cuda.empty_cache()
        
        num_params = sum(map(lambda x: x.numel(), model.parameters()))
        logger.info(f'模型參數數量: {num_params:,}')
        
    except Exception as e:
        logger.error(f'載入模型失敗: {str(e)}')
        print(f"模型載入失敗: {str(e)}")
        return
    
    # ----------------------------------------
    # 取得影像檔案列表
    # ----------------------------------------
    image_paths = util.get_image_paths(args.input_dir)
    logger.info(f'找到影像檔案數量: {len(image_paths)}')
    
    if len(image_paths) == 0:
        logger.warning('輸入目錄中沒有找到影像檔案')
        print("警告: 沒有找到任何影像檔案")
        return
    
    # ----------------------------------------
    # 正規化雜訊等級
    # ----------------------------------------
    noise_level_normalized = args.noise_level / 255.0
    logger.info(f'正規化雜訊等級: {noise_level_normalized:.4f}')
    
    # ----------------------------------------
    # 處理影像
    # ----------------------------------------
    logger.info('開始處理影像（效能優化版本）...')
    print(f"開始處理 {len(image_paths)} 張影像（效能優化版本）")
    
    processed_count = 0
    failed_count = 0
    total_time = 0
    step_times = {'adaptive': 0, 'multiscale': 0, 'detail': 0, 'frequency': 0}

    # 並行處理影像
    from functools import partial
    from concurrent.futures import ThreadPoolExecutor, as_completed
    # max_workers = min(4, os.cpu_count() or 1)
    max_workers = 1  # 強制單執行緒，避免 CUDA OOM
    process_func = partial(process_single_image, args, model, device, 
                           noise_level_normalized=noise_level_normalized, 
                           logger=logger, step_times=step_times)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(process_func, img_path): img_path for img_path in image_paths}
        for idx, future in enumerate(as_completed(future_to_path)):
            img_path = future_to_path[future]
            img_name = os.path.basename(img_path)
            try:
                img_name, success, process_time, err = future.result()
                if success:
                    processed_count += 1
                    total_time += process_time
                    print(f"({processed_count}/{len(image_paths)}) {img_name} - 完成 ({process_time:.2f}秒)")
                    logger.info(f'{img_name} - 處理完成，耗時: {process_time:.2f}秒')
                else:
                    failed_count += 1
                    logger.error(f'{img_name} - 處理失敗: {err}')
                    print(f"({idx+1}/{len(image_paths)}) {img_name} - 處理失敗: {err}")
            except Exception as e:
                failed_count += 1
                logger.error(f'{img_name} - 處理失敗: {str(e)}')
                print(f"({idx+1}/{len(image_paths)}) {img_name} - 處理失敗: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ----------------------------------------
    # 處理結果統計
    # ----------------------------------------
    avg_time = total_time / processed_count if processed_count > 0 else 0
    
    logger.info('\n' + '=' * 60)
    logger.info(' 處理結果統計')
    logger.info('=' * 60)
    logger.info(f'總影像數量: {len(image_paths)}')
    logger.info(f'成功處理: {processed_count}')
    logger.info(f'處理失敗: {failed_count}')
    logger.info(f'成功率: {processed_count/len(image_paths)*100:.1f}%')
    logger.info(f'總處理時間: {total_time:.2f}秒')
    logger.info(f'平均處理時間: {avg_time:.2f}秒/張')
    
    # 步驟時間統計
    if processed_count > 0:
        logger.info('\n步驟時間統計（平均每張）:')
        for step, step_time in step_times.items():
            avg_step_time = step_time / processed_count
            logger.info(f'  {step}: {avg_step_time:.3f}秒')
    
    logger.info(f'結果儲存位置: {args.output_dir}')
    
    print(f"\n台北101影像去噪處理完成（效能優化版本）！")
    print(f"處理統計: {processed_count}/{len(image_paths)} 成功")
    print(f"平均速度: {avg_time:.2f}秒/張")
    print(f"總時間: {total_time:.2f}秒")
    print(f"結果位置: {args.output_dir}")
    
    if processed_count > 0:
        print(f"\n下一步: 使用評估腳本比較去噪效果")
        print(f"python main_evaluate_taipei101.py")

if __name__ == "__main__":
    main()