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
            img_L = util.uint2tensor4(img_L)
            img_L = img_L.to(device)
            
            # 添加雜訊等級資訊
            noise_level_tensor = torch.full((1, 1, img_L.shape[2], img_L.shape[3]), 
                                          noise_level_normalized).to(device)
            img_input = torch.cat([img_L, noise_level_tensor], dim=1)
            
            # 去噪處理
            with torch.no_grad():
                img_E = model(img_input)
                img_E = util.tensor2uint(img_E)
            
            # 確保輸出影像尺寸正確
            if img_E.shape != original_shape:
                logger.warning(f'{img_name} - 輸出尺寸不匹配: {img_E.shape} vs {original_shape}')
            
            # 細節增強
            img_E = advanced_detail_enhancement(img_E, util.imread_uint(img_path, n_channels=3))
            
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
