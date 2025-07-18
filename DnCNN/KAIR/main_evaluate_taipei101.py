#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import argparse
import numpy as np
from collections import OrderedDict

from utils import utils_logger
from utils import utils_image as util

"""
台北101影像去噪評估腳本
專門用於比對原圖和去噪後結果的多項品質指標

支援的評估指標:
- PSNR (Peak Signal-to-Noise Ratio): 峰值信噪比
- SSIM (Structural Similarity Index): 結構相似性指標
- LPIPS (Learned Perceptual Image Patch Similarity): 學習感知相似性
- NIQE (No-Reference Image Quality Evaluator): 無參考影像品質評估

使用方法:
python main_evaluate_taipei101.py --original_dir testsets/taipei101 --denoised_dir taipei101_color_denoised

原圖路徑: testsets/taipei101
去噪結果: taipei101_color_denoised

注意: 
- 請先安裝依賴: pip install -r requirements.txt
- LPIPS 和 NIQE 需要額外套件，如無法安裝會顯示警告並跳過計算
"""

def main():
    # ----------------------------------------
    # 參數設定
    # ----------------------------------------
    parser = argparse.ArgumentParser(description='台北101影像去噪品質評估')
    parser.add_argument('--original_dir', type=str, default='testsets/taipei101', 
                       help='原始影像目錄路徑')
    parser.add_argument('--denoised_dir', type=str, default='taipei101_color_denoised', 
                       help='去噪後影像目錄路徑')
    parser.add_argument('--results_dir', type=str, default='evaluation_results', 
                       help='評估結果儲存目錄')
    parser.add_argument('--border', type=int, default=0, 
                       help='計算PSNR/SSIM時忽略的邊界像素數')
    args = parser.parse_args()

    # ----------------------------------------
    # 建立評估結果目錄和日誌
    # ----------------------------------------
    util.mkdir(args.results_dir)
    
    logger_name = 'taipei101_evaluation'
    utils_logger.logger_info(logger_name, log_path=os.path.join(args.results_dir, 'evaluation.log'))
    logger = logging.getLogger(logger_name)
    
    logger.info('=== 台北101影像去噪品質評估 ===')
    logger.info(f'原始影像目錄: {args.original_dir}')
    logger.info(f'去噪影像目錄: {args.denoised_dir}')
    logger.info(f'評估結果目錄: {args.results_dir}')
    
    # ----------------------------------------
    # 檢查目錄是否存在
    # ----------------------------------------
    if not os.path.exists(args.original_dir):
        logger.error(f'原始影像目錄不存在: {args.original_dir}')
        return
    
    if not os.path.exists(args.denoised_dir):
        logger.error(f'去噪影像目錄不存在: {args.denoised_dir}')
        return
    
    # ----------------------------------------
    # 取得影像檔案列表
    # ----------------------------------------
    original_paths = util.get_image_paths(args.original_dir)
    denoised_paths = util.get_image_paths(args.denoised_dir)
    
    logger.info(f'找到原始影像數量: {len(original_paths)}')
    logger.info(f'找到去噪影像數量: {len(denoised_paths)}')
    
    # 確保影像數量一致
    if len(original_paths) != len(denoised_paths):
        logger.warning(f'原始影像和去噪影像數量不一致！原始: {len(original_paths)}, 去噪: {len(denoised_paths)}')
    
    # 建立檔名映射
    original_dict = {os.path.basename(path): path for path in original_paths}
    denoised_dict = {os.path.basename(path): path for path in denoised_paths}
    
    # 找到共同的檔案
    common_files = set(original_dict.keys()) & set(denoised_dict.keys())
    logger.info(f'共同的影像檔案數量: {len(common_files)}')
    
    if len(common_files) == 0:
        logger.error('沒有找到對應的影像檔案！請檢查檔案名稱是否一致。')
        return
    
    # ----------------------------------------
    # 品質評估
    # ----------------------------------------
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['lpips'] = []
    test_results['niqe'] = []
    test_results['filenames'] = []
    
    logger.info('\n開始評估影像品質...')
    logger.info('-' * 80)
    
    for idx, filename in enumerate(sorted(common_files)):
        original_path = original_dict[filename]
        denoised_path = denoised_dict[filename]
        
        # 載入原始影像
        img_original = util.imread_uint(original_path, n_channels=3)
        
        # 載入去噪影像
        img_denoised = util.imread_uint(denoised_path, n_channels=3)
        
        # 確保影像尺寸一致
        if img_original.shape != img_denoised.shape:
            logger.warning(f'{filename} - 影像尺寸不一致! 原始: {img_original.shape}, 去噪: {img_denoised.shape}')
            # 調整去噪影像尺寸以匹配原始影像
            if img_denoised.shape[:2] != img_original.shape[:2]:
                continue  # 跳過尺寸差異過大的影像
        
        # 計算所有品質指標
        try:
            # 計算傳統指標
            psnr = util.calculate_psnr(img_denoised, img_original, border=args.border)
            ssim = util.calculate_ssim(img_denoised, img_original, border=args.border)
            
            # 計算新指標
            lpips_score = util.calculate_lpips(img_denoised, img_original)
            niqe_score = util.calculate_niqe(img_denoised)
            
            # 儲存結果
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['lpips'].append(lpips_score if lpips_score is not None else 0.0)
            test_results['niqe'].append(niqe_score if niqe_score is not None else 0.0)
            test_results['filenames'].append(filename)
            
            # 顯示結果
            lpips_str = f'LPIPS: {lpips_score:.4f}' if lpips_score is not None else 'LPIPS: N/A'
            niqe_str = f'NIQE: {niqe_score:.4f}' if niqe_score is not None else 'NIQE: N/A'
            
            logger.info(f'{idx+1:3d}/{len(common_files):3d} - {filename:<40}')
            logger.info(f'    PSNR: {psnr:6.2f} dB, SSIM: {ssim:.4f}, {lpips_str}, {niqe_str}')
            
        except Exception as e:
            logger.error(f'{filename} - 評估失敗: {str(e)}')
            continue
    
    # ----------------------------------------
    # 統計結果
    # ----------------------------------------
    if len(test_results['psnr']) > 0:
        # 計算基本統計
        avg_psnr = np.mean(test_results['psnr'])
        avg_ssim = np.mean(test_results['ssim'])
        std_psnr = np.std(test_results['psnr'])
        std_ssim = np.std(test_results['ssim'])
        
        # 計算新指標統計（排除 None 值）
        valid_lpips = [x for x in test_results['lpips'] if x is not None and x != 0.0]
        valid_niqe = [x for x in test_results['niqe'] if x is not None and x != 0.0]
        
        avg_lpips = np.mean(valid_lpips) if valid_lpips else None
        std_lpips = np.std(valid_lpips) if valid_lpips else None
        avg_niqe = np.mean(valid_niqe) if valid_niqe else None
        std_niqe = np.std(valid_niqe) if valid_niqe else None
        
        logger.info('\n' + '=' * 80)
        logger.info(' 評估結果統計')
        logger.info('=' * 80)
        logger.info(f'處理影像數量: {len(test_results["psnr"])}')
        logger.info(f'平均 PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB')
        logger.info(f'平均 SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}')
        
        if avg_lpips is not None:
            logger.info(f'平均 LPIPS: {avg_lpips:.4f} ± {std_lpips:.4f}')
            logger.info(f'最低 LPIPS: {min(valid_lpips):.4f} ({test_results["filenames"][test_results["lpips"].index(min(valid_lpips))]})')
            logger.info(f'最高 LPIPS: {max(valid_lpips):.4f} ({test_results["filenames"][test_results["lpips"].index(max(valid_lpips))]})')
        else:
            logger.info('LPIPS: 無法計算')
        
        if avg_niqe is not None:
            logger.info(f'平均 NIQE: {avg_niqe:.4f} ± {std_niqe:.4f}')
            logger.info(f'最低 NIQE: {min(valid_niqe):.4f} ({test_results["filenames"][test_results["niqe"].index(min(valid_niqe))]})')
            logger.info(f'最高 NIQE: {max(valid_niqe):.4f} ({test_results["filenames"][test_results["niqe"].index(max(valid_niqe))]})')
        else:
            logger.info('NIQE: 無法計算')
        
        logger.info(f'最高 PSNR: {max(test_results["psnr"]):.2f} dB ({test_results["filenames"][test_results["psnr"].index(max(test_results["psnr"]))]})')
        logger.info(f'最低 PSNR: {min(test_results["psnr"]):.2f} dB ({test_results["filenames"][test_results["psnr"].index(min(test_results["psnr"]))]})')
        logger.info(f'最高 SSIM: {max(test_results["ssim"]):.4f} ({test_results["filenames"][test_results["ssim"].index(max(test_results["ssim"]))]})')
        logger.info(f'最低 SSIM: {min(test_results["ssim"]):.4f} ({test_results["filenames"][test_results["ssim"].index(min(test_results["ssim"]))]})')
        
        # 儲存詳細結果到CSV檔案
        import csv
        csv_path = os.path.join(args.results_dir, 'detailed_results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['檔案名稱', 'PSNR (dB)', 'SSIM', 'LPIPS', 'NIQE'])
            for i, filename in enumerate(test_results['filenames']):
                lpips_val = f'{test_results["lpips"][i]:.4f}' if test_results["lpips"][i] is not None and test_results["lpips"][i] != 0.0 else 'N/A'
                niqe_val = f'{test_results["niqe"][i]:.4f}' if test_results["niqe"][i] is not None and test_results["niqe"][i] != 0.0 else 'N/A'
                writer.writerow([filename, f'{test_results["psnr"][i]:.2f}', f'{test_results["ssim"][i]:.4f}', lpips_val, niqe_val])
            
            # 寫入平均值
            avg_lpips_str = f'{avg_lpips:.4f}' if avg_lpips is not None else 'N/A'
            avg_niqe_str = f'{avg_niqe:.4f}' if avg_niqe is not None else 'N/A'
            writer.writerow(['平均值', f'{avg_psnr:.2f}', f'{avg_ssim:.4f}', avg_lpips_str, avg_niqe_str])
            
            # 寫入標準差
            std_lpips_str = f'{std_lpips:.4f}' if std_lpips is not None else 'N/A'
            std_niqe_str = f'{std_niqe:.4f}' if std_niqe is not None else 'N/A'
            writer.writerow(['標準差', f'{std_psnr:.2f}', f'{std_ssim:.4f}', std_lpips_str, std_niqe_str])
        
        logger.info(f'\n 詳細結果已儲存至: {csv_path}')
        logger.info(' 評估完成！')
        
        print(f'\n 台北101去噪效果評估結果:')
        print(f' 平均 PSNR: {avg_psnr:.2f} dB')
        print(f' 平均 SSIM: {avg_ssim:.4f}')
        if avg_lpips is not None:
            print(f' 平均 LPIPS: {avg_lpips:.4f}')
        else:
            print(f' 平均 LPIPS: 無法計算')
        if avg_niqe is not None:
            print(f' 平均 NIQE: {avg_niqe:.4f}')
        else:
            print(f' 平均 NIQE: 無法計算')
        print(f' 詳細結果: {csv_path}')
        
    else:
        logger.error('沒有成功評估任何影像！')

if __name__ == '__main__':
    main() 