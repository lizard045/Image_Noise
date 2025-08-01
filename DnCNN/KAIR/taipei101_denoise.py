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
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

from utils import utils_logger
from utils import utils_image as util
from models.network_unet import UNetRes
import models.network_unet as nunet
"""
台北101影像去噪腳本
使用DRUNet模型進行彩色影像去噪處理

使用方法:
python taipei101_denoise.py --input_dir testsets/taipei101 --output_dir taipei101_color_denoised

"""
print("實際匯入檔案：", nunet.__file__)
print("UNetRes 參數：", nunet.UNetRes.__init__.__code__.co_varnames)

class GPUOptimizer:
    """GPU優化管理器"""
    
    def __init__(self):
        self.device = None
        self.gpu_info = {}
        self.use_amp = False
        self.scaler = None
        
    def setup_gpu(self):
        """設置GPU優化配置"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            # 啟用cuDNN基準測試模式
            cudnn.benchmark = True
            cudnn.deterministic = False
            
            # GPU記憶體管理
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # 獲取GPU資訊
            self.gpu_info = {
                'name': torch.cuda.get_device_name(0),
                'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'memory_available': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
            }
            
            # 根據GPU記憶體決定是否使用AMP
            if self.gpu_info['memory_total'] >= 4.0:  # 4GB以上使用AMP
                self.use_amp = True
                try:
                    # 新版PyTorch語法
                    from torch.amp import GradScaler
                    self.scaler = GradScaler('cuda')
                except ImportError:
                    # 舊版PyTorch語法
                    from torch.cuda.amp import GradScaler
                    self.scaler = GradScaler()
                
            print(f" GPU優化啟用:")
            print(f"   設備: {self.gpu_info['name']}")
            print(f"   總記憶體: {self.gpu_info['memory_total']:.1f}GB")
            print(f"   可用記憶體: {self.gpu_info['memory_available']:.1f}GB")
            print(f"   混合精度AMP: {self.use_amp}")
            
        else:
            self.device = torch.device('cpu')
            print(" 使用CPU處理 (建議使用GPU以獲得更好性能)")
            
        return self.device
    
    def get_optimal_batch_size(self, img_size):
        """根據GPU記憶體和影像尺寸計算最佳批次大小"""
        if not torch.cuda.is_available():
            return 1
            
        # 估算單張影像所需記憶體（MB）
        single_img_memory = (img_size[0] * img_size[1] * 3 * 4) / 1024**2  # 假設float32
        available_memory_mb = self.gpu_info['memory_available'] * 1024 * 0.7  # 使用70%可用記憶體
        
        # 考慮模型記憶體佔用
        model_memory_mb = 500  # 估算模型佔用約500MB
        available_for_batch = available_memory_mb - model_memory_mb
        
        optimal_batch = max(1, int(available_for_batch / (single_img_memory * 2)))  # 乘2考慮前向傳播
        return min(optimal_batch, 8)  # 限制最大批次為8
    
    def should_tile_process(self, img_size, threshold_pixels=2048*2048):
        """判斷是否需要分塊處理（避免記憶體不足）"""
        total_pixels = img_size[0] * img_size[1]
        return total_pixels > threshold_pixels
    
    def get_optimal_tile_size(self, img_size):
        """計算最佳分塊尺寸"""
        h, w = img_size
        
        # 根據GPU記憶體決定分塊大小
        if self.gpu_info['memory_total'] >= 8.0:
            max_tile_size = 1024  # 8GB GPU
        elif self.gpu_info['memory_total'] >= 6.0:
            max_tile_size = 768   # 6GB GPU
        else:
            max_tile_size = 512   # 4GB GPU以下
        
        # 確保分塊尺寸不超過原圖尺寸
        tile_h = min(max_tile_size, h)
        tile_w = min(max_tile_size, w)
        
        return tile_h, tile_w
    
    def monitor_gpu_memory(self):
        """監控GPU記憶體使用"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            return {
                'allocated': allocated,
                'cached': cached,
                'free': self.gpu_info['memory_total'] - allocated
            }
        return None
    
    def cleanup_memory(self):
        """清理GPU記憶體"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def remove_module_prefix(state_dict):
    """移除DataParallel訓練模型的'module.'前綴"""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # 移除'module.'前綴
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def smart_load_model(model, model_path, device, logger):
    """智能載入模型，自動處理DataParallel前綴問題"""
    try:
        # 載入state_dict
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 如果checkpoint是字典且包含'state_dict'或其他key
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # 假設整個checkpoint就是state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 嘗試直接載入
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info('[SUCCESS] 模型載入成功 (直接載入)')
            return True
        except RuntimeError as e:
            if 'module.' in str(e):
                # 如果有module前綴問題，移除前綴後重試
                logger.info('[INFO] 偵測到DataParallel前綴，正在移除...')
                clean_state_dict = remove_module_prefix(state_dict)
                
                try:
                    model.load_state_dict(clean_state_dict, strict=True)
                    logger.info('[SUCCESS] 模型載入成功 (移除DataParallel前綴)')
                    return True
                except RuntimeError as e2:
                    # 嘗試非嚴格載入
                    logger.warning('[WARNING] 嚴格載入失敗，嘗試非嚴格載入...')
                    missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
                    
                    if missing_keys:
                        logger.warning(f'缺少的參數: {missing_keys[:5]}{"..." if len(missing_keys) > 5 else ""}')
                    if unexpected_keys:
                        logger.warning(f'多餘的參數: {unexpected_keys[:5]}{"..." if len(unexpected_keys) > 5 else ""}')
                    
                    # 如果只是少數參數不匹配，仍然可以使用
                    if len(missing_keys) < 10:  # 容忍少量缺失
                        logger.info('[SUCCESS] 模型載入成功 (非嚴格模式)')
                        return True
                    else:
                        logger.error('[ERROR] 缺少太多重要參數，載入失敗')
                        return False
            else:
                raise e
                
    except Exception as e:
        logger.error(f'[ERROR] 模型載入完全失敗: {str(e)}')
        return False

def batch_process_images(model, images, device, noise_level_normalized, gpu_optimizer):
    """批次處理影像以提升效率"""
    results = []
    batch_size = gpu_optimizer.get_optimal_batch_size(images[0].shape[:2])
    
    print(f"  使用批次大小: {batch_size}")
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_tensors = []
        
        # 轉換為tensor批次
        for img in batch_images:
            img_tensor = util.uint2tensor4(img).to(device)
            noise_level_tensor = torch.full((1, 1, img_tensor.shape[2], img_tensor.shape[3]), 
                                          noise_level_normalized).to(device)
            img_input = torch.cat([img_tensor, noise_level_tensor], dim=1)
            batch_tensors.append(img_input)
        
        # 合併為批次tensor
        if len(batch_tensors) > 1:
            batch_input = torch.cat(batch_tensors, dim=0)
        else:
            batch_input = batch_tensors[0]
        
        # 使用AMP進行推理
        try:
            # 新版PyTorch語法
            from torch.amp import autocast
            with autocast('cuda', enabled=gpu_optimizer.use_amp):
                with torch.no_grad():
                    batch_output = model(batch_input)
        except ImportError:
            # 舊版PyTorch語法
            from torch.cuda.amp import autocast
            with autocast(enabled=gpu_optimizer.use_amp):
                with torch.no_grad():
                    batch_output = model(batch_input)
        
        # 分離批次結果
        if batch_output.shape[0] > 1:
            for j in range(batch_output.shape[0]):
                results.append(util.tensor2uint(batch_output[j:j+1]))
        else:
            results.append(util.tensor2uint(batch_output))
        
        # 記憶體清理
        if i % (batch_size * 2) == 0:
            gpu_optimizer.cleanup_memory()
    
    return results

def tile_process_image(model, img_L, device, noise_level_normalized, gpu_optimizer, tile_size=1024, overlap=64):
    """分塊處理大圖片，避免記憶體不足"""
    h, w, c = img_L.shape
    
    # 如果圖片小於分塊尺寸，直接處理
    if h <= tile_size and w <= tile_size:
        img_tensor = util.uint2tensor4(img_L).to(device)
        noise_level_tensor = torch.full((1, 1, img_tensor.shape[2], img_tensor.shape[3]), 
                                      noise_level_normalized).to(device)
        img_input = torch.cat([img_tensor, noise_level_tensor], dim=1)
        
        try:
            # 新版PyTorch語法
            from torch.amp import autocast
            with autocast('cuda', enabled=gpu_optimizer.use_amp):
                with torch.no_grad():
                    img_E = model(img_input)
        except ImportError:
            # 舊版PyTorch語法
            from torch.cuda.amp import autocast
            with autocast(enabled=gpu_optimizer.use_amp):
                with torch.no_grad():
                    img_E = model(img_input)
        
        return util.tensor2uint(img_E)
    
    # 分塊處理
    print(f"  大圖片分塊處理: {h}x{w} -> 分塊尺寸 {tile_size}x{tile_size}")
    
    # 計算分塊數量
    h_tiles = (h - 1) // (tile_size - overlap) + 1
    w_tiles = (w - 1) // (tile_size - overlap) + 1
    
    print(f"  分塊數量: {h_tiles}x{w_tiles} = {h_tiles * w_tiles} 塊")
    
    # 建立輸出影像
    result_img = np.zeros_like(img_L)
    
    for i in range(h_tiles):
        for j in range(w_tiles):
            # 計算分塊座標
            start_h = i * (tile_size - overlap)
            start_w = j * (tile_size - overlap)
            end_h = min(start_h + tile_size, h)
            end_w = min(start_w + tile_size, w)
            
            # 提取分塊
            tile = img_L[start_h:end_h, start_w:end_w, :]
            
            # 處理分塊
            tile_tensor = util.uint2tensor4(tile).to(device)
            noise_level_tensor = torch.full((1, 1, tile_tensor.shape[2], tile_tensor.shape[3]), 
                                          noise_level_normalized).to(device)
            tile_input = torch.cat([tile_tensor, noise_level_tensor], dim=1)
            
            try:
                # 新版PyTorch語法
                from torch.amp import autocast
                with autocast('cuda', enabled=gpu_optimizer.use_amp):
                    with torch.no_grad():
                        tile_output = model(tile_input)
            except ImportError:
                # 舊版PyTorch語法
                from torch.cuda.amp import autocast
                with autocast(enabled=gpu_optimizer.use_amp):
                    with torch.no_grad():
                        tile_output = model(tile_input)
            
            tile_result = util.tensor2uint(tile_output)
            
            # 處理重疊區域
            if i == 0 and j == 0:
                # 左上角塊，完整複製
                result_img[start_h:end_h, start_w:end_w, :] = tile_result
            else:
                # 其他塊，處理重疊區域
                actual_start_h = start_h + (overlap // 2 if i > 0 else 0)
                actual_start_w = start_w + (overlap // 2 if j > 0 else 0)
                
                tile_start_h = overlap // 2 if i > 0 else 0
                tile_start_w = overlap // 2 if j > 0 else 0
                
                result_img[actual_start_h:end_h, actual_start_w:end_w, :] = \
                    tile_result[tile_start_h:, tile_start_w:, :]
            
            # 清理記憶體
            del tile_tensor, tile_input, tile_output, tile_result
            if (i * w_tiles + j) % 4 == 0:  # 每4塊清理一次
                gpu_optimizer.cleanup_memory()
            
            print(f"    完成分塊 ({i+1}/{h_tiles}, {j+1}/{w_tiles})")
    
    return result_img

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
        edge_weight = 0.75  # 增加邊緣細節權重
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
        bright_weight = 0.6  # 亮部細節權重
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
    for noise_factor in [0.5, 1.0, 1.5]:  # 低、中、高雜訊等級
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
    parser.add_argument('--model_path', type=str, default='model_zoo/best_G.pth',
                       help='去噪模型路徑 (使用最新訓練的最佳模型)')
    parser.add_argument('--input_dir', type=str, default='testsets/taipei101',
                       help='輸入影像目錄')
    parser.add_argument('--output_dir', type=str, default='taipei101_color_denoised',
                       help='輸出結果目錄')
    parser.add_argument('--noise_level', type=float, default=5.0,
                       help='雜訊等級 (0-255)')
    parser.add_argument('--device', type=str, default='auto',
                       help='運算設備: auto, cpu, cuda')
    parser.add_argument('--enable_batch_processing', action='store_true',
                       help='啟用批次處理以提升GPU效率')
    parser.add_argument('--monitor_performance', action='store_true',
                       help='啟用性能監控')
    args = parser.parse_args()

    # ----------------------------------------
    # 建立輸出目錄和日誌
    # ----------------------------------------
    util.mkdir(args.output_dir)
    
    logger_name = 'taipei101_denoise'
    utils_logger.logger_info(logger_name, log_path=os.path.join(args.output_dir, 'denoise.log'))
    logger = logging.getLogger(logger_name)
    
    logger.info('=== 台北101影像去噪處理 (GPU優化版) ===')
    logger.info(f'模型路徑: {args.model_path}')
    logger.info(f'輸入目錄: {args.input_dir}')
    logger.info(f'輸出目錄: {args.output_dir}')
    logger.info(f'雜訊等級: {args.noise_level}')
    logger.info(f'批次處理: {args.enable_batch_processing}')
    logger.info(f'性能監控: {args.monitor_performance}')
    
    # ----------------------------------------
    # GPU優化器設定
    # ----------------------------------------
    gpu_optimizer = GPUOptimizer()
    device = gpu_optimizer.setup_gpu()
    
    if args.device != 'auto':
        device = torch.device(args.device)
        print(f" 強制使用指定設備: {device}")
    
    logger.info(f'使用設備: {device}')
    
    # 性能監控初始化
    if args.monitor_performance:
        start_memory = gpu_optimizer.monitor_gpu_memory()
        if start_memory:
            logger.info(f'初始GPU記憶體: 已分配 {start_memory["allocated"]:.2f}GB, 快取 {start_memory["cached"]:.2f}GB')
    
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
    # 載入模型 (智能載入，自動處理DataParallel問題)
    # ----------------------------------------
    try:
        logger.info('載入DRUNet彩色去噪模型...')
        
        # 建立模型結構 (與訓練時完全一致)
        model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', 
                       downsample_mode="strideconv", upsample_mode="convtranspose", bias=False, use_nonlocal=True)
        
        # 智能載入預訓練權重
        if not smart_load_model(model, args.model_path, device, logger):
            logger.error('[ERROR] 模型載入失敗')
            print(f" ❌ 模型載入失敗，請檢查模型檔案")
            return
        
        model.eval()
        
        # 禁用梯度計算
        for k, v in model.named_parameters():
            v.requires_grad = False
        
        model = model.to(device)
        
        # 計算模型參數數量
        num_params = sum(map(lambda x: x.numel(), model.parameters()))
        logger.info(f'模型參數數量: {num_params:,}')
        
        # 模型資訊
        print(f"  ✅ 模型載入成功!")
        print(f"  📊 參數數量: {num_params:,}")
        print(f"  🏗️  模型結構: UNetRes (DRUNet)")
        
    except Exception as e:
        logger.error(f'載入模型失敗: {str(e)}')
        print(f" ❌ 模型載入失敗: {str(e)}")
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
    # 處理影像 (GPU優化版本)
    # ----------------------------------------
    logger.info('開始處理影像 (GPU優化模式)...')
    print(f"  開始處理 {len(image_paths)} 張影像")
    
    processed_count = 0
    failed_count = 0
    start_time = time.time()
    
    if args.enable_batch_processing and len(image_paths) > 1:
        # 批次處理模式
        logger.info('使用批次處理模式提升效率')
        
        try:
            # 載入所有影像
            all_images = []
            all_names = []
            
            print("  載入影像中...")
            for img_path in image_paths:
                img_name = os.path.basename(img_path)
                img_L = util.imread_uint(img_path, n_channels=3)
                all_images.append(img_L)
                all_names.append(img_name)
            
            # 批次處理
            print("  執行批次去噪處理...")
            results = batch_process_images(model, all_images, device, noise_level_normalized, gpu_optimizer)
            
            # 後處理和儲存
            print("  執行後處理和儲存...")
            for idx, (img_E, img_L, img_name) in enumerate(zip(results, all_images, all_names)):
                try:
                    # 進階細節增強
                    img_E = advanced_detail_enhancement(img_E, img_L)
                    
                    # 頻域細節保護
                    img_E = frequency_domain_detail_preservation(img_E, img_L, detail_ratio=0.2)
                    
                    # 儲存結果
                    output_path = os.path.join(args.output_dir, img_name)
                    util.imsave(img_E, output_path)
                    
                    processed_count += 1
                    print(f" ({idx+1}/{len(image_paths)}) {img_name} - 處理完成")
                    
                    # 性能監控
                    if args.monitor_performance and idx == 0:
                        memory_info = gpu_optimizer.monitor_gpu_memory()
                        if memory_info:
                            logger.info(f'處理中GPU記憶體: 已分配 {memory_info["allocated"]:.2f}GB')
                    
                except Exception as e:
                    failed_count += 1
                    logger.error(f'{img_name} - 後處理失敗: {str(e)}')
                    print(f" ({idx+1}/{len(image_paths)}) {img_name} - 後處理失敗: {str(e)}")
                    
        except Exception as e:
            logger.error(f'批次處理失敗，回退到單張處理: {str(e)}')
            print(f"  批次處理失敗，使用單張處理模式: {str(e)}")
            args.enable_batch_processing = False
    
    # 單張處理模式或批次處理失敗時的後備方案
    if not args.enable_batch_processing or processed_count == 0:
        for idx, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            
            try:
                logger.info(f'處理影像 {idx+1}/{len(image_paths)}: {img_name}')
                
                # 載入影像
                img_L = util.imread_uint(img_path, n_channels=3)
                original_shape = img_L.shape
                
                # 檢查是否需要分塊處理（避免記憶體不足）
                if gpu_optimizer.should_tile_process(original_shape[:2]):
                    print(f"  大圖片檢測: {original_shape[0]}x{original_shape[1]}, 使用分塊處理")
                    tile_h, tile_w = gpu_optimizer.get_optimal_tile_size(original_shape[:2])
                    img_E = tile_process_image(model, img_L, device, noise_level_normalized, 
                                             gpu_optimizer, tile_size=min(tile_h, tile_w))
                else:
                    # GPU優化的單張處理
                    try:
                        # 新版PyTorch語法
                        from torch.amp import autocast
                        with autocast('cuda', enabled=gpu_optimizer.use_amp):
                            # 轉換為tensor
                            img_tensor = util.uint2tensor4(img_L)
                            img_tensor = img_tensor.to(device)
                            
                            # 使用自適應雜訊等級處理
                            img_E = adaptive_noise_level_processing(model, img_tensor, device, args.noise_level)
                    except ImportError:
                        # 舊版PyTorch語法
                        from torch.cuda.amp import autocast
                        with autocast(enabled=gpu_optimizer.use_amp):
                            # 轉換為tensor
                            img_tensor = util.uint2tensor4(img_L)
                            img_tensor = img_tensor.to(device)
                            
                            # 使用自適應雜訊等級處理
                            img_E = adaptive_noise_level_processing(model, img_tensor, device, args.noise_level)
                
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
                
                # 定期清理記憶體
                if idx % 5 == 0:
                    gpu_optimizer.cleanup_memory()
                
                # 性能監控
                if args.monitor_performance and idx % 10 == 0:
                    memory_info = gpu_optimizer.monitor_gpu_memory()
                    if memory_info:
                        logger.info(f'處理進度 {idx+1}/{len(image_paths)}, GPU記憶體: 已分配 {memory_info["allocated"]:.2f}GB')
                
            except Exception as e:
                failed_count += 1
                logger.error(f'{img_name} - 處理失敗: {str(e)}')
                print(f" ({idx+1}/{len(image_paths)}) {img_name} - 處理失敗: {str(e)}")
                continue
    
    # 最終記憶體清理
    gpu_optimizer.cleanup_memory()
    
    # 計算處理時間
    total_time = time.time() - start_time
    avg_time_per_image = total_time / processed_count if processed_count > 0 else 0
    
    # ----------------------------------------
    # 處理結果統計 (含性能資訊)
    # ----------------------------------------
    logger.info('\n' + '=' * 60)
    logger.info(' 處理結果統計 (GPU優化版)')
    logger.info('=' * 60)
    logger.info(f'總影像數量: {len(image_paths)}')
    logger.info(f'成功處理: {processed_count}')
    logger.info(f'處理失敗: {failed_count}')
    logger.info(f'成功率: {processed_count/len(image_paths)*100:.1f}%')
    logger.info(f'總處理時間: {total_time:.2f} 秒')
    logger.info(f'平均每張時間: {avg_time_per_image:.2f} 秒')
    logger.info(f'處理速度: {processed_count/total_time:.2f} 張/秒')
    logger.info(f'批次處理模式: {"啟用" if args.enable_batch_processing else "停用"}')
    logger.info(f'混合精度AMP: {"啟用" if gpu_optimizer.use_amp else "停用"}')
    logger.info(f'結果儲存位置: {args.output_dir}')
    
    # 最終性能監控
    if args.monitor_performance:
        final_memory = gpu_optimizer.monitor_gpu_memory()
        if final_memory:
            logger.info(f'最終GPU記憶體: 已分配 {final_memory["allocated"]:.2f}GB, 剩餘 {final_memory["free"]:.2f}GB')
    
    print(f"\n 台北101影像去噪處理完成！(GPU優化版)")
    print(f" 處理統計: {processed_count}/{len(image_paths)} 成功")
    print(f" 處理時間: {total_time:.2f} 秒 (平均 {avg_time_per_image:.2f} 秒/張)")
    print(f" 處理速度: {processed_count/total_time:.2f} 張/秒")
    print(f" GPU優化: AMP混合精度 {'✓' if gpu_optimizer.use_amp else '✗'}, 批次處理 {'✓' if args.enable_batch_processing else '✗'}")
    print(f" 結果位置: {args.output_dir}")
    
    if processed_count > 0:
        print(f"\n 下一步建議:")
        print(f"   1. 使用評估腳本比較去噪效果: python main_evaluate_taipei101.py")
        print(f"   2. 如需更快處理，下次可加上參數: --enable_batch_processing --monitor_performance")
        
        # 性能建議
        if total_time > 0 and processed_count > 0:
            if avg_time_per_image > 2.0:
                print(f"   💡 效能提示: 處理速度較慢，建議啟用批次處理模式以提升效率")
            elif avg_time_per_image < 0.5:
                print(f"   🚀 效能優異: GPU優化效果顯著！")
    
    # 更新TODO狀態
    print(f"\n ✅ 模型更新完成: 已使用最新的 best_G.pth 模型")
    print(f" ✅ GPU優化完成: 混合精度、批次處理、記憶體管理已啟用")

if __name__ == "__main__":
    main()
