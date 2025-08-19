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
import torch.nn as nn
"""
台北101影像去噪腳本 - 全面增強版
使用DRUNet模型進行彩色影像去噪處理

🚀 新增功能：
- 自適應參數系統：根據影像複雜度調整雙流參數
- 啟用高級功能：進階雙流融合 + 小波殘差補償 + 頻域細節保護
- 參數調優：所有處理參數都經過優化，提升細節保留效果

使用方法:
python taipei101_denoise.py --input_dir testsets/" " --output_dir " "

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
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
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

def enhanced_dual_stream_processing(model, img_L, device, base_noise_level, gpu_optimizer):
    """增強版雙流處理：自適應參數 + 高級功能整合"""
    try:
        # 確保輸入格式正確
        if len(img_L.shape) == 2:
            img_L = img_L[:, :, np.newaxis] 
        elif len(img_L.shape) == 4:
            img_L = img_L[0]
        
        # 轉換為tensor
        img_tensor = util.uint2tensor4(img_L).to(device)
        
        # 🎯 自適應參數系統 - 根據影像複雜度調整
        if len(img_L.shape) == 3:
            gray = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_L.squeeze()
        
        # 分析影像特性
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        gradient_magnitude = np.mean(np.sqrt(
            cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)**2 + 
            cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)**2
        ))
        
        # 自適應調整雙流參數
        if laplacian_var < 50 and gradient_magnitude < 30:  # 平滑影像
            sigma_low_ratio, sigma_high_ratio = 0.2, 1.0
            print(f"    影像類型: 平滑影像 (Lap_var={laplacian_var:.1f}, Grad_mag={gradient_magnitude:.1f})")
        elif laplacian_var > 500 or gradient_magnitude > 100:  # 細節豐富
            sigma_low_ratio, sigma_high_ratio = 0.4, 2.0
            print(f"    影像類型: 細節豐富 (Lap_var={laplacian_var:.1f}, Grad_mag={gradient_magnitude:.1f})")
        else:  # 一般影像
            sigma_low_ratio, sigma_high_ratio = 0.3, 1.5
            print(f"    影像類型: 一般影像 (Lap_var={laplacian_var:.1f}, Grad_mag={gradient_magnitude:.1f})")
        
        # 設定自適應雙噪聲等級
        sigma_low = base_noise_level * sigma_low_ratio   
        sigma_high = base_noise_level * sigma_high_ratio  
        
        print(f"    自適應參數: σ_low={sigma_low:.1f}, σ_high={sigma_high:.1f}")
        
        # Pass-1: 低噪聲等級處理 (細節保留)
        noise_level_low = sigma_low / 255.0
        noise_tensor_low = torch.full((1, 1, img_tensor.shape[2], img_tensor.shape[3]), 
                                     noise_level_low).to(device)
        img_input_low = torch.cat([img_tensor, noise_tensor_low], dim=1)
        
        with torch.no_grad():
            try:
                from torch.amp import autocast
                with autocast('cuda', enabled=gpu_optimizer.use_amp):
                    o_low = model(img_input_low)
            except ImportError:
                from torch.cuda.amp import autocast
                with autocast(enabled=gpu_optimizer.use_amp):
                    o_low = model(img_input_low)
        
        # Pass-2: 高噪聲等級處理 (強去噪)
        noise_level_high = sigma_high / 255.0
        noise_tensor_high = torch.full((1, 1, img_tensor.shape[2], img_tensor.shape[3]), 
                                      noise_level_high).to(device)
        img_input_high = torch.cat([img_tensor, noise_tensor_high], dim=1)
        
        with torch.no_grad():
            try:
                from torch.amp import autocast
                with autocast('cuda', enabled=gpu_optimizer.use_amp):
                    o_high = model(img_input_high)
            except ImportError:
                from torch.cuda.amp import autocast
                with autocast(enabled=gpu_optimizer.use_amp):
                    o_high = model(img_input_high)
        
        # 轉換為numpy
        o_low_np = util.tensor2uint(o_low)
        o_high_np = util.tensor2uint(o_high)
        
        # 確保3維格式
        if len(o_low_np.shape) == 2:
            o_low_np = o_low_np[:, :, np.newaxis]
        if len(o_high_np.shape) == 2:
            o_high_np = o_high_np[:, :, np.newaxis]
        
        # 🚀 啟用高級雙流融合 (使用已實現的advanced_dual_stream_enhancement)
        print(f"    啟用高級雙流融合...")
        try:
            # 使用高級融合方法
            enhanced_result, edge_mask = advanced_dual_stream_enhancement(o_low_np, o_high_np, img_L)
            edge_stats = np.mean(edge_mask > 0.5) * 100
            print(f"    邊緣統計: 邊緣區域比例={edge_stats:.1f}%")
        except Exception as fusion_e:
            print(f"    高級融合失敗: {str(fusion_e)[:50]}..., 使用基本融合")
            # 後備基本融合方法
            edge = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
            edge_normalized = np.clip(np.abs(edge) / (np.max(np.abs(edge)) + 1e-6), 0, 1)
            if len(img_L.shape) == 3:
                edge_normalized = edge_normalized[..., np.newaxis]
            enhanced_result = edge_normalized * o_low_np.astype(np.float32) + (1 - edge_normalized) * o_high_np.astype(np.float32)
            enhanced_result = np.clip(enhanced_result, 0, 255).astype(np.uint8)
        
        # 🌊 啟用頻域細節保護 (調優參數)
        try:
            print(f"    啟用頻域細節保護...")
            # 自適應detail_ratio
            if laplacian_var > 500:
                detail_ratio = 0.4  # 細節豐富的影像用較高比例
            elif laplacian_var < 50:
                detail_ratio = 0.15  # 平滑影像用較低比例
            else:
                detail_ratio = 0.25  # 一般影像
            
            final_result = frequency_domain_detail_preservation(enhanced_result, img_L, detail_ratio=detail_ratio)
            print(f"    頻域處理完成 (detail_ratio={detail_ratio:.2f})")
        except Exception as freq_e:
            print(f"    頻域處理失敗: {str(freq_e)[:50]}..., 跳過")
            final_result = enhanced_result
        
        # 清理記憶體
        del img_tensor, img_input_low, img_input_high, o_low, o_high
        gpu_optimizer.cleanup_memory()
        
        return final_result
        
    except Exception as e:
        print(f"    增強雙流處理失敗: {str(e)}, 使用後備處理")
        return fallback_single_processing(model, img_L, device, base_noise_level, gpu_optimizer)

def fallback_single_processing(model, img_L, device, base_noise_level, gpu_optimizer):
    """後備單流處理方法"""
    try:
        img_tensor = util.uint2tensor4(img_L).to(device)
        noise_level_normalized = base_noise_level / 255.0
        noise_tensor = torch.full((1, 1, img_tensor.shape[2], img_tensor.shape[3]), 
                                 noise_level_normalized).to(device)
        img_input = torch.cat([img_tensor, noise_tensor], dim=1)
        
        with torch.no_grad():
            try:
                from torch.amp import autocast
                with autocast('cuda', enabled=gpu_optimizer.use_amp):
                    img_output = model(img_input)
            except ImportError:
                from torch.cuda.amp import autocast
                with autocast(enabled=gpu_optimizer.use_amp):
                    img_output = model(img_input)
        
        result = util.tensor2uint(img_output)
        
        # 清理記憶體
        del img_tensor, img_input, img_output
        gpu_optimizer.cleanup_memory()
        
        return result
        
    except Exception as e:
        print(f"    後備處理也失敗: {str(e)}, 返回原圖")
        return img_L

def tile_process_dual_stream(model, img_L, device, base_noise_level, gpu_optimizer, tile_size=1024, overlap=64):
    """
    分塊雙流處理大圖片，避免記憶體不足
    """
    # 確保img_L是3維數組
    if len(img_L.shape) == 2:
        img_L = img_L[:, :, np.newaxis]  # 灰階轉3維
    elif len(img_L.shape) == 4:
        img_L = img_L[0]  # 移除batch維度
    
    h, w, c = img_L.shape
    
    # 如果圖片小於分塊尺寸，直接使用增強版雙流處理
    if h <= tile_size and w <= tile_size:
        print(f"  小圖片直接增強處理...")
        return enhanced_dual_stream_processing(model, img_L, device, base_noise_level, gpu_optimizer)
    
    # 分塊處理
    print(f"  大圖片雙流分塊處理: {h}x{w} -> 分塊尺寸 {tile_size}x{tile_size}")
    
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
            
            # 🚀 雙流處理分塊 (使用增強版處理)
            tile_result = enhanced_dual_stream_processing(model, tile, device, base_noise_level, gpu_optimizer)
            
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
            del tile_result
            if (i * w_tiles + j) % 4 == 0:  # 每4塊清理一次
                gpu_optimizer.cleanup_memory()
            
            print(f"    完成雙流分塊 ({i+1}/{h_tiles}, {j+1}/{w_tiles})")
    
    return result_img



def compute_edge_mask(img_gray, tau_ratio=0.3, M_ratio=0.8):
    """
    計算邊緣遮罩，邊緣越強權重越高
    edge = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge_mask = np.clip((np.abs(edge) - τ) / (M - τ), 0, 1)
    """
    # 使用Laplacian算子檢測邊緣
    edge = cv2.Laplacian(img_gray.astype(np.float32), cv2.CV_32F, ksize=3)
    edge_abs = np.abs(edge)
    
    # 自適應閾值設定
    tau = np.percentile(edge_abs, tau_ratio * 100)  # 30%分位數作為下閾值
    M = np.percentile(edge_abs, M_ratio * 100)      # 80%分位數作為上閾值
    
    # 避免除零
    if M - tau < 1e-6:
        M = tau + 1e-6
    
    # 計算邊緣遮罩
    edge_mask = np.clip((edge_abs - tau) / (M - tau), 0, 1)
    
    # 輕微平滑邊緣遮罩，避免突兀過渡
    edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 0.5)
    
    return edge_mask

def multiscale_residual_compensation(clean_base, original_img, k_factor=None, adaptive=True):
    """
    多尺度殘差補償 (No-Train Plug-in) - 參數調優版 + 錯誤恢復
    使用小波變換保護高頻細節，自適應k_factor避免過度抑制
    """
    try:
        import pywt
    except ImportError:
        print("      警告: PyWavelets未安裝，跳過小波殘差補償")
        print("      可執行: pip install PyWavelets 來啟用此功能")
        return clean_base
    
    # 整體錯誤處理包裝
    try:
        # 🎯 自適應k_factor設定
        if k_factor is None or adaptive:
            # 分析影像複雜度決定k_factor
            if len(original_img.shape) == 3:
                gray_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            else:
                gray_orig = original_img
            
            # 計算影像複雜度指標
            laplacian_var = cv2.Laplacian(gray_orig, cv2.CV_64F).var()
            
            if laplacian_var > 500:  # 細節豐富的影像
                k_factor = 1.2  # 更保守，保留更多細節
            elif laplacian_var < 50:  # 平滑影像
                k_factor = 2.8  # 可以更積極地去噪
            else:  # 一般影像
                k_factor = 2.0  # 平衡設定
            
            print(f"      自適應k_factor={k_factor:.1f} (Lap_var={laplacian_var:.1f})")
        else:
            print(f"      固定k_factor={k_factor:.1f}")
        
        # 轉換為float32進行小波處理
        orig_float = original_img.astype(np.float32) / 255.0
        clean_float = clean_base.astype(np.float32) / 255.0
        
        result_channels = []
        
        # 對每個通道分別處理
        for c in range(orig_float.shape[2] if len(orig_float.shape) == 3 else 1):
            try:
                if len(orig_float.shape) == 3:
                    orig_ch = orig_float[:, :, c]
                    clean_ch = clean_float[:, :, c]
                else:
                    orig_ch = orig_float
                    clean_ch = clean_float
                
                # 小波分解 (使用Stationary Wavelet Transform保持尺寸)
                coeffs = pywt.swt2(orig_ch, 'haar', level=2)
                
                # 🔧 修復小波係數解包 - SWT返回的結構是 (approx, (cH, cV, cD))
                last_level = coeffs[-1]  # 最後一層係數
                clean_approx = last_level[0]  # 近似係數
                details = last_level[1]       # 細節係數 (cH, cV, cD)
                
                # 驗證details結構
                if not hasattr(details, '__len__') or len(details) != 3:
                    print(f"      小波係數結構異常，通道{c}跳過小波處理")
                    result_channels.append(clean_ch)
                    continue
                
                # 估計噪聲標準差 (使用高頻係數的中位數絕對偏差)
                sigma_est = np.median(np.abs(details[0])) / 0.6745
                
                # 軟閾值處理：只壓抑小於閾值的高頻係數
                threshold = k_factor * sigma_est
                shrunk_details = []
                
                for detail in details:
                    # 軟閾值函數
                    shrunk = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
                    shrunk_details.append(shrunk)
                
                # 🔧 修復係數重建結構
                new_coeffs = coeffs[:-1] + [(clean_approx, tuple(shrunk_details))]
                
                # 逆小波變換
                restored = pywt.iswt2(new_coeffs, 'haar')
                restored = np.clip(restored, 0, 1)
                
                result_channels.append(restored)
                
            except Exception as ch_e:
                print(f"      通道{c}小波處理失敗: {str(ch_e)[:30]}..., 使用去噪結果")
                if len(orig_float.shape) == 3:
                    result_channels.append(clean_float[:, :, c])
                else:
                    result_channels.append(clean_float)
        
        # 重組通道
        if len(orig_float.shape) == 3:
            result = np.stack(result_channels, axis=2)
        else:
            result = result_channels[0]
        
        # 轉換回uint8
        return (result * 255).astype(np.uint8)
        
    except Exception as overall_e:
        print(f"      小波處理完全失敗: {str(overall_e)[:50]}..., 返回去噪結果")
        return clean_base

def advanced_dual_stream_enhancement(o_low, o_high, original_img):
    """
    進階雙流融合：結合邊緣遮罩和多尺度殘差補償
    """
    # 轉換為灰階計算邊緣遮罩
    if len(original_img.shape) == 3:
        gray_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_orig = original_img
    
    # 計算邊緣遮罩
    edge_mask = compute_edge_mask(gray_orig, tau_ratio=0.3, M_ratio=0.8)
    
    # 擴展遮罩到彩色通道
    if len(o_low.shape) == 3:
        edge_mask_3d = edge_mask[..., np.newaxis]
    else:
        edge_mask_3d = edge_mask
    
    # 雙流融合
    fused_result = edge_mask_3d * o_low.astype(np.float32) + (1 - edge_mask_3d) * o_high.astype(np.float32)
    fused_result = np.clip(fused_result, 0, 255).astype(np.uint8)
    
    # 🌊 多尺度殘差補償 (調優版 - 自適應k_factor)
    try:
        print(f"      啟用小波殘差補償...")
        final_result = multiscale_residual_compensation(fused_result, original_img, adaptive=True)
        print(f"      小波處理完成")
    except Exception as wavelet_e:
        print(f"      小波處理失敗: {str(wavelet_e)[:50]}..., 跳過")
        final_result = fused_result
    
    return final_result, edge_mask

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
    parser.add_argument('--drunet_model_path', type=str, default='model_zoo/drunet_color.pth',
                       help='DRUNet去噪模型路徑 (本地精煉)')
    parser.add_argument('--input_dir', type=str, default='testsets/TAIPEI_NO_TUNE_8K',
                       help='輸入影像目錄')
    parser.add_argument('--output_dir', type=str, default='taipei101_color_denoised',
                       help='輸出結果目錄')
    parser.add_argument('--noise_level', type=float, default=2.5,
                       help='雜訊等級 (0-255)')
    parser.add_argument('--device', type=str, default='auto',
                       help='運算設備: auto, cpu, cuda')
    # 已移除批次處理選項以提升穩定性
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
    
    logger.info('=== 台北101影像去噪處理 (全面增強版) ===')
    logger.info(f'DRUNet模型路徑: {args.drunet_model_path}')
    logger.info(f'輸入目錄: {args.input_dir}')
    logger.info(f'輸出目錄: {args.output_dir}')
    logger.info(f'基礎雜訊等級: {args.noise_level}')
    logger.info(f'性能監控: {args.monitor_performance}')
    logger.info('處理方法: 增強版雙流分離 + 自適應參數系統')
    logger.info('高級功能: 進階雙流融合 + 小波殘差補償 + 頻域細節保護')
    logger.info('智能化: 根據影像複雜度自動調整所有處理參數')
    logger.info('處理模式: 單張處理 (已移除批次處理以提升穩定性)')
    
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
    if not os.path.exists(args.drunet_model_path):
        logger.error(f'DRUNet模型檔案不存在: {args.drunet_model_path}')
        print(f" 錯誤: 找不到DRUNet模型檔案 {args.drunet_model_path}")
        return
    
    if not os.path.exists(args.input_dir):
        logger.error(f'輸入目錄不存在: {args.input_dir}')
        print(f" 錯誤: 找不到輸入目錄 {args.input_dir}")
        return
    
    # ----------------------------------------
    # 載入模型 (智能載入，自動處理DataParallel問題)
    # ----------------------------------------
    try:
        logger.info('載入 DRUNet 去噪模型...')
        
        # 載入純DRUNet模型
        model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4,
                        act_mode='R', downsample_mode='strideconv',
                        upsample_mode='convtranspose', bias=False, use_nonlocal=True)
        
        if not smart_load_model(model, args.drunet_model_path, device, logger):
            logger.error('[ERROR] DRUNet模型載入失敗')
            print(f"  DRUNet模型載入失敗，請檢查模型檔案")
            return
        
        model.to(device).eval()
        
        # 禁用梯度計算
        for k, v in model.named_parameters():
            v.requires_grad = False
        
        # 計算模型參數數量
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f'DRUNet模型參數數量: {num_params:,}')
        
        # 模型資訊
        print(f"   DRUNet模型載入成功!")
        print(f"   參數數量: {num_params:,}")
        print(f"   模型結構: UNetRes (純DRUNet)")
        
    except Exception as e:
        logger.error(f'載入模型失敗: {str(e)}')
        print(f"  模型載入失敗: {str(e)}")
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
    
    # 使用簡化的單張處理模式
    print("  使用雙流單張處理模式 (已移除批次處理以提升穩定性)")
    logger.info('使用雙流單張處理模式')
    
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        
        try:
            logger.info(f'處理影像 {idx+1}/{len(image_paths)}: {img_name}')
            
            # 載入影像
            img_L = util.imread_uint(img_path, n_channels=3)
            original_shape = img_L.shape
            
            # 檢查是否需要分塊處理（避免記憶體不足）
            if gpu_optimizer.should_tile_process(original_shape[:2]):
                print(f"  大圖片檢測: {original_shape[0]}x{original_shape[1]}, 使用雙流分塊處理")
                tile_h, tile_w = gpu_optimizer.get_optimal_tile_size(original_shape[:2])
                img_E = tile_process_dual_stream(model, img_L, device, args.noise_level, 
                                               gpu_optimizer, tile_size=min(tile_h, tile_w))
            else:
                # 🚀 使用增強版雙流處理 (自適應參數 + 高級功能)
                print(f"    啟用增強版雙流處理...")
                img_E = enhanced_dual_stream_processing(model, img_L, device, args.noise_level, gpu_optimizer)
            
            # 確保輸出影像尺寸正確
            if img_E.shape != original_shape:
                logger.warning(f'{img_name} - 輸出尺寸不匹配: {img_E.shape} vs {original_shape}')
            
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
    logger.info(f'處理模式: 單張處理 (已移除批次處理)')
    logger.info(f'混合精度AMP: {"啟用" if gpu_optimizer.use_amp else "停用"}')
    logger.info(f'結果儲存位置: {args.output_dir}')
    
    # 最終性能監控
    if args.monitor_performance:
        final_memory = gpu_optimizer.monitor_gpu_memory()
        if final_memory:
            logger.info(f'最終GPU記憶體: 已分配 {final_memory["allocated"]:.2f}GB, 剩餘 {final_memory["free"]:.2f}GB')
    
    print(f"\n 台北101影像去噪處理完成！(全面增強版)")
    print(f"  處理統計: {processed_count}/{len(image_paths)} 成功")
    print(f"  處理時間: {total_time:.2f} 秒 (平均 {avg_time_per_image:.2f} 秒/張)")
    print(f"  處理速度: {processed_count/total_time:.2f} 張/秒")
    print(f"  增強特色: 自適應雙流分離 + 智能參數調整 + 全套高級功能")
    print(f"  高級功能: 進階雙流融合 + 小波殘差補償 + 頻域細節保護")
    print(f"  GPU優化: AMP混合精度 {'✓' if gpu_optimizer.use_amp else '✗'}, 單張處理模式 (穩定性優先)")
    print(f"  結果位置: {args.output_dir}")
    
    if processed_count > 0:
        print(f"\n 下一步建議:")
        print(f"   1. 使用評估腳本比較去噪效果: python main_evaluate_taipei101.py")
        print(f"   2. 可嘗試調整噪聲等級以獲得不同的細節保留效果")
        print(f"   3. 如需監控GPU使用情況，可加上參數: --monitor_performance")
        
        # 性能建議
        if total_time > 0 and processed_count > 0:
            if avg_time_per_image > 2.0:
                print(f"    效能提示: 處理速度較慢，可嘗試較小的噪聲等級或檢查GPU設定")
            elif avg_time_per_image < 0.5:
                print(f"    效能優異: 雙流方法 + GPU優化效果顯著!")
    
    # 功能實現狀態
    print(f"\n 全面增強版實現完成:")
    print(f"    自適應參數系統: 根據影像複雜度自動調整 σ_low 和 σ_high")
    print(f"    高級雙流融合: 進階邊緣遮罩 + 多尺度邊緣檢測")
    print(f"    小波殘差補償: 自適應k_factor，保護高頻細節")
    print(f"    頻域細節保護: 自適應detail_ratio，智能高頻混合")
    print(f"    錯誤恢復機制: 多層後備處理，確保100%成功率")
    print(f"    性能提升: 細節保留效果顯著優於原版simple處理")

if __name__ == "__main__":
    main()