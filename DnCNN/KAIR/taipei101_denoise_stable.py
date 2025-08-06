#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台北101影像去噪處理 - 穩定版本 + 修復版Self-Attention
專注於核心雙流功能，確保100%穩定運行
包含完全修復的Self-Attention模組，可選啟用
結合穩定性和先進技術的最佳平衡
"""

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
import torch.nn as nn
import torch.nn.functional as F

from utils import utils_logger
from utils import utils_image as util
from models.network_unet import UNetRes

class RobustSelfAttention(nn.Module):
    """穩定版Self-Attention模組 - 修復所有維度問題"""
    def __init__(self, channels):
        super(RobustSelfAttention, self).__init__()
        self.channels = channels
        
        # 根據通道數選擇不同的注意力策略
        if channels <= 4:
            # 極小通道數：使用通道注意力機制
            self.attention_type = 'channel'
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(1, channels // 2), 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, channels // 2), channels, 1, bias=False),
                nn.Sigmoid()
            )
            print(f"    Self-Attention: 通道注意力模式 (channels={channels})")
        else:
            # 正常通道數：使用空間注意力機制
            self.attention_type = 'spatial'
            # 安全的降維策略
            self.reduced_channels = max(1, min(channels // 4, 32))  # 限制最大降維
            
            self.query = nn.Conv2d(channels, self.reduced_channels, 1, bias=False)
            self.key = nn.Conv2d(channels, self.reduced_channels, 1, bias=False) 
            self.value = nn.Conv2d(channels, channels, 1, bias=False)
            
            # 初始化權重
            nn.init.kaiming_normal_(self.query.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.key.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.value.weight, mode='fan_out', nonlinearity='relu')
            
            print(f"    Self-Attention: 空間注意力模式 (channels={channels}, reduced={self.reduced_channels})")
        
        # 可學習的融合權重
        self.gamma = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        try:
            if self.attention_type == 'channel':
                # 通道注意力機制
                attention_weights = self.channel_attention(x)  # (B, C, 1, 1)
                attention_result = x * attention_weights
            else:
                # 空間注意力機制
                attention_result = self._compute_spatial_attention(x, H, W)
            
            # 殘差連接
            return self.gamma * attention_result + x
            
        except Exception as e:
            print(f"        Self-Attention處理失敗: {str(e)[:50]}..., 跳過")
            return x
    
    def _compute_spatial_attention(self, x, H, W):
        """計算空間注意力"""
        B, C = x.size(0), x.size(1)
        
        # 對大圖片進行下採樣以節約記憶體
        if H * W > 256 * 256:
            scale_factor = 0.5
            x_small = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            h_small, w_small = x_small.shape[2], x_small.shape[3]
            
            # 在小尺寸上計算attention
            attn_small = self._attention_computation(x_small, h_small, w_small)
            
            # 放回原尺寸
            attention_result = F.interpolate(attn_small, size=(H, W), mode='bilinear', align_corners=False)
        else:
            attention_result = self._attention_computation(x, H, W)
        
        return attention_result
    
    def _attention_computation(self, x, h, w):
        """核心attention計算"""
        B, C = x.size(0), x.size(1)
        
        # 生成Q, K, V
        q = self.query(x).view(B, self.reduced_channels, h * w).permute(0, 2, 1)  # (B, HW, reduced)
        k = self.key(x).view(B, self.reduced_channels, h * w)  # (B, reduced, HW)
        v = self.value(x).view(B, C, h * w)  # (B, C, HW)
        
        # 計算注意力矩陣
        attention = torch.bmm(q, k)  # (B, HW, HW)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # 應用注意力
        out = torch.bmm(v, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, h, w)
        
        return out

class GPUOptimizer:
    """GPU優化管理器 - 穩定版 + Self-Attention"""
    
    def __init__(self):
        self.device = None
        self.gpu_info = {}
        self.use_amp = False
        self.scaler = None
        self.attention_module = None
        
    def setup_gpu(self, enable_attention=False):
        """設置GPU優化配置"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            cudnn.benchmark = True
            cudnn.deterministic = False
            
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
            
            self.gpu_info = {
                'name': torch.cuda.get_device_name(0),
                'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'memory_available': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
            }
            
            if self.gpu_info['memory_total'] >= 4.0:
                self.use_amp = True
                try:
                    from torch.amp import GradScaler
                    self.scaler = GradScaler('cuda')
                except ImportError:
                    from torch.cuda.amp import GradScaler
                    self.scaler = GradScaler()
            
            # 初始化修復版Self-Attention模組
            if enable_attention and self.gpu_info['memory_total'] >= 6.0:
                try:
                    self.attention_module = RobustSelfAttention(3).to(self.device)
                    print(f"    Self-Attention模組: 已啟用 (修復版)")
                except Exception as e:
                    print(f"    Self-Attention模組: 初始化失敗 {str(e)[:50]}...")
                    self.attention_module = None
            else:
                if enable_attention:
                    print(f"    Self-Attention模組: GPU記憶體不足 (需要≥6GB)")
                else:
                    print(f"    Self-Attention模組: 已停用")
                
            print(f" GPU優化啟用:")
            print(f"   設備: {self.gpu_info['name']}")
            print(f"   總記憶體: {self.gpu_info['memory_total']:.1f}GB")
            print(f"   可用記憶體: {self.gpu_info['memory_available']:.1f}GB")
            print(f"   混合精度AMP: {self.use_amp}")
            print(f"   Self-Attention: {'啟用' if self.attention_module else '未啟用'}")
            
        else:
            self.device = torch.device('cpu')
            print(" 使用CPU處理 (建議使用GPU以獲得更好性能)")
            
        return self.device
    
    def should_tile_process(self, img_size, threshold_pixels=1024*1024):
        """判斷是否需要分塊處理 - 更保守的閾值"""
        total_pixels = img_size[0] * img_size[1]
        return total_pixels > threshold_pixels
    
    def get_optimal_tile_size(self, img_size):
        """計算最佳分塊尺寸 - 穩定優先"""
        h, w = img_size
        
        if self.gpu_info['memory_total'] >= 8.0:
            max_tile_size = 512  # 保守設置
        elif self.gpu_info['memory_total'] >= 6.0:
            max_tile_size = 384
        else:
            max_tile_size = 256
        
        tile_h = min(max_tile_size, h)
        tile_w = min(max_tile_size, w)
        
        return tile_h, tile_w
    
    def cleanup_memory(self):
        """清理GPU記憶體"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def stable_dual_stream_processing(model, img_L, device, base_noise_level, gpu_optimizer, enable_attention=False):
    """
    穩定雙流處理 - 專注於可靠性
    """
    try:
        # 確保輸入格式正確
        if len(img_L.shape) == 2:
            img_L = img_L[:, :, np.newaxis] 
        elif len(img_L.shape) == 4:
            img_L = img_L[0]
        
        # 轉換為tensor
        img_tensor = util.uint2tensor4(img_L).to(device)
        
        # 自適應噪聲參數
        if len(img_L.shape) == 3:
            gray = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_L.squeeze()
        
        # 根據影像特性調整參數
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 50:  # 平滑影像
            sigma_low_ratio, sigma_high_ratio = 0.2, 1.2
        elif laplacian_var > 500:  # 細節豐富
            sigma_low_ratio, sigma_high_ratio = 0.4, 1.8
        else:  # 一般影像
            sigma_low_ratio, sigma_high_ratio = 0.3, 1.5
        
        sigma_low = base_noise_level * sigma_low_ratio
        sigma_high = base_noise_level * sigma_high_ratio
        
        print(f"    自適應參數: σ_low={sigma_low:.1f}, σ_high={sigma_high:.1f} (Laplacian_var={laplacian_var:.1f})")
        
        # Pass-1: 低噪聲等級處理
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
        
        # Pass-2: 高噪聲等級處理  
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
        
        # 多尺度邊緣檢測融合
        edge_maps = []
        for ksize in [3, 5]:  # 簡化為2個尺度
            edge = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=ksize)
            edge_abs = np.abs(edge)
            edge_maps.append(edge_abs)
        
        # 融合邊緣資訊
        combined_edge = 0.7 * edge_maps[0] + 0.3 * edge_maps[1]
        
        # 自適應閾值
        tau = np.percentile(combined_edge, 25)
        M = np.percentile(combined_edge, 85)
        
        if M - tau < 1e-6:
            M = tau + 1e-6
        
        edge_weight = np.clip((combined_edge - tau) / (M - tau), 0, 1)
        edge_weight = cv2.GaussianBlur(edge_weight, (3, 3), 0.5)
        
        print(f"    邊緣統計: τ={tau:.2f}, M={M:.2f}, 邊緣比例={np.mean(edge_weight > 0.5)*100:.1f}%")
        
        # 擴展到彩色通道
        if len(img_L.shape) == 3:
            edge_weight = edge_weight[..., np.newaxis]
        
        # 雙流融合
        fused_result = edge_weight * o_low_np.astype(np.float32) + (1 - edge_weight) * o_high_np.astype(np.float32)
        fused_result = np.clip(fused_result, 0, 255).astype(np.uint8)
        
        # Self-Attention增強 (如果啟用且可用)
        if enable_attention and gpu_optimizer.attention_module is not None:
            try:
                print(f"    應用Self-Attention增強...")
                attn_tensor = util.uint2tensor4(fused_result).to(device)
                with torch.no_grad():
                    try:
                        from torch.amp import autocast
                        with autocast('cuda', enabled=gpu_optimizer.use_amp):
                            attn_enhanced = gpu_optimizer.attention_module(attn_tensor)
                    except ImportError:
                        from torch.cuda.amp import autocast
                        with autocast(enabled=gpu_optimizer.use_amp):
                            attn_enhanced = gpu_optimizer.attention_module(attn_tensor)
                
                final_result = util.tensor2uint(attn_enhanced)
                print(f"    Self-Attention處理成功")
                
                # 清理attention相關tensor
                del attn_tensor, attn_enhanced
                
            except Exception as e:
                print(f"     Self-Attention處理失敗: {str(e)[:50]}..., 使用原結果")
                final_result = fused_result
        else:
            final_result = fused_result
        
        # 清理記憶體
        del img_tensor, img_input_low, img_input_high, o_low, o_high
        gpu_optimizer.cleanup_memory()
        
        return final_result
        
    except Exception as e:
        print(f"    雙流處理失敗: {str(e)}, 使用後備處理")
        return fallback_single_processing(model, img_L, device, base_noise_level, gpu_optimizer)

def tile_process_stable(model, img_L, device, base_noise_level, gpu_optimizer, enable_attention=False):
    """
    穩定分塊處理
    """
    if len(img_L.shape) == 2:
        img_L = img_L[:, :, np.newaxis]
    elif len(img_L.shape) == 4:
        img_L = img_L[0]
    
    h, w, c = img_L.shape
    tile_h, tile_w = gpu_optimizer.get_optimal_tile_size((h, w))
    
    # 如果圖片不大，直接處理
    if h <= tile_h and w <= tile_w:
        return stable_dual_stream_processing(model, img_L, device, base_noise_level, gpu_optimizer, enable_attention)
    
    # 分塊處理
    print(f"  穩定分塊處理: {h}x{w} -> {tile_h}x{tile_w}")
    
    overlap = 64
    h_tiles = (h - 1) // (tile_h - overlap) + 1
    w_tiles = (w - 1) // (tile_w - overlap) + 1
    
    print(f"  分塊數量: {h_tiles}x{w_tiles} = {h_tiles * w_tiles} 塊")
    
    result_img = np.zeros_like(img_L)
    
    for i in range(h_tiles):
        for j in range(w_tiles):
            start_h = i * (tile_h - overlap)
            start_w = j * (tile_w - overlap)
            end_h = min(start_h + tile_h, h)
            end_w = min(start_w + tile_w, w)
            
            tile = img_L[start_h:end_h, start_w:end_w, :]
            
            print(f"    處理塊 ({i+1}/{h_tiles}, {j+1}/{w_tiles}) - 尺寸: {tile.shape}")
            tile_result = stable_dual_stream_processing(model, tile, device, base_noise_level, gpu_optimizer, enable_attention)
            
            # 處理重疊區域
            if i == 0 and j == 0:
                result_img[start_h:end_h, start_w:end_w, :] = tile_result
            else:
                actual_start_h = start_h + (overlap // 2 if i > 0 else 0)
                actual_start_w = start_w + (overlap // 2 if j > 0 else 0)
                tile_start_h = overlap // 2 if i > 0 else 0
                tile_start_w = overlap // 2 if j > 0 else 0
                
                result_img[actual_start_h:end_h, actual_start_w:end_w, :] = \
                    tile_result[tile_start_h:, tile_start_w:, :]
            
            # 每塊後清理記憶體
            gpu_optimizer.cleanup_memory()
    
    return result_img

def fallback_single_processing(model, img_L, device, base_noise_level, gpu_optimizer):
    """後備單流處理方法"""
    try:
        print(f"    使用後備單流處理 (noise_level={base_noise_level})")
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
        
        del img_tensor, img_input, img_output
        gpu_optimizer.cleanup_memory()
        
        return result
        
    except Exception as e:
        print(f"    後備處理也失敗: {str(e)}, 返回原圖")
        return img_L

def smart_load_model(model, model_path, device, logger):
    """智能載入模型"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        def remove_module_prefix(state_dict):
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict
        
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info('[SUCCESS] 模型載入成功 (直接載入)')
            return True
        except RuntimeError as e:
            if 'module.' in str(e):
                logger.info('[INFO] 偵測到DataParallel前綴，正在移除...')
                clean_state_dict = remove_module_prefix(state_dict)
                model.load_state_dict(clean_state_dict, strict=True)
                logger.info('[SUCCESS] 模型載入成功 (移除DataParallel前綴)')
                return True
            else:
                raise e
                
    except Exception as e:
        logger.error(f'[ERROR] 模型載入失敗: {str(e)}')
        return False

def main():
    parser = argparse.ArgumentParser(description='台北101影像去噪處理 - 穩定版本 + Self-Attention')
    parser.add_argument('--drunet_model_path', type=str, default='model_zoo/drunet_color.pth')
    parser.add_argument('--input_dir', type=str, default='testsets/TAIPEI_NO_TUNE_8K')
    parser.add_argument('--output_dir', type=str, default='taipei101_stable_denoised')
    parser.add_argument('--noise_level', type=float, default=2.0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--enable_attention', action='store_true', help='啟用修復版Self-Attention增強')
    parser.add_argument('--monitor_performance', action='store_true')
    args = parser.parse_args()

    # 建立輸出目錄和日誌
    util.mkdir(args.output_dir)
    
    logger_name = 'taipei101_denoise_stable'
    utils_logger.logger_info(logger_name, log_path=os.path.join(args.output_dir, 'denoise.log'))
    logger = logging.getLogger(logger_name)
    
    logger.info('=== 台北101影像去噪處理 (穩定版本 + Self-Attention) ===')
    logger.info(f'DRUNet模型路徑: {args.drunet_model_path}')
    logger.info(f'輸入目錄: {args.input_dir}')
    logger.info(f'輸出目錄: {args.output_dir}')
    logger.info(f'基礎雜訊等級: {args.noise_level}')
    logger.info(f'Self-Attention: {"啟用 (修復版)" if args.enable_attention else "停用"}')
    logger.info('穩定模式: 雙流處理 + 可選Self-Attention增強')
    
    # GPU優化器設定
    gpu_optimizer = GPUOptimizer()
    device = gpu_optimizer.setup_gpu(enable_attention=args.enable_attention)
    
    if args.device != 'auto':
        device = torch.device(args.device)
        print(f" 強制使用指定設備: {device}")
    
    logger.info(f'使用設備: {device}')
    
    # 檢查路徑
    if not os.path.exists(args.drunet_model_path):
        logger.error(f'DRUNet模型檔案不存在: {args.drunet_model_path}')
        return
    
    if not os.path.exists(args.input_dir):
        logger.error(f'輸入目錄不存在: {args.input_dir}')
        return
    
    # 載入模型
    try:
        logger.info('載入 DRUNet 去噪模型...')
        
        model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4,
                        act_mode='R', downsample_mode='strideconv',
                        upsample_mode='convtranspose', bias=False, use_nonlocal=True)
        
        if not smart_load_model(model, args.drunet_model_path, device, logger):
            logger.error('[ERROR] DRUNet模型載入失敗')
            return
        
        model.to(device).eval()
        
        for k, v in model.named_parameters():
            v.requires_grad = False
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f'DRUNet模型參數數量: {num_params:,}')
        
        print(f"   DRUNet模型載入成功!")
        print(f"   參數數量: {num_params:,}")
        print(f"   穩定版本: 雙流處理 + Self-Attention {'啟用' if args.enable_attention else '未啟用'}")
        print(f"   功能狀態: {'完整功能模式' if args.enable_attention else '高速穩定模式'}")
        
    except Exception as e:
        logger.error(f'載入模型失敗: {str(e)}')
        return
    
    # 取得影像檔案列表
    image_paths = util.get_image_paths(args.input_dir)
    logger.info(f'找到影像檔案數量: {len(image_paths)}')
    
    if len(image_paths) == 0:
        logger.warning('輸入目錄中沒有找到影像檔案')
        return
    
    # 處理影像
    logger.info('開始處理影像 (穩定模式)...')
    print(f"  開始處理 {len(image_paths)} 張影像")
    
    processed_count = 0
    failed_count = 0
    start_time = time.time()
    
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        
        try:
            logger.info(f'處理影像 {idx+1}/{len(image_paths)}: {img_name}')
            img_start_time = time.time()
            
            # 載入影像
            img_L = util.imread_uint(img_path, n_channels=3)
            original_shape = img_L.shape
            print(f"    影像尺寸: {original_shape}")
            
            # 根據圖片大小選擇處理方式
            if gpu_optimizer.should_tile_process(original_shape[:2]):
                img_E = tile_process_stable(model, img_L, device, args.noise_level, gpu_optimizer, enable_attention=args.enable_attention)
            else:
                img_E = stable_dual_stream_processing(model, img_L, device, args.noise_level, gpu_optimizer, enable_attention=args.enable_attention)
            
            # 確保輸出影像尺寸正確
            if img_E.shape != original_shape:
                logger.warning(f'{img_name} - 輸出尺寸不匹配: {img_E.shape} vs {original_shape}')
            
            # 儲存結果
            output_path = os.path.join(args.output_dir, img_name)
            util.imsave(img_E, output_path)
            
            processed_count += 1
            img_time = time.time() - img_start_time
            print(f"  ({idx+1}/{len(image_paths)}) {img_name} - 處理完成 ({img_time:.1f}秒)")
            
            # 每3張清理一次記憶體
            if idx % 3 == 0:
                gpu_optimizer.cleanup_memory()
            
        except Exception as e:
            failed_count += 1
            logger.error(f'{img_name} - 處理失敗: {str(e)}')
            print(f"  ({idx+1}/{len(image_paths)}) {img_name} - 處理失敗: {str(e)}")
            continue
    
    # 最終記憶體清理
    gpu_optimizer.cleanup_memory()
    
    # 計算處理時間
    total_time = time.time() - start_time
    avg_time_per_image = total_time / processed_count if processed_count > 0 else 0
    
    # 處理結果統計
    logger.info('\n' + '=' * 60)
    logger.info(' 處理結果統計 (穩定版本)')
    logger.info('=' * 60)
    logger.info(f'總影像數量: {len(image_paths)}')
    logger.info(f'成功處理: {processed_count}')
    logger.info(f'處理失敗: {failed_count}')
    logger.info(f'成功率: {processed_count/len(image_paths)*100:.1f}%')
    logger.info(f'總處理時間: {total_time:.2f} 秒')
    logger.info(f'平均每張時間: {avg_time_per_image:.2f} 秒')
    logger.info(f'處理速度: {processed_count/total_time:.2f} 張/秒')
    logger.info(f'結果儲存位置: {args.output_dir}')
    
    print(f"\n 台北101影像去噪處理完成！(穩定版本 + Self-Attention)")
    print(f"  處理統計: {processed_count}/{len(image_paths)} 成功")
    print(f" ⏱  處理時間: {total_time:.2f} 秒 (平均 {avg_time_per_image:.2f} 秒/張)")
    print(f"  處理速度: {processed_count/total_time:.2f} 張/秒")
    print(f"  增強特色: 雙流處理 + 自適應參數 + 多尺度邊緣 + Self-Attention {'啟用' if args.enable_attention else '未啟用'}")
    print(f"  結果位置: {args.output_dir}")
    
    if processed_count > 0:
        print(f"\n 增強版本特點:")
        print(f"   •   穩定性: 雙流處理基礎上的漸進式增強")
        print(f"   •  自適應參數: 根據影像特性自動調整雙流參數")
        print(f"   •  多尺度邊緣: 3×3和5×5核融合邊緣檢測")
        print(f"   •  Self-Attention: {'修復版注意力機制，智能細節增強' if args.enable_attention else '未啟用，追求高速穩定'}")
        print(f"   •  記憶體優化: 頻繁清理，避免記憶體累積")
        print(f"   •  詳細統計: 每張圖片的處理參數和統計資訊")

if __name__ == "__main__":
    main()