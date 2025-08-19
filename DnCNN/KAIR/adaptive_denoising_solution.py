#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自適應去噪解決方案
專門解決台北101等建築圖像去噪效果不均勻的問題
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

def enhanced_adaptive_dual_stream_processing(model, img_L, device, base_noise_level, gpu_optimizer, enable_attention=False):
    """
    增強版自適應雙流處理 - 解決去噪效果不均勻問題
    """
    try:
        # 確保輸入格式正確
        if len(img_L.shape) == 2:
            img_L = img_L[:, :, np.newaxis] 
        elif len(img_L.shape) == 4:
            img_L = img_L[0]
        
        # 轉換為tensor
        img_tensor = util.uint2tensor4(img_L).to(device)
        
        # 自適應區域分析
        if len(img_L.shape) == 3:
            gray = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_L.squeeze()
        
        print(f"    啟用增強版自適應區域分析...")
        
        # 1. 區域特性分析
        region_analysis = analyze_image_regions(gray)
        
        # 2. 基於區域特性的自適應參數計算
        adaptive_params = compute_region_adaptive_parameters(region_analysis, base_noise_level)
        
        print(f"    檢測到 {len(adaptive_params['regions'])} 個不同特性區域")
        for i, region_info in enumerate(adaptive_params['regions']):
            print(f"      區域{i+1}: {region_info['name']}, 比例={region_info['ratio']:.1f}%, σ_low={region_info['sigma_low']:.1f}, σ_high={region_info['sigma_high']:.1f}")
        
        # 3. 多區域雙流處理
        results = []
        for region_info in adaptive_params['regions']:
            # Pass-1: 低噪聲等級處理
            noise_level_low = region_info['sigma_low'] / 255.0
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
            noise_level_high = region_info['sigma_high'] / 255.0
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
            
            results.append({
                'o_low': o_low_np,
                'o_high': o_high_np,
                'region_info': region_info
            })
        
        # 4. 區域自適應融合
        print(f"    執行區域自適應融合...")
        final_result = region_adaptive_fusion(results, gray, adaptive_params)
        
        # 5. Self-Attention增強 (如果啟用且可用)
        if enable_attention and gpu_optimizer.attention_module is not None:
            try:
                print(f"    應用區域感知Self-Attention增強...")
                final_result = apply_region_aware_attention(final_result, adaptive_params, gpu_optimizer, device)
                print(f"    區域感知Self-Attention處理成功")
            except Exception as e:
                print(f"    Self-Attention處理失敗: {str(e)[:50]}..., 使用無attention結果")
        
        # 清理記憶體
        del img_tensor, img_input_low, img_input_high, o_low, o_high
        gpu_optimizer.cleanup_memory()
        
        return final_result
        
    except Exception as e:
        print(f"    增強版自適應處理失敗: {str(e)}, 使用標準處理")
        return stable_dual_stream_processing(model, img_L, device, base_noise_level, gpu_optimizer, enable_attention)

def analyze_image_regions(gray):
    """
    分析圖像區域特性
    """
    h, w = gray.shape
    
    # 1. 紋理複雜度分析
    # 局部標準差
    kernel = np.ones((9, 9), np.float32) / 81
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_std = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
    texture_complexity = np.sqrt(local_std)
    
    # 2. 邊緣強度分析
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    
    # 局部邊緣密度
    kernel_edge = np.ones((15, 15), np.float32) / 225
    edge_density = cv2.filter2D(edge_strength.astype(np.float32), -1, kernel_edge)
    
    # 3. 亮度分析
    brightness = gray.astype(np.float32)
    
    # 局部亮度變化
    brightness_variance = cv2.filter2D((brightness - cv2.GaussianBlur(brightness, (15, 15), 5))**2, -1, kernel)
    
    # 4. 區域分類
    # 使用K-means或閾值方法分類區域
    texture_low = np.percentile(texture_complexity, 30)
    texture_high = np.percentile(texture_complexity, 70)
    edge_low = np.percentile(edge_density, 30)
    edge_high = np.percentile(edge_density, 70)
    brightness_low = np.percentile(brightness, 30)
    brightness_high = np.percentile(brightness, 70)
    
    # 定義5種區域類型
    region_map = np.zeros((h, w), dtype=np.uint8)
    
    # 0: 平滑暗區域 (天空暗部、陰影)
    smooth_dark = (texture_complexity < texture_low) & (edge_density < edge_low) & (brightness < brightness_low)
    region_map[smooth_dark] = 0
    
    # 1: 平滑亮區域 (天空亮部、大面積牆面)
    smooth_bright = (texture_complexity < texture_low) & (edge_density < edge_low) & (brightness >= brightness_low)
    region_map[smooth_bright] = 1
    
    # 2: 細節區域 (建築細節、窗戶、裝飾)
    detail_region = (texture_complexity > texture_high) & (edge_density > edge_high)
    region_map[detail_region] = 2
    
    # 3: 邊緣區域 (建築輪廓、天際線)
    edge_region = (edge_density > edge_high) & (texture_complexity <= texture_high)
    region_map[edge_region] = 3
    
    # 4: 中等複雜度區域 (其他區域)
    medium_region = ~(smooth_dark | smooth_bright | detail_region | edge_region)
    region_map[medium_region] = 4
    
    return {
        'region_map': region_map,
        'texture_complexity': texture_complexity,
        'edge_density': edge_density,
        'brightness': brightness,
        'brightness_variance': brightness_variance,
        'thresholds': {
            'texture_low': texture_low,
            'texture_high': texture_high,
            'edge_low': edge_low,
            'edge_high': edge_high,
            'brightness_low': brightness_low,
            'brightness_high': brightness_high
        }
    }

def compute_region_adaptive_parameters(region_analysis, base_noise_level):
    """
    根據區域分析計算自適應參數
    """
    region_map = region_analysis['region_map']
    total_pixels = region_map.size
    
    # 定義每種區域的特性和參數
    region_configs = [
        {  # 0: 平滑暗區域
            'name': '平滑暗區域(天空暗部)',
            'sigma_low_ratio': 0.15,   # 更保守的去噪
            'sigma_high_ratio': 0.8,   # 避免過度平滑
            'fusion_weight': 0.8,      # 偏向低噪聲處理
        },
        {  # 1: 平滑亮區域  
            'name': '平滑亮區域(天空亮部)',
            'sigma_low_ratio': 0.2,
            'sigma_high_ratio': 1.0,
            'fusion_weight': 0.75,
        },
        {  # 2: 細節區域
            'name': '細節區域(建築細節)',
            'sigma_low_ratio': 0.4,    # 更激進的細節保護
            'sigma_high_ratio': 2.2,   # 更強的噪聲抑制對比
            'fusion_weight': 0.6,      # 平衡細節和去噪
        },
        {  # 3: 邊緣區域
            'name': '邊緣區域(建築輪廓)',
            'sigma_low_ratio': 0.35,   # 邊緣保護
            'sigma_high_ratio': 1.8,
            'fusion_weight': 0.65,
        },
        {  # 4: 中等複雜度區域
            'name': '中等區域(混合紋理)',
            'sigma_low_ratio': 0.3,
            'sigma_high_ratio': 1.5,
            'fusion_weight': 0.7,
        }
    ]
    
    # 計算每個區域的像素比例和參數
    regions = []
    for region_id, config in enumerate(region_configs):
        mask = region_map == region_id
        pixel_count = np.sum(mask)
        
        if pixel_count > 0:  # 只處理存在的區域
            ratio = pixel_count / total_pixels * 100
            
            region_info = {
                'id': region_id,
                'name': config['name'],
                'mask': mask,
                'ratio': ratio,
                'pixel_count': pixel_count,
                'sigma_low': base_noise_level * config['sigma_low_ratio'],
                'sigma_high': base_noise_level * config['sigma_high_ratio'],
                'fusion_weight': config['fusion_weight']
            }
            regions.append(region_info)
    
    return {
        'regions': regions,
        'region_analysis': region_analysis
    }

def region_adaptive_fusion(results, gray, adaptive_params):
    """
    區域自適應融合
    """
    h, w = gray.shape
    
    if len(results[0]['o_low'].shape) == 3:
        final_result = np.zeros_like(results[0]['o_low'], dtype=np.float32)
    else:
        final_result = np.zeros((h, w, 1), dtype=np.float32)
    
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)
    
    for result in results:
        region_info = result['region_info']
        mask = region_info['mask']
        
        if np.sum(mask) == 0:  # 跳過空區域
            continue
        
        # 計算該區域的邊緣權重
        region_gray = gray.copy()
        region_gray[~mask] = 0  # 遮蔽其他區域
        
        # 自適應邊緣檢測
        edge_weight = compute_adaptive_edge_weights(region_gray, mask, region_info)
        
        # 擴展到彩色通道
        if len(final_result.shape) == 3 and final_result.shape[2] == 3:
            edge_weight_3d = edge_weight[..., np.newaxis]
        else:
            edge_weight_3d = edge_weight[..., np.newaxis]
        
        # 區域內雙流融合
        region_fused = edge_weight_3d * result['o_low'].astype(np.float32) + \
                      (1 - edge_weight_3d) * result['o_high'].astype(np.float32)
        
        # 使用區域掩碼加權累積
        region_mask_3d = mask[..., np.newaxis].astype(np.float32)
        final_result += region_fused * region_mask_3d
        weight_sum += region_mask_3d
    
    # 歸一化
    weight_sum = np.maximum(weight_sum, 1e-8)  # 避免除零
    final_result = final_result / weight_sum
    
    return np.clip(final_result, 0, 255).astype(np.uint8)

def compute_adaptive_edge_weights(gray, mask, region_info):
    """
    計算自適應邊緣權重
    """
    # 只在該區域內計算邊緣
    masked_gray = gray.copy()
    masked_gray[~mask] = np.mean(masked_gray[mask])  # 用區域均值填充
    
    # 多尺度邊緣檢測
    edge_maps = []
    kernel_sizes = [3, 5, 7] if region_info['name'].find('細節') != -1 else [3, 5]
    
    for ksize in kernel_sizes:
        edge = cv2.Laplacian(masked_gray.astype(np.float32), cv2.CV_32F, ksize=ksize)
        edge_abs = np.abs(edge)
        edge_maps.append(edge_abs)
    
    # 自適應權重融合邊緣
    if len(edge_maps) == 3:
        combined_edge = 0.5 * edge_maps[0] + 0.3 * edge_maps[1] + 0.2 * edge_maps[2]
    else:
        combined_edge = 0.7 * edge_maps[0] + 0.3 * edge_maps[1]
    
    # 只在該區域內的像素上計算百分位數
    region_edges = combined_edge[mask]
    
    if len(region_edges) > 0:
        # 根據區域類型自適應調整閾值
        if '細節' in region_info['name']:
            # 細節區域：更敏感的閾值
            tau = np.percentile(region_edges, 20)
            M = np.percentile(region_edges, 90)
        elif '邊緣' in region_info['name']:
            # 邊緣區域：中等敏感度
            tau = np.percentile(region_edges, 25)
            M = np.percentile(region_edges, 85)
        elif '平滑' in region_info['name']:
            # 平滑區域：較低敏感度
            tau = np.percentile(region_edges, 35)
            M = np.percentile(region_edges, 75)
        else:
            # 中等區域：標準閾值
            tau = np.percentile(region_edges, 30)
            M = np.percentile(region_edges, 85)
    else:
        # 後備值
        tau = np.mean(combined_edge) * 0.5
        M = np.mean(combined_edge) * 2.0
    
    if M - tau < 1e-6:
        M = tau + 1e-6
    
    # 計算邊緣權重
    edge_weight = np.clip((combined_edge - tau) / (M - tau), 0, 1)
    
    # 區域自適應平滑
    if '細節' in region_info['name']:
        # 細節區域：較少平滑
        edge_weight = cv2.GaussianBlur(edge_weight, (3, 3), 0.3)
    elif '平滑' in region_info['name']:
        # 平滑區域：較多平滑
        edge_weight = cv2.GaussianBlur(edge_weight, (7, 7), 1.0)
    else:
        # 其他區域：中等平滑
        edge_weight = cv2.GaussianBlur(edge_weight, (5, 5), 0.5)
    
    # 只在該區域內有效
    edge_weight[~mask] = 0
    
    return edge_weight

def apply_region_aware_attention(image, adaptive_params, gpu_optimizer, device):
    """
    應用區域感知的Self-Attention
    """
    try:
        if gpu_optimizer.attention_module is None:
            return image
        
        # 轉換為tensor
        img_tensor = util.uint2tensor4(image).to(device)
        
        # 為不同區域調整attention參數
        original_gamma = gpu_optimizer.attention_module.gamma.clone()
        
        # 計算區域平均權重
        total_pixels = 0
        weighted_gamma = 0
        
        for region_info in adaptive_params['regions']:
            pixel_count = region_info['pixel_count']
            total_pixels += pixel_count
            
            # 根據區域類型調整gamma
            if '細節' in region_info['name']:
                region_gamma = 0.7  # 細節區域用較高的attention
            elif '邊緣' in region_info['name']:
                region_gamma = 0.6
            elif '平滑' in region_info['name']:
                region_gamma = 0.3  # 平滑區域用較低的attention
            else:
                region_gamma = 0.5
            
            weighted_gamma += region_gamma * pixel_count
        
        # 設置加權平均的gamma
        optimal_gamma = weighted_gamma / total_pixels if total_pixels > 0 else 0.5
        gpu_optimizer.attention_module.gamma.data = torch.tensor(optimal_gamma)
        
        # 應用attention
        with torch.no_grad():
            try:
                from torch.amp import autocast
                with autocast('cuda', enabled=gpu_optimizer.use_amp):
                    enhanced_tensor = gpu_optimizer.attention_module(img_tensor)
            except ImportError:
                from torch.cuda.amp import autocast
                with autocast(enabled=gpu_optimizer.use_amp):
                    enhanced_tensor = gpu_optimizer.attention_module(img_tensor)
        
        # 恢復原始gamma
        gpu_optimizer.attention_module.gamma.data = original_gamma
        
        # 轉換回numpy
        enhanced_result = util.tensor2uint(enhanced_tensor)
        
        # 清理記憶體
        del img_tensor, enhanced_tensor
        
        return enhanced_result
        
    except Exception as e:
        print(f"      區域感知attention失敗: {str(e)[:50]}...")
        return image
