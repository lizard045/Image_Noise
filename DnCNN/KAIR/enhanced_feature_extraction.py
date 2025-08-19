#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版特徵提取模組
提供多層次、多尺度、多方向的特徵分析
專為台北101影像去噪優化
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction import image
from scipy import ndimage
import pywt

class AdvancedFeatureExtractor:
    """高級特徵提取器 - 專注於初期特徵分析"""
    
    def __init__(self):
        self.feature_cache = {}
        
    def extract_multiscale_features(self, image, scales=[1, 2, 4, 8]):
        """
        多尺度特徵提取 - 比原始的2尺度更豐富
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.astype(np.float32)
        
        features = {}
        
        for scale in scales:
            # 高斯金字塔下採樣
            if scale > 1:
                scaled_img = cv2.pyrDown(gray) if scale == 2 else \
                           cv2.resize(gray, (gray.shape[1]//scale, gray.shape[0]//scale))
            else:
                scaled_img = gray
            
            # 多方向邊緣檢測 (比單一Laplacian強)
            edges = self._compute_directional_edges(scaled_img)
            
            # 結構張量特徵
            structure = self._compute_structure_tensor(scaled_img)
            
            # 局部二值模式 (紋理特徵)
            lbp = self._compute_lbp_features(scaled_img)
            
            features[f'scale_{scale}'] = {
                'edges': edges,
                'structure': structure,
                'texture': lbp,
                'complexity': np.std(scaled_img)  # 影像複雜度指標
            }
        
        return features
    
    def _compute_directional_edges(self, img):
        """計算多方向邊緣特徵 - 替代單一Laplacian"""
        # Sobel各方向梯度
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        # 梯度幅值和方向
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Laplacian (二階導數)
        laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
        
        # Canny邊緣 (自適應閾值)
        canny_low = max(50, np.percentile(magnitude, 50))
        canny_high = min(200, np.percentile(magnitude, 85))
        canny = cv2.Canny(img.astype(np.uint8), int(canny_low), int(canny_high))
        
        return {
            'magnitude': magnitude,
            'direction': direction,
            'laplacian': laplacian,
            'canny': canny.astype(np.float32),
            'gradient_coherence': self._compute_gradient_coherence(grad_x, grad_y)
        }
    
    def _compute_structure_tensor(self, img):
        """計算結構張量 - 局部結構特徵"""
        # 圖像梯度
        Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        # 結構張量元素
        Ixx = cv2.GaussianBlur(Ix * Ix, (3, 3), 1)
        Iyy = cv2.GaussianBlur(Iy * Iy, (3, 3), 1)  
        Ixy = cv2.GaussianBlur(Ix * Iy, (3, 3), 1)
        
        # 本徵值計算
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy**2
        
        # 避免除零
        discriminant = np.maximum(0, trace**2 - 4*det)
        lambda1 = (trace + np.sqrt(discriminant)) / 2
        lambda2 = (trace - np.sqrt(discriminant)) / 2
        
        # 結構特徵
        coherence = np.divide(lambda1 - lambda2, lambda1 + lambda2 + 1e-8)  # 線性度
        energy = lambda1 + lambda2  # 能量
        
        return {
            'coherence': coherence,
            'energy': energy,
            'anisotropy': lambda1 / (lambda2 + 1e-8),  # 各向異性度
            'corner_strength': det - 0.04 * trace**2  # Harris角點強度
        }
    
    def _compute_lbp_features(self, img, radius=3, n_points=24):
        """計算局部二值模式 - 紋理特徵"""
        from skimage.feature import local_binary_pattern
        
        try:
            lbp = local_binary_pattern(img, n_points, radius, method='uniform')
            
            # LBP統計特徵
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points+2, 
                                     range=(0, n_points+2), density=True)
            
            # 紋理複雜度指標
            texture_complexity = np.sum(lbp_hist * np.arange(len(lbp_hist)))
            uniformity = np.sum(lbp_hist**2)  # 均勻性
            
            return {
                'lbp_map': lbp,
                'histogram': lbp_hist,
                'complexity': texture_complexity,
                'uniformity': uniformity
            }
        except ImportError:
            # 簡化版LBP實現
            return self._simple_lbp(img)
    
    def _simple_lbp(self, img):
        """簡化版LBP實現 (不依賴scikit-image)"""
        h, w = img.shape
        lbp = np.zeros_like(img)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = img[i, j]
                code = 0
                for k, (di, dj) in enumerate([(-1,-1), (-1,0), (-1,1), (0,1), 
                                            (1,1), (1,0), (1,-1), (0,-1)]):
                    if img[i+di, j+dj] >= center:
                        code |= (1 << k)
                lbp[i, j] = code
        
        return {'lbp_map': lbp, 'complexity': np.std(lbp)}
    
    def _compute_gradient_coherence(self, grad_x, grad_y):
        """計算梯度一致性 - 評估邊緣連續性"""
        # 梯度方向的局部一致性
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 避免除零
        safe_magnitude = magnitude + 1e-8
        norm_grad_x = grad_x / safe_magnitude
        norm_grad_y = grad_y / safe_magnitude
        
        # 計算局部窗口內的梯度一致性
        kernel = np.ones((3, 3)) / 9
        mean_grad_x = cv2.filter2D(norm_grad_x, -1, kernel)
        mean_grad_y = cv2.filter2D(norm_grad_y, -1, kernel)
        
        coherence = mean_grad_x**2 + mean_grad_y**2
        return coherence
    
    def compute_adaptive_weights(self, features, base_noise_level):
        """
        根據特徵自適應計算融合權重
        替代固定的tau=25%, M=85%閾值
        """
        # 從多尺度特徵中提取關鍵指標
        main_scale = features['scale_1']
        edge_strength = main_scale['edges']['magnitude']
        structure_energy = main_scale['structure']['energy']
        texture_complexity = main_scale['texture']['complexity']
        
        # 自適應閾值計算
        edge_percentiles = np.percentile(edge_strength, [10, 25, 50, 75, 90])
        
        # 根據影像特性動態調整閾值
        if texture_complexity > 100:  # 高紋理複雜度
            tau_ratio, M_ratio = 0.15, 0.9  # 更敏感的閾值
        elif texture_complexity < 50:  # 低紋理複雜度  
            tau_ratio, M_ratio = 0.35, 0.75  # 更寬鬆的閾值
        else:
            tau_ratio, M_ratio = 0.25, 0.85  # 標準閾值
        
        tau = edge_percentiles[int(tau_ratio * 4)]
        M = edge_percentiles[int(M_ratio * 4)]
        
        # 計算自適應權重
        edge_weight = np.clip((edge_strength - tau) / (M - tau + 1e-8), 0, 1)
        
        # 結合結構信息進行權重調整
        structure_modifier = np.tanh(structure_energy / (np.mean(structure_energy) + 1e-8))
        final_weight = edge_weight * 0.7 + structure_modifier * 0.3
        
        # 高斯平滑 (自適應核大小)
        blur_kernel = 3 if base_noise_level < 5 else 5
        final_weight = cv2.GaussianBlur(final_weight, (blur_kernel, blur_kernel), 0.8)
        
        return final_weight, {
            'tau': tau,
            'M': M,
            'texture_complexity': texture_complexity,
            'adaptive_ratios': (tau_ratio, M_ratio)
        }
    
    def extract_wavelet_features(self, image, wavelet='db4', levels=3):
        """小波特徵提取 - 頻域特徵分析"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 多層小波分解
        coeffs = pywt.wavedec2(gray, wavelet, levels=levels)
        
        features = {}
        features['approximation'] = coeffs[0]  # 低頻近似
        
        # 細節係數 (高頻)
        for i, (cH, cV, cD) in enumerate(coeffs[1:], 1):
            features[f'detail_h_level_{i}'] = cH  # 水平細節
            features[f'detail_v_level_{i}'] = cV  # 垂直細節  
            features[f'detail_d_level_{i}'] = cD  # 對角細節
            
            # 細節能量統計
            features[f'energy_level_{i}'] = np.sum(cH**2 + cV**2 + cD**2)
        
        return features


class EnhancedAttentionTuner:
    """Attention參數調優器"""
    
    def __init__(self, attention_module):
        self.attention_module = attention_module
        self.best_params = {}
        self.performance_history = []
        
    def detect_saddle_point(self, param_name, param_range, test_images, 
                          metric_func, tolerance=1e-4):
        """
        檢測參數鞍點 - 找到性能穩定點
        """
        print(f"    檢測 {param_name} 的鞍點...")
        
        param_values = np.linspace(param_range[0], param_range[1], 20)
        performance_scores = []
        
        original_param = self._get_param(param_name)
        
        for param_val in param_values:
            self._set_param(param_name, param_val)
            
            # 在測試圖像上評估性能
            scores = []
            for img in test_images[:3]:  # 使用前3張圖片快速測試
                with torch.no_grad():
                    score = metric_func(img, self.attention_module)
                scores.append(score)
            
            avg_score = np.mean(scores)
            performance_scores.append(avg_score)
            
        # 恢復原始參數
        self._set_param(param_name, original_param)
        
        # 尋找鞍點 (性能梯度接近零的點)
        gradients = np.gradient(performance_scores)
        saddle_idx = np.argmin(np.abs(gradients))
        saddle_point = param_values[saddle_idx]
        
        print(f"      鞍點檢測: {param_name} = {saddle_point:.4f}")
        print(f"      性能範圍: {min(performance_scores):.4f} - {max(performance_scores):.4f}")
        
        return saddle_point, performance_scores[saddle_idx]
    
    def grid_search_optimal(self, param_grid, test_images, metric_func):
        """
        網格搜索最優參數組合
        """
        print(f"    開始網格搜索最優參數...")
        best_score = float('-inf')
        best_params = {}
        
        # 生成參數組合
        param_combinations = self._generate_param_combinations(param_grid)
        total_combinations = len(param_combinations)
        
        for i, params in enumerate(param_combinations):
            if i % 10 == 0:
                print(f"      進度: {i+1}/{total_combinations}")
            
            # 設置參數
            for param_name, param_value in params.items():
                self._set_param(param_name, param_value)
            
            # 評估性能
            scores = []
            for img in test_images:
                with torch.no_grad():
                    score = metric_func(img, self.attention_module)
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params.copy()
            
            # 記錄性能歷史
            self.performance_history.append({
                'params': params.copy(),
                'score': avg_score
            })
        
        print(f"      最優參數: {best_params}")
        print(f"      最優得分: {best_score:.4f}")
        
        return best_params, best_score
    
    def adaptive_parameter_adjustment(self, current_performance, target_performance):
        """
        自適應參數調整 - 根據性能反饋動態調整
        """
        performance_gap = target_performance - current_performance
        
        if performance_gap > 0.1:  # 性能差距較大
            # 增加attention強度
            if hasattr(self.attention_module, 'gamma'):
                current_gamma = self.attention_module.gamma.item()
                new_gamma = min(1.0, current_gamma * 1.2)
                self.attention_module.gamma.data = torch.tensor(new_gamma)
                
            # 減少dropout
            if hasattr(self.attention_module, 'dropout'):
                current_dropout = self.attention_module.dropout.p
                new_dropout = max(0.05, current_dropout * 0.9)
                self.attention_module.dropout.p = new_dropout
                
        elif performance_gap < -0.05:  # 可能過擬合
            # 減少attention強度
            if hasattr(self.attention_module, 'gamma'):
                current_gamma = self.attention_module.gamma.item()
                new_gamma = max(0.1, current_gamma * 0.9)
                self.attention_module.gamma.data = torch.tensor(new_gamma)
                
            # 增加dropout
            if hasattr(self.attention_module, 'dropout'):
                current_dropout = self.attention_module.dropout.p
                new_dropout = min(0.3, current_dropout * 1.1)
                self.attention_module.dropout.p = new_dropout
    
    def _get_param(self, param_name):
        """獲取參數值"""
        if param_name == 'gamma':
            return self.attention_module.gamma.item()
        elif param_name == 'dropout':
            return self.attention_module.dropout.p
        # 可擴展其他參數
        
    def _set_param(self, param_name, value):
        """設置參數值"""
        if param_name == 'gamma':
            self.attention_module.gamma.data = torch.tensor(float(value))
        elif param_name == 'dropout':
            self.attention_module.dropout.p = float(value)
        # 可擴展其他參數
    
    def _generate_param_combinations(self, param_grid):
        """生成參數組合"""
        import itertools
        
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
            
        return combinations


def enhanced_feature_based_fusion(o_low, o_high, original_img, base_noise_level):
    """
    基於增強特徵提取的融合方法
    """
    print(f"    啟用增強版特徵融合...")
    
    # 初始化特徵提取器
    extractor = AdvancedFeatureExtractor()
    
    # 提取多尺度特徵
    features = extractor.extract_multiscale_features(original_img)
    
    # 計算自適應權重
    adaptive_weights, weight_info = extractor.compute_adaptive_weights(features, base_noise_level)
    
    # 擴展權重到彩色通道
    if len(o_low.shape) == 3:
        adaptive_weights = adaptive_weights[..., np.newaxis]
    
    # 特徵引導融合
    fused_result = adaptive_weights * o_low.astype(np.float32) + \
                  (1 - adaptive_weights) * o_high.astype(np.float32)
    fused_result = np.clip(fused_result, 0, 255).astype(np.uint8)
    
    print(f"      特徵統計: τ={weight_info['tau']:.2f}, M={weight_info['M']:.2f}")
    print(f"      紋理複雜度: {weight_info['texture_complexity']:.1f}")
    print(f"      自適應比例: {weight_info['adaptive_ratios']}")
    
    return fused_result, adaptive_weights


# 使用示例函數
def attention_performance_metric(image_tensor, attention_module):
    """
    Attention性能評估指標
    """
    # 這裡可以實現具體的性能評估
    # 例如：細節保留度、噪聲抑制度、計算效率等
    
    # 簡化示例：基於輸出方差的穩定性指標
    with torch.no_grad():
        output = attention_module(image_tensor)
        stability = 1.0 / (torch.var(output).item() + 1e-8)
        
    return stability
