#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
優化配置文件
為不同硬體環境提供最佳化設定

使用方法:
from optimization_config import get_optimal_config
config = get_optimal_config()
"""

import torch
import psutil
import gc
import os

class OptimizationConfig:
    """優化配置類"""
    
    def __init__(self):
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpu_memory_gb = 0
        self.cpu_cores = psutil.cpu_count()
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if self.device_type == 'cuda':
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        self.config = self._determine_optimal_config()
    
    def _determine_optimal_config(self):
        """根據硬體配置決定最佳設定"""
        config = {
            'batch_size': 1,
            'num_workers': 2,
            'use_amp': False,
            'enable_jit_compile': False,
            'memory_fraction': 0.8,
            'enable_detail_enhancement': True,
            'enable_frequency_preservation': True,
            'prefetch_factor': 2,
            'pin_memory': False
        }
        
        if self.device_type == 'cuda':
            # GPU優化設定
            if self.gpu_memory_gb >= 8:
                # 高端GPU (8GB+)
                config.update({
                    'batch_size': 8,
                    'num_workers': min(4, self.cpu_cores),
                    'use_amp': True,
                    'enable_jit_compile': True,
                    'memory_fraction': 0.9,
                    'prefetch_factor': 4,
                    'pin_memory': True
                })
            elif self.gpu_memory_gb >= 4:
                # 中端GPU (4-8GB)
                config.update({
                    'batch_size': 4,
                    'num_workers': min(3, self.cpu_cores),
                    'use_amp': True,
                    'enable_jit_compile': True,
                    'memory_fraction': 0.85,
                    'prefetch_factor': 3,
                    'pin_memory': True
                })
            else:
                # 低端GPU (<4GB)
                config.update({
                    'batch_size': 2,
                    'num_workers': 2,
                    'use_amp': True,
                    'enable_jit_compile': False,
                    'memory_fraction': 0.8,
                    'prefetch_factor': 2,
                    'pin_memory': False
                })
        else:
            # CPU優化設定
            if self.cpu_cores >= 8 and self.system_memory_gb >= 16:
                # 高端CPU
                config.update({
                    'batch_size': 4,
                    'num_workers': min(6, self.cpu_cores),
                    'use_amp': False,
                    'enable_jit_compile': True,
                    'prefetch_factor': 4
                })
            elif self.cpu_cores >= 4 and self.system_memory_gb >= 8:
                # 中端CPU
                config.update({
                    'batch_size': 2,
                    'num_workers': min(4, self.cpu_cores),
                    'use_amp': False,
                    'enable_jit_compile': False,
                    'prefetch_factor': 2
                })
            else:
                # 低端CPU
                config.update({
                    'batch_size': 1,
                    'num_workers': 2,
                    'use_amp': False,
                    'enable_jit_compile': False,
                    'prefetch_factor': 1
                })
        
        return config
    
    def get_config(self):
        """獲取配置"""
        return self.config
    
    def print_config(self):
        """打印配置信息"""
        print("=" * 50)
        print("硬體信息:")
        print(f"  設備類型: {self.device_type.upper()}")
        if self.device_type == 'cuda':
            print(f"  GPU記憶體: {self.gpu_memory_gb:.1f}GB")
            print(f"  GPU名稱: {torch.cuda.get_device_name(0)}")
        print(f"  CPU核心數: {self.cpu_cores}")
        print(f"  系統記憶體: {self.system_memory_gb:.1f}GB")
        
        print("\n最佳化配置:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print("=" * 50)

def get_optimal_config():
    """獲取最佳配置"""
    optimizer = OptimizationConfig()
    return optimizer.get_config()

def print_system_info():
    """打印系統信息"""
    optimizer = OptimizationConfig()
    optimizer.print_config()

# 預設配置選項
PERFORMANCE_PROFILES = {
    'ultra_fast': {
        'enable_detail_enhancement': False,
        'enable_frequency_preservation': False,
        'batch_size_multiplier': 2.0,
        'description': '最快速度模式 - 犧牲部分質量換取最高速度'
    },
    'balanced': {
        'enable_detail_enhancement': True,
        'enable_frequency_preservation': True,
        'batch_size_multiplier': 1.0,
        'description': '平衡模式 - 速度與質量的最佳平衡'
    },
    'high_quality': {
        'enable_detail_enhancement': True,
        'enable_frequency_preservation': True,
        'batch_size_multiplier': 0.7,
        'description': '高質量模式 - 最佳去噪效果'
    },
    'memory_efficient': {
        'enable_detail_enhancement': True,
        'enable_frequency_preservation': False,
        'batch_size_multiplier': 0.5,
        'description': '記憶體節約模式 - 適用於低記憶體設備'
    }
}

def apply_performance_profile(config, profile_name='balanced'):
    """應用性能配置檔案"""
    if profile_name not in PERFORMANCE_PROFILES:
        print(f"警告: 未知的配置檔案 '{profile_name}', 使用預設配置")
        profile_name = 'balanced'
    
    profile = PERFORMANCE_PROFILES[profile_name]
    
    # 應用配置
    config['enable_detail_enhancement'] = profile['enable_detail_enhancement']
    config['enable_frequency_preservation'] = profile['enable_frequency_preservation']
    config['batch_size'] = max(1, int(config['batch_size'] * profile['batch_size_multiplier']))
    
    print(f"已應用性能配置檔案: {profile_name}")
    print(f"描述: {profile['description']}")
    
    return config

def optimize_memory_usage():
    """記憶體優化建議"""
    suggestions = []
    
    # 檢查可用記憶體
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    if available_gb < 4:
        suggestions.append("建議關閉其他應用程式以釋放記憶體")
        suggestions.append("考慮使用 'memory_efficient' 配置檔案")
    
    if torch.cuda.is_available():
        # GPU記憶體檢查
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 4:
            suggestions.append("GPU記憶體較小，建議降低批次大小")
            suggestions.append("建議啟用混合精度 (use_amp=True)")
    
    # 環境變數建議
    env_suggestions = {
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        'CUDA_VISIBLE_DEVICES': '0',  # 如果有多GPU，指定使用的GPU
    }
    
    return suggestions, env_suggestions

def benchmark_configuration(config, test_image_path=None):
    """基準測試配置效能"""
    import time
    import numpy as np
    
    print("開始基準測試...")
    
    # 模擬測試
    test_times = []
    memory_usage = []
    
    for i in range(3):  # 測試3次
        start_time = time.time()
        
        # 模擬處理流程
        if torch.cuda.is_available():
            # GPU記憶體使用測試
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
            
            # 創建測試張量
            test_tensor = torch.randn(config['batch_size'], 3, 512, 512).cuda()
            torch.cuda.synchronize()
            
            memory_after = torch.cuda.memory_allocated()
            memory_usage.append((memory_after - memory_before) / (1024**2))  # MB
            
            del test_tensor
            torch.cuda.empty_cache()
        
        end_time = time.time()
        test_times.append(end_time - start_time)
    
    avg_time = np.mean(test_times)
    avg_memory = np.mean(memory_usage) if memory_usage else 0
    
    print(f"基準測試結果:")
    print(f"  平均處理時間: {avg_time:.3f}秒")
    if avg_memory > 0:
        print(f"  平均記憶體使用: {avg_memory:.1f}MB")
    
    return avg_time, avg_memory

if __name__ == "__main__":
    # 示例使用
    print("系統硬體信息和最佳化建議:")
    print_system_info()
    
    print("\n可用的性能配置檔案:")
    for name, profile in PERFORMANCE_PROFILES.items():
        print(f"  {name}: {profile['description']}")
    
    print("\n記憶體優化建議:")
    suggestions, env_vars = optimize_memory_usage()
    for suggestion in suggestions:
        print(f"  - {suggestion}")
    
    if env_vars:
        print("\n建議的環境變數:")
        for var, value in env_vars.items():
            print(f"  export {var}={value}")