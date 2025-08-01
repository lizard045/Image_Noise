#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU性能監控腳本
實時監控訓練過程中的GPU利用率和記憶體使用
"""

import time
import subprocess
import psutil
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import matplotlib.font_manager as fm

class GPUMonitor:
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.gpu_usage_history = []
        self.memory_usage_history = []
        self.timestamps = []
        self.monitoring = False
        
    def get_gpu_info(self):
        """獲取GPU使用情況"""
        try:
            # 使用nvidia-smi獲取GPU信息
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'gpu_utilization': float(values[0]),
                    'memory_used_mb': float(values[1]),
                    'memory_total_mb': float(values[2]),
                    'temperature': float(values[3])
                }
        except Exception as e:
            print(f"獲取GPU信息失敗: {e}")
            
        # 備用方案：使用PyTorch
        if torch.cuda.is_available():
            return {
                'gpu_utilization': 0,  # PyTorch無法直接獲取利用率
                'memory_used_mb': torch.cuda.memory_allocated() / 1024**2,
                'memory_total_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2,
                'temperature': 0
            }
        
        return None
    
    def get_system_info(self):
        """獲取系統信息"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {}
        }
    
    def log_performance(self):
        """記錄性能數據"""
        gpu_info = self.get_gpu_info()
        system_info = self.get_system_info()
        
        if gpu_info:
            timestamp = datetime.now()
            self.timestamps.append(timestamp)
            self.gpu_usage_history.append(gpu_info['gpu_utilization'])
            self.memory_usage_history.append(gpu_info['memory_used_mb'] / gpu_info['memory_total_mb'] * 100)
            
            # 實時輸出
            print(f"[{timestamp.strftime('%H:%M:%S')}] "
                  f"GPU: {gpu_info['gpu_utilization']:5.1f}% | "
                  f"VRAM: {gpu_info['memory_used_mb']/1024:.1f}GB/"
                  f"{gpu_info['memory_total_mb']/1024:.1f}GB "
                  f"({gpu_info['memory_used_mb']/gpu_info['memory_total_mb']*100:.1f}%) | "
                  f"Temp: {gpu_info['temperature']}°C | "
                  f"CPU: {system_info['cpu_percent']:5.1f}% | "
                  f"RAM: {system_info['memory_percent']:5.1f}%")
            
            # 保存到文件
            log_data = {
                'timestamp': timestamp.isoformat(),
                'gpu': gpu_info,
                'system': system_info
            }
            
            with open('gpu_performance.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
    
    def start_monitoring(self):
        """開始監控"""
        self.monitoring = True
        print("🚀 開始GPU性能監控...")
        print("=" * 80)
        
        try:
            while self.monitoring:
                self.log_performance()
                time.sleep(self.log_interval)
        except KeyboardInterrupt:
            print("\n⏹️  監控已停止")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        self.generate_report()
    
    def generate_report(self):
        """生成性能報告"""
        if not self.gpu_usage_history:
            print("沒有性能數據可生成報告")
            return
        
        # 計算統計數據
        avg_gpu_usage = sum(self.gpu_usage_history) / len(self.gpu_usage_history)
        max_gpu_usage = max(self.gpu_usage_history)
        avg_memory_usage = sum(self.memory_usage_history) / len(self.memory_usage_history)
        max_memory_usage = max(self.memory_usage_history)
        
        print("\n📊 性能統計報告:")
        print("=" * 50)
        print(f"監控時長: {len(self.gpu_usage_history) * self.log_interval / 60:.1f} 分鐘")
        print(f"平均GPU利用率: {avg_gpu_usage:.1f}%")
        print(f"最高GPU利用率: {max_gpu_usage:.1f}%")
        print(f"平均記憶體使用: {avg_memory_usage:.1f}%")
        print(f"最高記憶體使用: {max_memory_usage:.1f}%")
        
        # 生成圖表
        self.plot_performance()
        
        # 優化建議
        self.provide_recommendations(avg_gpu_usage, max_gpu_usage, avg_memory_usage)
    
    def plot_performance(self):
        """繪製性能圖表"""
        if not self.gpu_usage_history:
            return
        
        # 設定中文字體
        font_path = r'C:\Users\brogent\Desktop\Image_Noise\Mamelon.otf'
        if os.path.exists(font_path):
            chinese_font = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
        else:
            chinese_font = None
            print(f"警告: 找不到字體檔案 {font_path}")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # GPU利用率圖
        ax1.plot(self.gpu_usage_history, 'b-', linewidth=2, label='GPU利用率')
        ax1.set_ylabel('GPU利用率 (%)', fontproperties=chinese_font)
        ax1.set_title('GPU性能監控', fontproperties=chinese_font)
        ax1.grid(True, alpha=0.3)
        if chinese_font:
            ax1.legend(prop=chinese_font)
        else:
            ax1.legend()
        ax1.set_ylim(0, 100)
        
        # 記憶體使用圖
        ax2.plot(self.memory_usage_history, 'r-', linewidth=2, label='記憶體使用')
        ax2.set_ylabel('記憶體使用 (%)', fontproperties=chinese_font)
        ax2.set_xlabel('時間點', fontproperties=chinese_font)
        ax2.grid(True, alpha=0.3)
        if chinese_font:
            ax2.legend(prop=chinese_font)
        else:
            ax2.legend()
        ax2.set_ylim(0, 100)
        
        # 設定刻度標籤字體
        if chinese_font:
            for label in ax1.get_xticklabels() + ax1.get_yticklabels():
                label.set_fontproperties(chinese_font)
            for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                label.set_fontproperties(chinese_font)
        
        plt.tight_layout()
        plt.savefig('gpu_performance_report.png', dpi=300, bbox_inches='tight')
        print("📈 性能圖表已保存至: gpu_performance_report.png")
    
    def provide_recommendations(self, avg_gpu, max_gpu, avg_memory):
        """提供優化建議"""
        print("\n💡 優化建議:")
        print("=" * 50)
        
        if avg_gpu < 70:
            print("🔸 GPU利用率偏低，建議:")
            print("   - 增加批次大小 (batch_size)")
            print("   - 減少數據載入工作線程數 (num_workers)")
            print("   - 檢查是否有CPU瓶頸")
        
        if max_gpu > 95:
            print("🔸 GPU利用率過高，可能出現瓶頸:")
            print("   - 考慮降低批次大小")
            print("   - 啟用混合精度訓練 (AMP)")
        
        if avg_memory > 85:
            print("🔸 記憶體使用率高，建議:")
            print("   - 降低批次大小")
            print("   - 啟用混合精度訓練")
            print("   - 增加記憶體清理頻率")
        
        if avg_gpu > 80 and avg_memory < 60:
            print("✅ GPU配置良好，利用率高且記憶體充足")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU性能監控工具')
    parser.add_argument('--interval', type=int, default=10, help='監控間隔(秒)')
    parser.add_argument('--duration', type=int, default=0, help='監控時長(秒)，0表示持續監控')
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(log_interval=args.interval)
    
    if args.duration > 0:
        print(f"將監控 {args.duration} 秒...")
        import threading
        timer = threading.Timer(args.duration, monitor.stop_monitoring)
        timer.start()
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n用戶中斷監控")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()