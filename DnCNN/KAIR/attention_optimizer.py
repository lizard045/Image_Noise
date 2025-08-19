#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Attention參數優化器
專為台北101去噪系統設計的智能調優工具
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
import time
import json
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import os

class AttentionParameterOptimizer:
    """Self-Attention參數智能優化器"""
    
    def __init__(self, attention_module, font_path="Mamelon.otf"):
        self.attention_module = attention_module
        self.optimization_history = []
        self.best_params = {}
        self.font_path = font_path
        
        # 設置中文字體
        try:
            if os.path.exists(font_path):
                # 清除matplotlib字體緩存
                font_manager.fontManager.ttflist = []
                font_manager.fontManager.afmlist = []
                font_manager.fontManager.addfont(font_path)
                
                # 獲取字體名稱
                font_name = font_manager.FontProperties(fname=font_path).get_name()
                self.font_prop = font_manager.FontProperties(fname=font_path)
                
                print(f"    成功載入字體: {font_path} (名稱: {font_name})")
            else:
                # 搜索字體文件
                possible_paths = [
                    font_path,
                    os.path.join(os.getcwd(), font_path),
                    os.path.join(os.path.dirname(__file__), font_path),
                    os.path.join("D:/Image_Noise", font_path)
                ]
                
                font_found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        font_manager.fontManager.addfont(path)
                        self.font_prop = font_manager.FontProperties(fname=path)
                        font_found = True
                        print(f"    字體找到並載入: {path}")
                        break
                
                if not font_found:
                    self.font_prop = None
                    print(f"    字體文件未找到，搜索路徑:")
                    for path in possible_paths:
                        print(f"      - {path}")
                        
        except Exception as e:
            self.font_prop = None
            print(f"    字體載入失敗: {str(e)}")
    
    def comprehensive_parameter_analysis(self, test_images, output_dir="attention_analysis"):
        """
        全面的參數分析 - 找到最佳配置
        """
        print(f" 開始Attention參數全面分析...")
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # 1. Gamma參數鞍點分析
        print(f"  1. Gamma參數鞍點分析")
        gamma_result = self.gamma_saddle_point_analysis(test_images)
        results['gamma'] = gamma_result
        
        # 2. Dropout最佳值搜索
        print(f"  2. Dropout最佳值搜索") 
        dropout_result = self.dropout_optimization(test_images)
        results['dropout'] = dropout_result
        
        # 3. 參數交互作用分析
        print(f"  3. 參數交互作用分析")
        interaction_result = self.parameter_interaction_analysis(test_images)
        results['interaction'] = interaction_result
        
        # 4. 自適應策略驗證
        print(f"  4. 自適應策略驗證")
        adaptive_result = self.validate_adaptive_strategy(test_images)
        results['adaptive'] = adaptive_result
        
        # 5. 生成可視化報告
        self.generate_visualization_report(results, output_dir)
        
        # 6. 保存分析結果
        self.save_analysis_results(results, output_dir)
        
        return results
    
    def gamma_saddle_point_analysis(self, test_images, gamma_range=(0.0, 1.0), n_points=50):
        """
        Gamma參數鞍點檢測 - 找到性能穩定的最佳點
        """
        gamma_values = np.linspace(gamma_range[0], gamma_range[1], n_points)
        performance_scores = []
        stability_scores = []
        
        original_gamma = self.attention_module.gamma.clone()
        
        print(f"    掃描gamma範圍: {gamma_range[0]:.2f} - {gamma_range[1]:.2f}")
        
        for i, gamma_val in enumerate(gamma_values):
            if i % 10 == 0:
                print(f"      進度: {i+1}/{n_points}")
            
            # 設置gamma值
            self.attention_module.gamma.data = torch.tensor(gamma_val)
            
            # 計算性能指標
            perf_score, stab_score = self._evaluate_attention_performance(test_images)
            performance_scores.append(perf_score)
            stability_scores.append(stab_score)
        
        # 恢復原始gamma
        self.attention_module.gamma.data = original_gamma
        
        # 鞍點檢測
        saddle_info = self._detect_saddle_point(gamma_values, performance_scores, stability_scores)
        
        print(f"      鞍點檢測結果:")
        print(f"        最佳Gamma: {saddle_info['optimal_gamma']:.4f}")
        print(f"        性能得分: {saddle_info['optimal_performance']:.4f}")
        print(f"        穩定性得分: {saddle_info['optimal_stability']:.4f}")
        
        return {
            'gamma_values': gamma_values,
            'performance_scores': performance_scores,
            'stability_scores': stability_scores,
            'saddle_info': saddle_info
        }
    
    def dropout_optimization(self, test_images, dropout_range=(0.05, 0.4), n_points=20):
        """
        Dropout參數優化 - 平衡過擬合和欠擬合
        """
        dropout_values = np.linspace(dropout_range[0], dropout_range[1], n_points)
        performance_scores = []
        overfitting_scores = []
        
        original_dropout = self.attention_module.dropout.p
        
        print(f"    掃描dropout範圍: {dropout_range[0]:.3f} - {dropout_range[1]:.3f}")
        
        for i, dropout_val in enumerate(dropout_values):
            print(f"      進度: {i+1}/{n_points}")
            
            # 設置dropout值
            self.attention_module.dropout.p = dropout_val
            
            # 評估性能和過擬合程度
            perf_score = self._evaluate_denoising_quality(test_images)
            overfit_score = self._evaluate_overfitting_risk(test_images)
            
            performance_scores.append(perf_score)
            overfitting_scores.append(overfit_score)
        
        # 恢復原始dropout
        self.attention_module.dropout.p = original_dropout
        
        # 找到最佳平衡點
        optimal_info = self._find_optimal_dropout(dropout_values, performance_scores, overfitting_scores)
        
        print(f"      最佳Dropout: {optimal_info['optimal_dropout']:.4f}")
        print(f"      平衡得分: {optimal_info['balance_score']:.4f}")
        
        return {
            'dropout_values': dropout_values,
            'performance_scores': performance_scores,
            'overfitting_scores': overfitting_scores,
            'optimal_info': optimal_info
        }
    
    def parameter_interaction_analysis(self, test_images):
        """
        參數交互作用分析 - 找到參數間的最佳組合
        """
        print(f"    分析Gamma-Dropout交互作用...")
        
        gamma_grid = np.linspace(0.1, 0.8, 8)
        dropout_grid = np.linspace(0.05, 0.3, 6)
        
        interaction_matrix = np.zeros((len(gamma_grid), len(dropout_grid)))
        best_combo = {'gamma': 0, 'dropout': 0, 'score': -1}
        
        original_gamma = self.attention_module.gamma.clone()
        original_dropout = self.attention_module.dropout.p
        
        total_combinations = len(gamma_grid) * len(dropout_grid)
        combination_count = 0
        
        for i, gamma in enumerate(gamma_grid):
            for j, dropout in enumerate(dropout_grid):
                combination_count += 1
                if combination_count % 5 == 0:
                    print(f"      進度: {combination_count}/{total_combinations}")
                
                # 設置參數組合
                self.attention_module.gamma.data = torch.tensor(gamma)
                self.attention_module.dropout.p = dropout
                
                # 評估組合性能
                combined_score = self._evaluate_combined_performance(test_images)
                interaction_matrix[i, j] = combined_score
                
                # 記錄最佳組合
                if combined_score > best_combo['score']:
                    best_combo = {
                        'gamma': gamma,
                        'dropout': dropout,
                        'score': combined_score
                    }
        
        # 恢復原始參數
        self.attention_module.gamma.data = original_gamma
        self.attention_module.dropout.p = original_dropout
        
        print(f"      最佳組合: Gamma={best_combo['gamma']:.4f}, Dropout={best_combo['dropout']:.4f}")
        print(f"      組合得分: {best_combo['score']:.4f}")
        
        return {
            'gamma_grid': gamma_grid,
            'dropout_grid': dropout_grid,
            'interaction_matrix': interaction_matrix,
            'best_combination': best_combo
        }
    
    def validate_adaptive_strategy(self, test_images):
        """
        驗證自適應調整策略的有效性
        """
        print(f"    驗證自適應調整策略...")
        
        # 模擬不同的性能場景
        scenarios = [
            {'name': '高性能需求', 'target_performance': 0.9, 'iterations': 10},
            {'name': '平衡模式', 'target_performance': 0.7, 'iterations': 8},
            {'name': '快速模式', 'target_performance': 0.5, 'iterations': 5}
        ]
        
        adaptive_results = []
        
        for scenario in scenarios:
            print(f"      測試場景: {scenario['name']}")
            
            # 重置到初始狀態
            self.attention_module.gamma.data = torch.tensor(0.5)
            self.attention_module.dropout.p = 0.1
            
            adaptation_history = []
            
            for iteration in range(scenario['iterations']):
                # 評估當前性能
                current_perf = self._evaluate_denoising_quality(test_images)
                
                # 記錄當前狀態
                adaptation_history.append({
                    'iteration': iteration,
                    'gamma': self.attention_module.gamma.item(),
                    'dropout': self.attention_module.dropout.p,
                    'performance': current_perf
                })
                
                # 自適應調整
                self._adaptive_parameter_adjustment(current_perf, scenario['target_performance'])
                
                print(f"        迭代{iteration+1}: 性能={current_perf:.4f}")
            
            adaptive_results.append({
                'scenario': scenario['name'],
                'history': adaptation_history,
                'final_performance': adaptation_history[-1]['performance']
            })
        
        return adaptive_results
    
    def _detect_saddle_point(self, x_values, y_values, stability_values):
        """
        檢測鞍點 - 基於梯度和穩定性分析
        """
        # 計算一階和二階導數
        grad1 = np.gradient(y_values)
        grad2 = np.gradient(grad1)
        
        # 找到梯度接近零的點
        zero_grad_indices = np.where(np.abs(grad1) < np.std(grad1) * 0.1)[0]
        
        if len(zero_grad_indices) == 0:
            # 如果沒有明顯的鞍點，選擇性能最高點
            optimal_idx = np.argmax(y_values)
        else:
            # 在鞍點候選中選擇穩定性最好的
            candidate_stabilities = [stability_values[i] for i in zero_grad_indices]
            best_candidate_idx = zero_grad_indices[np.argmax(candidate_stabilities)]
            optimal_idx = best_candidate_idx
        
        return {
            'optimal_gamma': x_values[optimal_idx],
            'optimal_performance': y_values[optimal_idx],
            'optimal_stability': stability_values[optimal_idx],
            'saddle_index': optimal_idx,
            'gradient_at_saddle': grad1[optimal_idx],
            'curvature_at_saddle': grad2[optimal_idx]
        }
    
    def _find_optimal_dropout(self, dropout_values, performance_scores, overfitting_scores):
        """
        找到最佳dropout值 - 平衡性能和過擬合風險
        """
        # 標準化分數
        norm_perf = np.array(performance_scores) / np.max(performance_scores)
        norm_overfit = 1.0 - (np.array(overfitting_scores) / np.max(overfitting_scores))  # 越低越好
        
        # 綜合得分 (性能權重0.7，抗過擬合權重0.3)
        balance_scores = 0.7 * norm_perf + 0.3 * norm_overfit
        
        optimal_idx = np.argmax(balance_scores)
        
        return {
            'optimal_dropout': dropout_values[optimal_idx],
            'balance_score': balance_scores[optimal_idx],
            'performance_at_optimal': performance_scores[optimal_idx],
            'overfitting_at_optimal': overfitting_scores[optimal_idx]
        }
    
    def _evaluate_attention_performance(self, test_images):
        """
        評估attention模組性能
        """
        performance_scores = []
        stability_scores = []
        
        for img_tensor in test_images[:3]:  # 使用前3張圖片
            with torch.no_grad():
                # 多次運行測試穩定性
                outputs = []
                for _ in range(5):
                    output = self.attention_module(img_tensor)
                    outputs.append(output)
                
                # 計算性能指標 (基於輸出質量)
                primary_output = outputs[0]
                perf_score = self._compute_quality_metric(primary_output, img_tensor)
                performance_scores.append(perf_score)
                
                # 計算穩定性指標 (多次運行的一致性)
                output_vars = []
                for i in range(len(outputs)-1):
                    var = torch.var(outputs[i] - outputs[i+1]).item()
                    output_vars.append(var)
                stab_score = 1.0 / (np.mean(output_vars) + 1e-8)
                stability_scores.append(stab_score)
        
        return np.mean(performance_scores), np.mean(stability_scores)
    
    def _evaluate_denoising_quality(self, test_images):
        """
        評估去噪質量
        """
        quality_scores = []
        
        for img_tensor in test_images[:3]:
            with torch.no_grad():
                enhanced_output = self.attention_module(img_tensor)
                
                # 計算PSNR或其他質量指標
                quality_score = self._compute_quality_metric(enhanced_output, img_tensor)
                quality_scores.append(quality_score)
        
        return np.mean(quality_scores)
    
    def _evaluate_overfitting_risk(self, test_images):
        """
        評估過擬合風險
        """
        # 這裡可以實現具體的過擬合檢測邏輯
        # 例如：比較訓練集和驗證集性能差異
        
        # 簡化實現：基於輸出的複雜度
        complexity_scores = []
        
        for img_tensor in test_images[:3]:
            with torch.no_grad():
                output = self.attention_module(img_tensor)
                
                # 計算輸出複雜度 (高頻成分)
                freq_complexity = torch.var(output).item()
                complexity_scores.append(freq_complexity)
        
        # 返回標準化的複雜度分數
        return np.mean(complexity_scores)
    
    def _evaluate_combined_performance(self, test_images):
        """
        評估參數組合的綜合性能
        """
        perf_score = self._evaluate_denoising_quality(test_images)
        overfit_risk = self._evaluate_overfitting_risk(test_images)
        
        # 綜合評分 (性能為主，過擬合風險為輔)
        combined_score = perf_score - 0.1 * overfit_risk
        
        return combined_score
    
    def _compute_quality_metric(self, output, input_tensor):
        """
        計算圖像質量指標
        """
        # 簡化的質量指標 - 可以替換為更複雜的評估
        with torch.no_grad():
            # 基於信噪比的簡化計算
            signal_power = torch.mean(output**2)
            noise_estimate = torch.mean((output - input_tensor)**2)
            snr = 10 * torch.log10(signal_power / (noise_estimate + 1e-8))
            
            return snr.item()
    
    def _adaptive_parameter_adjustment(self, current_performance, target_performance):
        """
        自適應參數調整邏輯
        """
        performance_gap = target_performance - current_performance
        adjustment_rate = 0.1
        
        if performance_gap > 0.1:  # 性能不足，增強attention
            # 增加gamma
            current_gamma = self.attention_module.gamma.item()
            new_gamma = min(1.0, current_gamma + adjustment_rate)
            self.attention_module.gamma.data = torch.tensor(new_gamma)
            
            # 減少dropout  
            current_dropout = self.attention_module.dropout.p
            new_dropout = max(0.05, current_dropout - adjustment_rate * 0.5)
            self.attention_module.dropout.p = new_dropout
            
        elif performance_gap < -0.05:  # 可能過擬合，減弱attention
            # 減少gamma
            current_gamma = self.attention_module.gamma.item()
            new_gamma = max(0.1, current_gamma - adjustment_rate)
            self.attention_module.gamma.data = torch.tensor(new_gamma)
            
            # 增加dropout
            current_dropout = self.attention_module.dropout.p  
            new_dropout = min(0.3, current_dropout + adjustment_rate * 0.5)
            self.attention_module.dropout.p = new_dropout
    
    def generate_visualization_report(self, results, output_dir):
        """
        生成可視化分析報告
        """
        print(f"    生成可視化報告...")
        
        # 設置matplotlib中文支持
        if self.font_prop:
            plt.rcParams['font.sans-serif'] = [self.font_prop.get_name()]
            plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
        else:
            # 如果字體載入失敗，嘗試使用系統中文字體
            try:
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                print(f"      使用後備中文字體")
            except:
                print(f"      警告：無法設置中文字體，圖表可能顯示亂碼")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Gamma鞍點分析圖
        ax1 = axes[0, 0]
        gamma_data = results['gamma']
        ax1.plot(gamma_data['gamma_values'], gamma_data['performance_scores'], 'b-', label='性能得分', linewidth=2)
        ax1.plot(gamma_data['gamma_values'], gamma_data['stability_scores'], 'r--', label='穩定性得分', linewidth=2)
        
        # 標記鞍點
        saddle_info = gamma_data['saddle_info']
        ax1.axvline(x=saddle_info['optimal_gamma'], color='g', linestyle=':', linewidth=2, alpha=0.7, label=f'最佳點: {saddle_info["optimal_gamma"]:.3f}')
        
        ax1.set_xlabel('Gamma值', fontproperties=self.font_prop)
        ax1.set_ylabel('性能得分', fontproperties=self.font_prop)
        ax1.set_title('Gamma參數鞍點分析', fontproperties=self.font_prop)
        ax1.legend(prop=self.font_prop)
        ax1.grid(True, alpha=0.3)
        
        # 2. Dropout優化圖
        ax2 = axes[0, 1]
        dropout_data = results['dropout']
        ax2.plot(dropout_data['dropout_values'], dropout_data['performance_scores'], 'g-', label='去噪性能', linewidth=2)
        ax2.plot(dropout_data['dropout_values'], dropout_data['overfitting_scores'], 'r-', label='過擬合風險', linewidth=2)
        
        # 標記最佳點
        optimal_info = dropout_data['optimal_info']
        ax2.axvline(x=optimal_info['optimal_dropout'], color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'最佳點: {optimal_info["optimal_dropout"]:.3f}')
        
        ax2.set_xlabel('Dropout概率', fontproperties=self.font_prop)
        ax2.set_ylabel('得分', fontproperties=self.font_prop)
        ax2.set_title('Dropout參數優化', fontproperties=self.font_prop)
        ax2.legend(prop=self.font_prop)
        ax2.grid(True, alpha=0.3)
        
        # 3. 參數交互熱圖
        ax3 = axes[1, 0]
        interaction_data = results['interaction']
        im = ax3.imshow(interaction_data['interaction_matrix'], cmap='viridis', aspect='auto')
        ax3.set_xlabel('Dropout索引', fontproperties=self.font_prop)
        ax3.set_ylabel('Gamma索引', fontproperties=self.font_prop)
        ax3.set_title('參數交互作用熱圖', fontproperties=self.font_prop)
        
        # 添加顏色條
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('組合性能得分', fontproperties=self.font_prop)
        
        # 標記最佳組合點
        best_combo = interaction_data['best_combination']
        gamma_idx = np.argmin(np.abs(interaction_data['gamma_grid'] - best_combo['gamma']))
        dropout_idx = np.argmin(np.abs(interaction_data['dropout_grid'] - best_combo['dropout']))
        ax3.plot(dropout_idx, gamma_idx, 'r*', markersize=15, label=f'最佳組合')
        ax3.legend(prop=self.font_prop)
        
        # 4. 自適應策略驗證
        ax4 = axes[1, 1]
        adaptive_data = results['adaptive']
        
        for scenario_data in adaptive_data:
            history = scenario_data['history']
            iterations = [h['iteration'] for h in history]
            performances = [h['performance'] for h in history]
            ax4.plot(iterations, performances, 'o-', label=scenario_data['scenario'], linewidth=2, markersize=6)
        
        ax4.set_xlabel('迭代次數', fontproperties=self.font_prop)
        ax4.set_ylabel('性能得分', fontproperties=self.font_prop)
        ax4.set_title('自適應策略驗證', fontproperties=self.font_prop)
        ax4.legend(prop=self.font_prop)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表
        report_path = os.path.join(output_dir, 'attention_parameter_analysis.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      報告已保存至: {report_path}")
    
    def save_analysis_results(self, results, output_dir):
        """
        保存分析結果到JSON文件
        """
        def convert_to_serializable(obj):
            """遞歸轉換為JSON可序列化格式"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int8, np.int16)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # 轉換所有數據為JSON可序列化格式
        json_results = convert_to_serializable(results)
        
        # 保存JSON文件
        json_path = os.path.join(output_dir, 'attention_analysis_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"      結果已保存至: {json_path}")
        
        # 生成簡要報告
        report_path = os.path.join(output_dir, 'optimization_summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Self-Attention參數優化報告 ===\n\n")
            
            # Gamma優化結果
            gamma_info = results['gamma']['saddle_info']
            f.write(f"1. Gamma參數優化:\n")
            f.write(f"   最佳值: {gamma_info['optimal_gamma']:.4f}\n")
            f.write(f"   性能得分: {gamma_info['optimal_performance']:.4f}\n")
            f.write(f"   穩定性得分: {gamma_info['optimal_stability']:.4f}\n\n")
            
            # Dropout優化結果
            dropout_info = results['dropout']['optimal_info'] 
            f.write(f"2. Dropout參數優化:\n")
            f.write(f"   最佳值: {dropout_info['optimal_dropout']:.4f}\n")
            f.write(f"   平衡得分: {dropout_info['balance_score']:.4f}\n\n")
            
            # 最佳組合
            best_combo = results['interaction']['best_combination']
            f.write(f"3. 最佳參數組合:\n")
            f.write(f"   Gamma: {best_combo['gamma']:.4f}\n") 
            f.write(f"   Dropout: {best_combo['dropout']:.4f}\n")
            f.write(f"   組合得分: {best_combo['score']:.4f}\n\n")
            
            # 自適應策略效果
            f.write(f"4. 自適應策略驗證:\n")
            for scenario_data in results['adaptive']:
                final_perf = scenario_data['final_performance']
                f.write(f"   {scenario_data['scenario']}: 最終性能 {final_perf:.4f}\n")
        
        print(f"      簡要報告已保存至: {report_path}")


# 使用示例
def run_attention_optimization_example():
    """
    運行attention優化的完整示例
    """
    print("=== Self-Attention參數優化示例 ===")
    
    # 模擬創建一個attention模組 (實際使用時替換為真實模組)
    from enhanced_feature_extraction import RobustSelfAttention
    
    attention_module = RobustSelfAttention(channels=3)
    
    # 創建優化器
    optimizer = AttentionParameterOptimizer(attention_module)
    
    # 模擬一些測試圖像 (實際使用時加載真實圖像)
    test_images = [torch.randn(1, 3, 256, 256) for _ in range(5)]
    
    # 運行全面分析
    results = optimizer.comprehensive_parameter_analysis(test_images)
    
    print("\n=== 優化完成！ ===")
    print("詳細結果請查看生成的報告文件。")
    
    return results

if __name__ == "__main__":
    # 運行示例
    run_attention_optimization_example()
