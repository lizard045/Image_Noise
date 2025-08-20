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
from collections import OrderedDict
import time

from utils import utils_logger
from utils import utils_image as util
from models.network_unet import UNetRes
from processing_utils import _prepare_image
from gpu_optimizer import GPUOptimizer
from dual_stream import stable_dual_stream_processing, tile_process_stable, fallback_single_processing
from residual_refinement import _refine_residual_noise_hotspots, _residual_bilateral_denoise

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    parser = argparse.ArgumentParser(description='台北101影像去噪處理 - 穩定版本 + Self-Attention + 增強功能')
    parser.add_argument('--drunet_model_path', type=str, default='model_zoo/drunet_color.pth')
    parser.add_argument('--input_dir', type=str, default='testsets/TAIPEI_NO_TUNE_8K')
    parser.add_argument('--output_dir', type=str, default='taipei101_stable_denoised')
    parser.add_argument('--noise_level', type=float, default=2.0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--enable_attention', action='store_true', help='啟用修復版Self-Attention增強')
    parser.add_argument('--enable_enhanced_features', action='store_true', help='啟用增強版特徵提取和調優')
    parser.add_argument('--optimize_attention_params', action='store_true', help='啟用Attention參數自動調優')
    parser.add_argument('--enable_region_adaptive', action='store_true', help='啟用區域自適應處理 (解決去噪不均勻問題)')
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
    logger.info(f'增強特徵提取: {"啟用" if args.enable_enhanced_features else "停用"}')
    logger.info(f'參數自動調優: {"啟用" if args.optimize_attention_params else "停用"}')
    logger.info(f'區域自適應: {"啟用" if args.enable_region_adaptive else "停用"}')
    logger.info('穩定模式: 雙流處理 + 可選增強功能')
    
    # GPU優化器設定
    gpu_optimizer = GPUOptimizer()

    device = gpu_optimizer.setup_gpu(
        enable_attention=args.enable_attention,
        enable_enhanced_features=args.enable_enhanced_features
    )
    
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
        print(f"   增強功能: {'完整模式' if args.enable_enhanced_features else '標準模式'}")
        
        # Attention參數調優 (如果啟用)
        if (args.optimize_attention_params and args.enable_attention and 
            gpu_optimizer.attention_optimizer is not None):
            print(f"\n  開始Attention參數調優...")
            try:
                # 載入一些樣本圖片進行調優
                sample_images = []
                sample_paths = util.get_image_paths(args.input_dir)[:3]  # 使用前3張圖片
                
                for sample_path in sample_paths:
                    sample_img = util.imread_uint(sample_path, n_channels=3)
                    sample_tensor = util.uint2tensor4(sample_img).to(device)
                    sample_images.append(sample_tensor)
                
                if len(sample_images) > 0:
                    # 運行參數優化
                    optimization_results = gpu_optimizer.attention_optimizer.comprehensive_parameter_analysis(
                        sample_images, 
                        output_dir=os.path.join(args.output_dir, "attention_optimization")
                    )
                    
                    # 應用最佳參數
                    best_combo = optimization_results['interaction']['best_combination']
                    gpu_optimizer.attention_module.gamma.data = torch.tensor(best_combo['gamma'])
                    gpu_optimizer.attention_module.dropout.p = best_combo['dropout']
                    
                    print(f"    參數調優完成!")
                    print(f"      最佳Gamma: {best_combo['gamma']:.4f}")
                    print(f"      最佳Dropout: {best_combo['dropout']:.4f}")
                    print(f"      優化得分: {best_combo['score']:.4f}")
                    
                    logger.info(f'Attention參數已優化: Gamma={best_combo["gamma"]:.4f}, Dropout={best_combo["dropout"]:.4f}')
                else:
                    print(f"    參數調優跳過: 無法載入樣本圖片")
                    
            except Exception as opt_e:
                print(f"    參數調優失敗: {str(opt_e)[:60]}...")
                logger.warning(f'Attention參數調優失敗: {str(opt_e)}')
        
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
            
            # 載入影像並統一格式
            img_L = _prepare_image(util.imread_uint(img_path, n_channels=3))
            original_shape = img_L.shape
            print(f"    影像尺寸: {original_shape}")
            
            # 根據圖片大小選擇處理方式
            if gpu_optimizer.should_tile_process(original_shape[:2]):
                img_E = tile_process_stable(model, img_L, device, args.noise_level, gpu_optimizer, 
                                          enable_attention=args.enable_attention, 
                                          enable_region_adaptive=args.enable_region_adaptive)
            else:
                img_E = stable_dual_stream_processing(model, img_L, device, args.noise_level, gpu_optimizer, 
                                                    enable_attention=args.enable_attention,
                                                    enable_region_adaptive=args.enable_region_adaptive)
            # 二次降噪：殘留噪點熱區精修
            img_E = _refine_residual_noise_hotspots(
                model, img_L, img_E, device, args.noise_level, gpu_optimizer
            )
            # 二次降噪：殘差域雙邊濾波
            img_E = _residual_bilateral_denoise(img_L, img_E)
            
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
        print(f"\n 系統特點與增強功能:")
        print(f"   •  穩定性: 雙流處理基礎上的漸進式增強")
        print(f"   •  自適應參數: 根據影像特性自動調整雙流參數")
        print(f"   •  邊緣檢測: {'多尺度、多方向增強特徵提取' if args.enable_enhanced_features else '3×3和5×5核融合邊緣檢測'}")
        print(f"   •  Self-Attention: {'修復版注意力機制，智能細節增強' if args.enable_attention else '未啟用，追求高速穩定'}")
        print(f"   •  參數調優: {'自動參數尋優，找到最佳鞍點' if args.optimize_attention_params else '使用預設參數'}")
        print(f"   •  特徵提取: {'結構張量+LBP紋理+小波變換' if args.enable_enhanced_features else '傳統Laplacian邊緣'}")
        print(f"   •  記憶體優化: 頻繁清理，避免記憶體累積")
        print(f"   •  詳細統計: 每張圖片的處理參數和統計資訊")
        
        if args.enable_enhanced_features:
            print(f"\n 增強功能詳情:")
            print(f"   •  多尺度特徵: 1x, 2x, 4x, 8x 尺度金字塔")
            print(f"   •  方向邊緣: Sobel+Canny+Laplacian+梯度一致性") 
            print(f"   •  結構分析: 本徵值+相干性+各向異性+角點強度")
            print(f"   •  紋理特徵: LBP局部二值模式+複雜度統計")
            print(f"   •  自適應閾值: 動態tau/M比例，非固定25%/85%")
            
        if args.optimize_attention_params:
            print(f"\n 參數調優功能:")
            print(f"   •  鞍點檢測: 自動找到性能穩定的最佳點")  
            print(f"   •  交互分析: Gamma-Dropout參數交互作用熱圖")
            print(f"   •  過擬合監控: 平衡性能和泛化能力")
            print(f"   •  自適應調整: 根據目標性能動態優化")
            print(f"   •  可視化報告: 完整的調優過程分析圖表")

if __name__ == "__main__":
    main()