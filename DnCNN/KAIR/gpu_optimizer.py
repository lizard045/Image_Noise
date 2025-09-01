import os
import torch
import torch.backends.cudnn as cudnn
from self_attention_module import RobustSelfAttention

try:
    from enhanced_feature_extraction import AdvancedFeatureExtractor, enhanced_feature_based_fusion
    from attention_optimizer import AttentionParameterOptimizer
    ENHANCED_FEATURES_AVAILABLE = True
    print("    增強功能模組載入成功")
except ImportError as e:
    print(f"    增強功能模組載入失敗: {str(e)}")
    ENHANCED_FEATURES_AVAILABLE = False


class GPUOptimizer:
    """GPU優化管理器 - 穩定版 + Self-Attention + 增強特徵提取"""

    def __init__(self):
        self.device = None
        self.gpu_info = {}
        self.use_amp = False
        self.scaler = None
        self.attention_module = None

        # 新增：增強功能組件
        self.feature_extractor = None
        self.attention_optimizer = None
        self.enhanced_mode = False

    def setup_gpu(self, enable_attention=False, enable_enhanced_features=False):
        """設置GPU優化配置 + 增強功能"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            cudnn.benchmark = True
            cudnn.deterministic = False

            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            try:
                torch.cuda.set_per_process_memory_fraction(0.8, 0)
            except Exception as e:
                print(f"    set_per_process_memory_fraction失敗: {str(e)[:40]}...")

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
                    self.attention_module.eval()
                    print(f"    Self-Attention模組: 已啟用 (修復版)")

                    # 初始化attention調優器 (如果啟用增強功能)
                    if enable_enhanced_features and ENHANCED_FEATURES_AVAILABLE:
                        try:
                            self.attention_optimizer = AttentionParameterOptimizer(
                                self.attention_module,
                                font_path="Mamelon.otf",
                            )
                            print(f"    Attention調優器: 已初始化")
                        except Exception as opt_e:
                            print(f"    Attention調優器: 初始化失敗 {str(opt_e)[:40]}...")
                            self.attention_optimizer = None

                except Exception as e:
                    print(f"    Self-Attention模組: 初始化失敗 {str(e)[:50]}...")
                    self.attention_module = None
            else:
                if enable_attention:
                    print(f"    Self-Attention模組: GPU記憶體不足 (需要≥6GB)")
                else:
                    print(f"    Self-Attention模組: 已停用")

            # 初始化增強特徵提取器
            if enable_enhanced_features and ENHANCED_FEATURES_AVAILABLE:
                try:
                    self.feature_extractor = AdvancedFeatureExtractor()
                    self.enhanced_mode = True
                    print(f"    增強特徵提取器: 已啟用")
                except Exception as feat_e:
                    print(f"    增強特徵提取器: 初始化失敗 {str(feat_e)[:40]}...")
                    self.feature_extractor = None
                    self.enhanced_mode = False
            else:
                if enable_enhanced_features:
                    print(f"    增強特徵提取器: 依賴模組未載入")
                else:
                    print(f"    增強特徵提取器: 已停用")

            print(f" GPU優化啟用:")
            print(f"   設備: {self.gpu_info['name']}")
            print(f"   總記憶體: {self.gpu_info['memory_total']:.1f}GB")
            print(f"   可用記憶體: {self.gpu_info['memory_available']:.1f}GB")
            print(f"   混合精度AMP: {self.use_amp}")
            print(f"   Self-Attention: {'啟用' if self.attention_module else '未啟用'}")
            print(f"   增強特徵提取: {'啟用' if self.enhanced_mode else '未啟用'}")
            print(f"   Attention調優: {'啟用' if self.attention_optimizer else '未啟用'}")

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

        # 進一步縮小初始分塊尺寸，降低單塊記憶體需求
        if self.gpu_info['memory_total'] >= 8.0:
            max_tile_size = 256  # 原為512，為了8GB VRAM改為256
        elif self.gpu_info['memory_total'] >= 6.0:
            max_tile_size = 192
        else:
            max_tile_size = 128


        tile_h = min(max_tile_size, h)
        tile_w = min(max_tile_size, w)

        return tile_h, tile_w

    def cleanup_memory(self):
        """清理GPU記憶體"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()