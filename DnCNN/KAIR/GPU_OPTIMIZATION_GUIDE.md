# GPU優化訓練指南

## 🚀 主要優化特性

### 1. 自動GPU配置
- **硬體檢測**: 自動檢測GPU記憶體和CPU核心數
- **動態調整**: 根據硬體自動調整批次大小和工作線程數
- **性能配置檔案**: 提供4種預設配置適應不同需求

### 2. 混合精度訓練 (AMP)
- **記憶體節省**: 減少50%的GPU記憶體使用
- **速度提升**: 在支援的GPU上提升1.5-2倍訓練速度
- **自動梯度縮放**: 防止數值下溢問題

### 3. DataLoader優化
- **預取機制**: 使用prefetch_factor加速數據載入
- **持久化工作線程**: 減少線程創建開銷
- **記憶體固定**: 啟用pin_memory加速GPU傳輸

### 4. JIT編譯優化
- **模型編譯**: 使用torch.jit.trace編譯模型
- **執行加速**: 減少Python解釋開銷

### 5. 記憶體管理
- **定期清理**: 每100步清理GPU記憶體碎片
- **智能分配**: 設定記憶體分配策略
- **實時監控**: 顯示GPU記憶體使用情況

## 📊 使用方法

### 基本使用

```bash
# 使用平衡模式（推薦）
python3 main_train_drunet.py --performance_profile balanced

# 使用最快速度模式
python3 main_train_drunet.py --performance_profile ultra_fast

# 使用高質量模式
python3 main_train_drunet.py --performance_profile high_quality

# 使用記憶體節約模式
python3 main_train_drunet.py --performance_profile memory_efficient
```

### 使用優化腳本

```bash
# 使用GPU優化啟動腳本
./train_gpu_optimized.sh balanced

# 同時監控GPU性能
python3 monitor_gpu_performance.py --interval 5 &
./train_gpu_optimized.sh balanced
```

### 性能配置檔案說明

| 配置檔案 | 批次大小 | 細節增強 | 頻域保護 | 適用場景 |
|---------|---------|---------|---------|---------|
| `ultra_fast` | 2x | ❌ | ❌ | 快速測試、原型開發 |
| `balanced` | 1x | ✅ | ✅ | 一般訓練（推薦） |
| `high_quality` | 0.7x | ✅ | ✅ | 高品質要求 |
| `memory_efficient` | 0.5x | ✅ | ❌ | 低記憶體設備 |

## 🔧 硬體建議

### GPU配置建議

| GPU記憶體 | 建議批次大小 | 建議配置檔案 | 預期提升 |
|----------|-------------|-------------|---------|
| 8GB+ | 8-16 | `balanced` / `ultra_fast` | 3-5倍 |
| 4-8GB | 4-8 | `balanced` | 2-3倍 |
| <4GB | 2-4 | `memory_efficient` | 1.5-2倍 |

### 環境變數優化

```bash
# GPU記憶體分配優化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 指定GPU設備
export CUDA_VISIBLE_DEVICES=0

# 啟用cuDNN基準測試
export TORCH_CUDNN_BENCHMARK=1
```

## 📈 性能監控

### 實時監控

```bash
# 啟動性能監控（每10秒記錄一次）
python3 monitor_gpu_performance.py --interval 10

# 監控指定時長（3600秒=1小時）
python3 monitor_gpu_performance.py --duration 3600
```

### 監控輸出示例

```
[14:30:15] GPU:  85.2% | VRAM: 6.2GB/8.0GB (77.5%) | Temp: 72°C | CPU:  45.3% | RAM:  68.2%
[14:30:25] GPU:  88.7% | VRAM: 6.4GB/8.0GB (80.0%) | Temp: 74°C | CPU:  48.1% | RAM:  69.5%
```

### 自動報告生成

監控結束後會自動生成：
- `gpu_performance.log`: 詳細性能日誌
- `gpu_performance_report.png`: 性能圖表
- 優化建議和統計數據

## ⚡ 預期性能提升

### 訓練速度提升

| 優化項目 | 提升倍數 | 說明 |
|---------|---------|------|
| 混合精度 (AMP) | 1.5-2x | 在支援的GPU上 |
| DataLoader優化 | 1.2-1.5x | 減少I/O等待時間 |
| JIT編譯 | 1.1-1.3x | 減少Python開銷 |
| 批次大小優化 | 1.2-2x | 提升GPU利用率 |
| **總體提升** | **2-5x** | 綜合所有優化 |

### 記憶體使用優化

- **混合精度**: 減少40-50%記憶體使用
- **智能分配**: 減少記憶體碎片
- **定期清理**: 避免記憶體洩漏

## 🛠️ 故障排除

### 常見問題

#### 1. GPU記憶體不足 (OOM)

```bash
# 解決方案1: 使用記憶體節約模式
python3 main_train_drunet.py --performance_profile memory_efficient

# 解決方案2: 手動調整批次大小
# 修改 options/train_drunet.json 中的 dataloader_batch_size
```

#### 2. GPU利用率低

```bash
# 檢查數據載入瓶頸
python3 monitor_gpu_performance.py --interval 5

# 增加批次大小
python3 main_train_drunet.py --performance_profile ultra_fast
```

#### 3. 訓練速度慢

```bash
# 確保使用GPU
nvidia-smi

# 檢查環境變數
echo $PYTORCH_CUDA_ALLOC_CONF
echo $CUDA_VISIBLE_DEVICES

# 使用最快配置
python3 main_train_drunet.py --performance_profile ultra_fast
```

#### 4. JIT編譯失敗

```
⚠️  JIT編譯失敗，使用標準模式: [錯誤信息]
```

這是正常的，程式會自動回退到標準模式，不影響訓練。

### 性能調優建議

1. **監控GPU利用率**: 目標保持在80-95%
2. **記憶體使用**: 建議不超過85%
3. **溫度監控**: 保持在80°C以下
4. **批次大小**: 根據GPU記憶體逐步調整

## 📋 檢查清單

訓練前檢查：
- [ ] GPU驅動和CUDA版本正確
- [ ] 設定環境變數
- [ ] 選擇合適的性能配置檔案
- [ ] 啟動性能監控（可選）

訓練中監控：
- [ ] GPU利用率 > 80%
- [ ] 記憶體使用 < 85%
- [ ] 溫度 < 80°C
- [ ] 無OOM錯誤

## 🎯 最佳實踐

1. **首次使用**: 建議從 `balanced` 配置開始
2. **性能測試**: 使用 `ultra_fast` 快速驗證配置
3. **生產訓練**: 根據監控結果調整配置
4. **記憶體不足**: 使用 `memory_efficient` 配置
5. **高品質需求**: 使用 `high_quality` 配置

---

**注意**: 所有優化都是自動應用的，無需手動修改程式碼。系統會根據硬體配置自動選擇最佳設定。