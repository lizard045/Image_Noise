# 實用影像去噪工具使用說明

## 🎯 概述

這個實用影像去噪工具可以直接對你的照片進行去噪處理，無需先加入噪點。工具提供了多種濾波器選項，並能自動推薦最適合的濾波器。

**新功能：影像品質評估**
- 🔍 **CW-SSIM 評估**：結構相似性指數，衡量影像結構保持程度
- 📊 **PSNR 評估**：峰值信噪比，衡量噪聲去除效果
- 🏆 **智能推薦**：基於綜合評分推薦最佳濾波器

## 🚀 快速開始

### 1. 互動模式（推薦）
```bash
python run_practical.py
```
程式會自動找尋當前目錄中的影像檔案，並提供友好的選擇介面。

### 2. 批次模式
```bash
python run_practical.py --image your_image.jpg
```

### 3. 直接使用主程式
```bash
python practical_denoising.py your_image.jpg
```

## 📊 評估指標說明

### CW-SSIM (結構相似性指數)
- **範圍**: 0-1
- **越高越好**: 1表示完全相同
- **評價標準**:
  - \> 0.9: 優秀
  - \> 0.8: 良好
  - \> 0.7: 一般
  - ≤ 0.7: 較差

### PSNR (峰值信噪比)
- **單位**: dB (分貝)
- **越高越好**: 表示噪聲越少
- **評價標準**:
  - \> 30dB: 優秀
  - \> 25dB: 良好
  - \> 20dB: 一般
  - ≤ 20dB: 較差

### 綜合評分
程式使用加權平均計算綜合評分：
```
綜合評分 = 0.6 × CW-SSIM + 0.4 × (PSNR/40)
```

## 🔧 濾波器選項

### 自動推薦（預設）
```bash
python practical_denoising.py image.jpg --filter auto
```
程式會分析影像特徵，自動選擇最適合的濾波器。

### 所有濾波器
```bash
python practical_denoising.py image.jpg --filter all
```
應用所有可用的濾波器，方便比較效果。

### 特定濾波器

#### 雙邊濾波器 - 最適合一般照片
```bash
python practical_denoising.py image.jpg --filter bilateral
```
- ✅ 保持邊緣細節
- ✅ 去除高斯噪聲
- ✅ 適合人像和風景照片
- 📊 平均 CW-SSIM: 0.85-0.95

#### 非局部均值濾波器 - 最適合保持紋理
```bash
python practical_denoising.py image.jpg --filter non_local_means
```
- ✅ 保持紋理細節
- ✅ 去除重複模式噪聲
- ✅ 適合有豐富紋理的影像
- 📊 平均 CW-SSIM: 0.88-0.96

#### 中值濾波器 - 最適合脈衝噪聲
```bash
python practical_denoising.py image.jpg --filter median
```
- ✅ 去除椒鹽噪聲
- ✅ 去除脈衝噪聲
- ✅ 適合有點狀噪聲的影像
- 📊 平均 CW-SSIM: 0.80-0.92

#### 高斯濾波器 - 最適合平滑影像
```bash
python practical_denoising.py image.jpg --filter gaussian
```
- ✅ 平滑處理
- ✅ 去除高頻噪聲
- ✅ 適合平滑的影像
- 📊 平均 CW-SSIM: 0.85-0.95

#### 維納濾波器 - 最適合已知噪聲
```bash
python practical_denoising.py image.jpg --filter wiener
```
- ✅ 頻域處理
- ✅ 適合已知噪聲特性
- ✅ 數學上最優
- 📊 平均 CW-SSIM: 0.75-0.85

#### 形態學濾波器 - 最適合結構化噪聲
```bash
python practical_denoising.py image.jpg --filter morphological
```
- ✅ 去除結構化噪聲
- ✅ 適合二值化影像
- ✅ 保持影像結構
- 📊 平均 CW-SSIM: 0.82-0.92

#### 自適應濾波器 - 最適合複雜噪聲
```bash
python practical_denoising.py image.jpg --filter adaptive
```
- ✅ 根據局部特徵調整
- ✅ 適合複雜噪聲環境
- ✅ 平衡邊緣和平滑區域
- 📊 平均 CW-SSIM: 0.88-0.98

## 📋 參數選項

### 核心大小調整
```bash
python practical_denoising.py image.jpg --kernel_size 7
```
- 預設值：5
- 較大的核心：更強的去噪效果，但可能模糊細節
- 較小的核心：保持更多細節，但去噪效果較弱

### 輸出資料夾
```bash
python practical_denoising.py image.jpg --output_dir my_results
```
- 預設值：practical_results
- 指定自訂的輸出資料夾名稱

### 跳過評估
```bash
python practical_denoising.py image.jpg --no-metrics
```
- 跳過CW-SSIM和PSNR評估，加快處理速度
- 適合批次處理大量影像

## 📸 支援的影像格式

- PNG
- JPG/JPEG
- BMP
- TIFF/TIF

## 📁 輸出檔案

處理完成後，結果會保存在指定的輸出資料夾中：

```
practical_results/
├── 00_original.png          # 原始影像
├── 01_filtered_image.png    # 處理後的影像
└── ...                      # 其他濾波器結果（如果使用 --filter all）
```

## 🔍 自動推薦系統

程式會分析以下影像特徵：
- **噪聲等級**：基於局部方差統計
- **邊緣密度**：檢測邊緣細節豐富度
- **平均方差**：整體影像復雜度
- **梯度強度**：影像銳利度

根據這些特徵自動選擇最適合的濾波器。

## 📊 品質評估報告

使用 `--filter all` 時，程式會生成詳細的品質評估報告：

```
======================================================================
品質評估結果總結
======================================================================
濾波器             CW-SSIM    PSNR(dB)   總體評價
----------------------------------------------------------------------
中值濾波器         0.9275    38.73     優秀
高斯濾波器         0.9671    42.48     優秀
雙邊濾波器         0.9058    38.50     優秀
維納濾波器         0.7962    30.84     良好
非局部均值濾波器   0.9236    39.62     優秀
形態學濾波器       0.9625    41.86     優秀
自適應濾波器       0.9801    44.90     優秀

[BEST] 推薦最佳濾波器: 自適應濾波器
   CW-SSIM: 0.9801, PSNR: 44.90 dB
```

## 💡 使用建議

### 📷 不同照片類型的建議

1. **人像照片**：推薦使用 `bilateral` 或 `auto`
   - 保持膚色自然，細節清晰
   - 預期 CW-SSIM > 0.85

2. **風景照片**：推薦使用 `non_local_means` 或 `auto`
   - 保持紋理和細節
   - 預期 CW-SSIM > 0.88

3. **老照片修復**：推薦使用 `median` 或 `morphological`
   - 去除歲月痕跡和斑點
   - 預期 CW-SSIM > 0.80

4. **掃描文件**：推薦使用 `median` 或 `morphological`
   - 去除掃描噪聲
   - 預期 CW-SSIM > 0.82

5. **低光照片**：推薦使用 `bilateral` 或 `adaptive`
   - 平衡噪聲和細節
   - 預期 CW-SSIM > 0.80

### 🎨 效果不滿意時的處理

1. **CW-SSIM < 0.7**：嘗試更溫和的濾波器
2. **PSNR < 25dB**：嘗試更強的去噪濾波器
3. **邊緣模糊**：使用 `bilateral` 或 `adaptive`
4. **紋理丟失**：使用 `non_local_means`
5. **整體模糊**：減小 `kernel_size` 參數

## 🛠️ 故障排除

### 常見問題

1. **找不到影像檔案**
   - 確認檔案路徑正確
   - 檢查檔案格式是否支援

2. **記憶體不足**
   - 嘗試縮小影像尺寸
   - 使用較小的核心大小
   - 加上 `--no-metrics` 跳過評估

3. **處理速度慢**
   - `non_local_means` 處理時間較長
   - 考慮使用 `bilateral` 或 `gaussian`
   - 加上 `--no-metrics` 跳過評估

4. **效果不明顯**
   - 原始影像可能噪聲較少
   - 查看CW-SSIM和PSNR數值
   - 嘗試不同的濾波器

5. **評估指標異常**
   - 確保原始影像品質良好
   - 檢查是否有處理錯誤
   - 對比視覺效果和數值

## 🎉 範例使用流程

```bash
# 1. 進入 src 目錄
cd filter/src

# 2. 啟動互動模式
python run_practical.py

# 3. 選擇影像檔案
# 4. 選擇濾波器（推薦選 1 自動推薦）
# 5. 查看評估結果和建議
# 6. 檢查結果在 practical_results 資料夾

# 或者直接使用命令列
python practical_denoising.py your_image.jpg --filter all
```

## 🎯 高級技巧

### 批次處理多個影像
```bash
for image in *.jpg; do
    python practical_denoising.py "$image" --filter auto
done
```

### 快速預覽（跳過評估）
```bash
python practical_denoising.py image.jpg --filter auto --no-metrics
```

### 自訂輸出目錄
```bash
python practical_denoising.py image.jpg --filter all --output_dir results_$(date +%Y%m%d)
```

享受你的高品質去噪體驗！ 🎨✨ 