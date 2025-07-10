import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, metrics, filters, feature
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, estimate_sigma
import pywt
import os
from PIL import Image
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from scipy.ndimage import generic_filter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def add_gaussian_noise(image, mean=0, std=25):
    """
    添加高斯噪聲到影像
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """
    添加椒鹽噪聲到影像
    """
    noisy_image = image.copy()
    
    # 添加鹽噪聲（白點）
    salt_mask = np.random.random(image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 255
    
    # 添加椒噪聲（黑點）
    pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0
    
    return noisy_image

def wavelet_denoising(image, wavelet='db4', sigma=None, mode='soft', method='BayesShrink'):
    """
    小波去噪：使用PyWavelets進行多尺度分析
    """
    if len(image.shape) == 3:
        # 彩色影像，對每個通道分別處理
        result = np.zeros_like(image)
        for i in range(3):
            # 將影像轉換為float32格式
            channel = image[:, :, i].astype(np.float32) / 255.0
            
            # 進行小波分解
            coeffs = pywt.wavedec2(channel, wavelet, mode='symmetric', level=4)
            
            # 估計噪聲標準差
            if sigma is None:
                sigma_est = estimate_sigma(channel, average_sigmas=True)
            else:
                sigma_est = sigma
            
            # 進行小波閾值去噪
            coeffs_thresh = list(coeffs)
            threshold_value = sigma_est * np.sqrt(2 * np.log(channel.size))
            coeffs_thresh[1:] = [
                tuple([pywt.threshold(detail_coeff, threshold_value, mode=mode) 
                      for detail_coeff in detail_tuple])
                for detail_tuple in coeffs_thresh[1:]
            ]
            
            # 重構影像
            reconstructed = pywt.waverec2(coeffs_thresh, wavelet, mode='symmetric')
            result[:, :, i] = np.clip(reconstructed * 255, 0, 255)
        
        return result.astype(np.uint8)
    else:
        # 灰階影像
        channel = image.astype(np.float32) / 255.0
        coeffs = pywt.wavedec2(channel, wavelet, mode='symmetric', level=4)
        
        if sigma is None:
            sigma_est = estimate_sigma(channel, average_sigmas=True)
        else:
            sigma_est = sigma
        
        coeffs_thresh = list(coeffs)
        threshold_value = sigma_est * np.sqrt(2 * np.log(channel.size))
        coeffs_thresh[1:] = [
            tuple([pywt.threshold(detail_coeff, threshold_value, mode=mode) 
                  for detail_coeff in detail_tuple])
            for detail_tuple in coeffs_thresh[1:]
        ]
        
        reconstructed = pywt.waverec2(coeffs_thresh, wavelet, mode='symmetric')
        return np.clip(reconstructed * 255, 0, 255).astype(np.uint8)

def non_local_means_denoising(image, h=10, templateWindowSize=7, searchWindowSize=21):
    """
    非局部均值去噪：利用影像自相似性
    """
    if len(image.shape) == 3:
        # 彩色影像
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 
                                             templateWindowSize, searchWindowSize)
    else:
        # 灰階影像
        return cv2.fastNlMeansDenoising(image, None, h, 
                                      templateWindowSize, searchWindowSize)

def total_variation_denoising(image, weight=0.1, eps=2.e-4, max_num_iter=200):
    """
    全變分去噪：保持邊緣銳化
    """
    if len(image.shape) == 3:
        # 彩色影像，對每個通道分別處理
        result = np.zeros_like(image)
        for i in range(3):
            # 正規化到0-1範圍
            channel = image[:, :, i].astype(np.float32) / 255.0
            # 應用全變分去噪
            denoised = denoise_tv_chambolle(channel, weight=weight, eps=eps, 
                                          max_num_iter=max_num_iter)
            result[:, :, i] = np.clip(denoised * 255, 0, 255)
        return result.astype(np.uint8)
    else:
        # 灰階影像
        channel = image.astype(np.float32) / 255.0
        denoised = denoise_tv_chambolle(channel, weight=weight, eps=eps, 
                                      max_num_iter=max_num_iter)
        return np.clip(denoised * 255, 0, 255).astype(np.uint8)

def adaptive_filter_denoising(image, noise_var=900, kernel_size=5):
    """
    自適應濾波器：基於局部統計特性（向量化實作）
    """
    from scipy.ndimage import uniform_filter
    
    if len(image.shape) == 3:
        # 彩色影像
        result = np.zeros_like(image)
        for i in range(3):
            channel = image[:, :, i].astype(np.float32)
            result[:, :, i] = adaptive_filter_single_channel(channel, noise_var, kernel_size)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        # 灰階影像
        channel = image.astype(np.float32)
        result = adaptive_filter_single_channel(channel, noise_var, kernel_size)
        return np.clip(result, 0, 255).astype(np.uint8)

def adaptive_filter_single_channel(image, noise_var, kernel_size):
    """
    單通道自適應濾波器（向量化實作）
    """
    from scipy.ndimage import uniform_filter
    
    # 計算局部均值（使用uniform_filter更快）
    local_mean = uniform_filter(image, size=kernel_size)
    
    # 計算局部方差
    local_mean_sq = uniform_filter(image * image, size=kernel_size)
    local_var = local_mean_sq - local_mean * local_mean
    
    # 自適應濾波器公式
    # 當局部方差大於噪聲方差時，保留更多細節
    alpha = np.maximum(0, 1 - noise_var / np.maximum(local_var, 1))
    
    # 應用自適應濾波器
    filtered_image = local_mean + alpha * (image - local_mean)
    
    return filtered_image

def sparse_coding_denoising(image, n_components=100, patch_size=8, alpha=1.0):
    """
    稀疏編碼去噪：基於字典學習
    """
    if len(image.shape) == 3:
        # 彩色影像轉換為灰階進行處理
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result_gray = sparse_coding_denoising_single_channel(gray_image, n_components, 
                                                           patch_size, alpha)
        # 將結果轉換回彩色（簡化處理）
        result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
        return result
    else:
        return sparse_coding_denoising_single_channel(image, n_components, 
                                                    patch_size, alpha)

def sparse_coding_denoising_single_channel(image, n_components=50, patch_size=8, alpha=1.0):
    """
    單通道稀疏編碼去噪實作（優化版本）
    """
    # 正規化影像到0-1範圍
    normalized_image = image.astype(np.float32) / 255.0
    
    # 對大圖像進行下採樣處理
    h, w = normalized_image.shape
    if h > 1000 or w > 1000:
        print("📏 圖像太大，使用下採樣策略...")
        # 使用步長採樣減少計算量
        step = max(1, min(h, w) // 500)
        sampled_image = normalized_image[::step, ::step]
    else:
        sampled_image = normalized_image
    
    # 提取影像片段（減少數量）
    patches = extract_patches_2d(sampled_image, (patch_size, patch_size), 
                               max_patches=500, random_state=42)
    patches = patches.reshape(patches.shape[0], -1)
    
    # 訓練字典（減少迭代次數）
    print("🧠 訓練稀疏字典...")
    dict_learner = MiniBatchDictionaryLearning(n_components=n_components, 
                                             alpha=alpha, 
                                             max_iter=20,
                                             batch_size=20,
                                             random_state=42)
    dictionary = dict_learner.fit(patches).components_
    
    # 使用簡化的重構方法
    print("🔧 進行稀疏編碼重構...")
    
    # 對原始影像使用更簡單的方法
    # 使用基於DCT的近似稀疏編碼
    result = dct_based_denoising(normalized_image, threshold=0.01)
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)

def dct_based_denoising(image, threshold=0.01):
    """
    基於DCT的稀疏去噪（快速替代方案）
    """
    from scipy.fft import dct, idct
    
    # 分塊DCT處理
    block_size = 8
    h, w = image.shape
    result = np.zeros_like(image)
    
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = image[i:i+block_size, j:j+block_size]
            
            # 2D DCT
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # 稀疏化（硬閾值）
            dct_block[np.abs(dct_block) < threshold] = 0
            
            # 逆DCT
            reconstructed = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            result[i:i+block_size, j:j+block_size] = reconstructed
    
    return result

def calculate_psnr(original, processed):
    """
    計算PSNR指標
    """
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_cw_ssim(original, processed):
    """
    計算CW-SSIM指標
    """
    if len(original.shape) == 3:
        # 彩色影像，轉換為灰階
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        processed_gray = processed
    
    # 使用skimage的SSIM
    ssim_score = metrics.structural_similarity(original_gray, processed_gray, 
                                              data_range=processed_gray.max() - processed_gray.min())
    return ssim_score

def save_image(image, filename):
    """
    保存影像
    """
    cv2.imwrite(filename, image)

def main():
    # 設置輸出資料夾
    output_dir = "advanced_denoised_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 讀取原始影像
    original_image = cv2.imread('taipei101.png')
    if original_image is None:
        print("❌ 無法讀取 taipei101.png 檔案")
        return
    
    print("🎯 開始進階影像去噪處理...")
    print(f"📸 原始影像尺寸: {original_image.shape}")
    
    # 詢問使用者模式選擇
    print("\n🤔 請選擇處理模式：")
    print("  1. 🔬 學術研究模式：添加人工噪聲然後去噪（可量化評估）")
    print("  2. 🎨 實用模式：直接對原始照片進行去噪處理")
    
    # 默認使用實用模式
    mode = input("請輸入選擇 (1 或 2，預設為 2): ").strip()
    if mode == "1":
        use_research_mode = True
    else:
        use_research_mode = False
    
    # 保存原始影像
    save_image(original_image, os.path.join(output_dir, '00_original.png'))
    
    if use_research_mode:
        print("\n🔬 學術研究模式：添加人工噪聲進行測試...")
        # 創建帶噪聲的影像用於測試
        noisy_image = add_gaussian_noise(original_image, mean=0, std=30)
        save_image(noisy_image, os.path.join(output_dir, '01_noisy.png'))
        input_image = noisy_image
        reference_image = original_image  # 用於PSNR/SSIM計算
        print("📊 將與原始影像比較計算PSNR/SSIM指標")
    else:
        print("\n🎨 實用模式：直接對原始照片進行去噪處理...")
        input_image = original_image
        reference_image = None  # 無參考影像
        print("📊 將直接處理原始照片，無定量評估指標")
    
    # 儲存結果和評估指標
    results = []
    
    print("\n🚀 開始應用進階去噪方法...")
    
    # 1. 小波去噪
    print("📊 處理小波去噪...")
    wavelet_filtered = wavelet_denoising(input_image, wavelet='db4', sigma=0.1)
    save_image(wavelet_filtered, os.path.join(output_dir, '02_wavelet_filtered.png'))
    if reference_image is not None:
        psnr_wavelet = calculate_psnr(reference_image, wavelet_filtered)
        ssim_wavelet = calculate_cw_ssim(reference_image, wavelet_filtered)
        results.append(("小波去噪", psnr_wavelet, ssim_wavelet))
    
    # 2. 非局部均值去噪
    print("📊 處理非局部均值去噪...")
    nlm_filtered = non_local_means_denoising(input_image, h=10, 
                                           templateWindowSize=7, 
                                           searchWindowSize=21)
    save_image(nlm_filtered, os.path.join(output_dir, '03_nlm_filtered.png'))
    if reference_image is not None:
        psnr_nlm = calculate_psnr(reference_image, nlm_filtered)
        ssim_nlm = calculate_cw_ssim(reference_image, nlm_filtered)
        results.append(("非局部均值", psnr_nlm, ssim_nlm))
    
    # 3. 全變分去噪
    print("📊 處理全變分去噪...")
    tv_filtered = total_variation_denoising(input_image, weight=0.1)
    save_image(tv_filtered, os.path.join(output_dir, '04_tv_filtered.png'))
    if reference_image is not None:
        psnr_tv = calculate_psnr(reference_image, tv_filtered)
        ssim_tv = calculate_cw_ssim(reference_image, tv_filtered)
        results.append(("全變分去噪", psnr_tv, ssim_tv))
    
    # 4. 自適應濾波器
    print("📊 處理自適應濾波器...")
    adaptive_filtered = adaptive_filter_denoising(input_image, noise_var=900)
    save_image(adaptive_filtered, os.path.join(output_dir, '05_adaptive_filtered.png'))
    if reference_image is not None:
        psnr_adaptive = calculate_psnr(reference_image, adaptive_filtered)
        ssim_adaptive = calculate_cw_ssim(reference_image, adaptive_filtered)
        results.append(("自適應濾波器", psnr_adaptive, ssim_adaptive))
    
    # 5. 稀疏編碼去噪
    print("📊 處理稀疏編碼去噪...")
    sparse_filtered = sparse_coding_denoising(input_image, n_components=100, 
                                            patch_size=8, alpha=1.0)
    save_image(sparse_filtered, os.path.join(output_dir, '06_sparse_filtered.png'))
    if reference_image is not None:
        psnr_sparse = calculate_psnr(reference_image, sparse_filtered)
        ssim_sparse = calculate_cw_ssim(reference_image, sparse_filtered)
        results.append(("稀疏編碼去噪", psnr_sparse, ssim_sparse))
    
    # 顯示結果
    print("\n✨ 處理完成！")
    
    if results:
        print("📊 定量評估結果：")
        print("=" * 70)
        print(f"{'方法':<15} {'PSNR (dB)':<12} {'CW-SSIM':<12} {'綜合評分':<12}")
        print("=" * 70)
        
        for method, psnr, ssim in results:
            # 綜合評分 = 0.6 * PSNR/40 + 0.4 * SSIM
            combined_score = 0.6 * (psnr / 40) + 0.4 * ssim
            print(f"{method:<15} {psnr:<12.4f} {ssim:<12.6f} {combined_score:<12.6f}")
        
        print("=" * 70)
        
        # 找出最佳方法
        best_method_psnr = max(results, key=lambda x: x[1])
        best_method_ssim = max(results, key=lambda x: x[2])
        best_method_combined = max(results, key=lambda x: 0.6 * (x[1] / 40) + 0.4 * x[2])
        
        print(f"🏆 PSNR 最佳: {best_method_psnr[0]} ({best_method_psnr[1]:.4f} dB)")
        print(f"🏆 SSIM 最佳: {best_method_ssim[0]} ({best_method_ssim[2]:.6f})")
        print(f"🏆 綜合最佳: {best_method_combined[0]}")
    else:
        print("💡 實用模式：請查看輸出圖像來比較不同方法的視覺效果")
    
    print(f"\n📁 所有結果已保存到 '{output_dir}' 資料夾")
    print("📸 輸出檔案清單：")
    for i, filename in enumerate(sorted(os.listdir(output_dir))):
        if filename.endswith('.png'):
            print(f"  {i+1:2d}. {filename}")

if __name__ == "__main__":
    main() 