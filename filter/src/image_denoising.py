import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, metrics, filters
from scipy.signal import wiener
import os
from PIL import Image

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

def add_gaussian_noise(image, mean=0, std=25):
    """
    添加高斯噪聲到影像
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def average_filter(image, kernel_size=5):
    """
    平均濾波器
    """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)

def median_filter(image, kernel_size=5):
    """
    中值濾波器
    """
    return cv2.medianBlur(image, kernel_size)

def gaussian_filter(image, kernel_size=5, sigma=1.0):
    """
    高斯濾波器
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    雙邊濾波器
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def wiener_filter(image, noise_var=0.01):
    """
    維納濾波器
    """
    if len(image.shape) == 3:
        # 彩色影像，對每個通道分別處理
        result = np.zeros_like(image)
        # 創建一個簡單的點擴散函數（PSF）
        psf = np.ones((5, 5)) / 25
        for i in range(3):
            result[:, :, i] = restoration.wiener(image[:, :, i], psf, noise_var)
        return (result * 255).astype(np.uint8)
    else:
        # 灰階影像
        psf = np.ones((5, 5)) / 25
        result = restoration.wiener(image, psf, noise_var)
        return (result * 255).astype(np.uint8)

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
    output_dir = "denoised_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 讀取原始影像
    original_image = cv2.imread('taipei101.png')
    if original_image is None:
        print("❌ 無法讀取 taipei101.png 檔案")
        return
    
    print("🎯 開始影像去噪處理...")
    print(f"📸 原始影像尺寸: {original_image.shape}")
    
    # 保存原始影像
    save_image(original_image, os.path.join(output_dir, '00_original.png'))
    
    # 創建帶噪聲的影像用於測試
    # 1. 椒鹽噪聲版本（專門用於中值濾波器測試）
    salt_pepper_noisy = add_salt_pepper_noise(original_image, salt_prob=0.02, pepper_prob=0.02)
    save_image(salt_pepper_noisy, os.path.join(output_dir, '01_salt_pepper_noisy.png'))
    
    # 2. 高斯噪聲版本（用於其他濾波器測試）
    gaussian_noisy = add_gaussian_noise(original_image, mean=0, std=25)
    save_image(gaussian_noisy, os.path.join(output_dir, '02_gaussian_noisy.png'))
    
    # 儲存結果和評估指標
    results = []
    
    print("\n🔧 開始應用各種濾波器...")
    
    # 1. 平均濾波器
    print("📊 處理平均濾波器...")
    avg_filtered = average_filter(gaussian_noisy, kernel_size=5)
    save_image(avg_filtered, os.path.join(output_dir, '03_average_filtered.png'))
    ssim_avg = calculate_cw_ssim(original_image, avg_filtered)
    results.append(("平均濾波器", ssim_avg))
    
    # 2. 中值濾波器（使用椒鹽噪聲影像）
    print("📊 處理中值濾波器...")
    median_filtered = median_filter(salt_pepper_noisy, kernel_size=5)
    save_image(median_filtered, os.path.join(output_dir, '04_median_filtered.png'))
    ssim_median = calculate_cw_ssim(original_image, median_filtered)
    results.append(("中值濾波器", ssim_median))
    
    # 3. 高斯濾波器
    print("📊 處理高斯濾波器...")
    gauss_filtered = gaussian_filter(gaussian_noisy, kernel_size=5, sigma=1.0)
    save_image(gauss_filtered, os.path.join(output_dir, '05_gaussian_filtered.png'))
    ssim_gauss = calculate_cw_ssim(original_image, gauss_filtered)
    results.append(("高斯濾波器", ssim_gauss))
    
    # 4. 雙邊濾波器
    print("📊 處理雙邊濾波器...")
    bilateral_filtered = bilateral_filter(gaussian_noisy, d=9, sigma_color=75, sigma_space=75)
    save_image(bilateral_filtered, os.path.join(output_dir, '06_bilateral_filtered.png'))
    ssim_bilateral = calculate_cw_ssim(original_image, bilateral_filtered)
    results.append(("雙邊濾波器", ssim_bilateral))
    
    # 5. 維納濾波器
    print("📊 處理維納濾波器...")
    # 正規化影像到0-1範圍
    normalized_noisy = gaussian_noisy.astype(np.float32) / 255.0
    wiener_filtered = wiener_filter(normalized_noisy, noise_var=0.01)
    save_image(wiener_filtered, os.path.join(output_dir, '07_wiener_filtered.png'))
    ssim_wiener = calculate_cw_ssim(original_image, wiener_filtered)
    results.append(("維納濾波器", ssim_wiener))
    
    # 顯示結果
    print("\n✨ 處理完成！結果統計：")
    print("=" * 50)
    print(f"{'濾波器方法':<15} {'CW-SSIM 分數':<15}")
    print("=" * 50)
    
    for method, ssim_score in results:
        print(f"{method:<15} {ssim_score:.6f}")
    
    print("=" * 50)
    
    # 找出最佳方法
    best_method, best_score = max(results, key=lambda x: x[1])
    print(f"🏆 最佳方法: {best_method} (CW-SSIM: {best_score:.6f})")
    
    print(f"\n📁 所有結果已保存到 '{output_dir}' 資料夾")
    print("📸 輸出檔案清單：")
    for i, filename in enumerate(os.listdir(output_dir)):
        if filename.endswith('.png'):
            print(f"  {i+1:2d}. {filename}")

if __name__ == "__main__":
    main() 