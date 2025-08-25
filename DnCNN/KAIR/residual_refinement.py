import cv2
import numpy as np

from dual_stream import tile_process_stable


def _refine_residual_noise_hotspots(model, original_img, current_result, device, base_noise_level, gpu_optimizer, tile_size=None):
    """全圖殘留噪點熱區精修"""
    if len(original_img.shape) == 2:
        base_bgr = original_img[:, :, np.newaxis]
        gray = original_img
    else:
        base_bgr = original_img
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if h < 200 or w < 200:
        return current_result
    cy, cx = h / 2.0, w / 2.0
    Y, X = np.ogrid[:h, :w]
    r = min(h, w) * 0.48
    vignette_mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= (r ** 2)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    highpass = cv2.absdiff(gray, blur).astype(np.float32)
    k = np.ones((7, 7), np.float32) / 49.0
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, k)
    local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, k)
    local_std = np.sqrt(np.maximum(local_var, 0))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx ** 2 + gy ** 2)
    eps = 1e-6
    noise_heat = 0.6 * (highpass / (edge + eps)) + 0.4 * (local_std / (np.mean(local_std[vignette_mask]) + eps))
    noise_heat = np.clip(noise_heat, 0, np.percentile(noise_heat, 99))
    noise_heat = noise_heat / (np.max(noise_heat) + eps)
    bright_mask = gray > 235
    noise_heat[bright_mask] *= 0.3
    trigger_score = np.percentile(noise_heat[vignette_mask], 92)
    if trigger_score < 0.45:
        return current_result
    print(f"    偵測到殘留噪點熱區，啟用精修 (score={trigger_score:.2f})")
    sigma_refine = np.clip(base_noise_level * 2.2, 0, 255)
    refined_np = tile_process_stable(
        model, base_bgr, device, sigma_refine, gpu_optimizer,
        tile_size=tile_size,
    )
    weight = np.zeros((h, w), dtype=np.float32)
    heat_valid = noise_heat * vignette_mask.astype(np.float32)
    thresh = max(0.35, float(cv2.threshold((heat_valid * 255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU)[0]) / 255.0)
    weight[heat_valid >= thresh] = heat_valid[heat_valid >= thresh]
    weight[bright_mask] = 0
    weight = cv2.GaussianBlur(weight, (31, 31), 7.0)
    weight = np.clip(weight, 0.0, 1.0)
    weight_3d = weight[..., np.newaxis]
    refined = (weight_3d * refined_np.astype(np.float32) + (1 - weight_3d) * current_result.astype(np.float32))
    refined = np.clip(refined, 0, 255).astype(np.uint8)
    mask_soft = (weight > 0.02).astype(np.uint8) * 255
    mask_soft = cv2.GaussianBlur(mask_soft, (21, 21), 5.0)
    mask_soft = (mask_soft.astype(np.float32) / 255.0)[..., np.newaxis]
    output = mask_soft * refined.astype(np.float32) + (1 - mask_soft) * current_result.astype(np.float32)
    return np.clip(output, 0, 255).astype(np.uint8)


def _residual_bilateral_denoise(original_img, current_result, sigma_color=30, sigma_space=7):
    """殘差域雙邊濾波"""
    residual = cv2.absdiff(original_img, current_result)
    if residual.ndim == 3:
        residual_gray = cv2.cvtColor(residual, cv2.COLOR_RGB2GRAY)
    else:
        residual_gray = residual
    norm_residual = residual_gray.astype(np.float32)
    norm_residual = norm_residual / (np.max(norm_residual) + 1e-6)
    norm_residual = cv2.GaussianBlur(norm_residual, (0, 0), 1.5)
    filtered = cv2.bilateralFilter(current_result, d=5, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    weight = norm_residual[..., np.newaxis]
    bright_mask = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY) > 235
    weight[bright_mask[..., np.newaxis]] = 0
    blended = weight * filtered.astype(np.float32) + (1 - weight) * current_result.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)