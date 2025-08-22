import cv2
import numpy as np

from dual_stream import tile_process_stable


def adaptive_idr_denoise(model, init_img, device, base_noise_level, gpu_optimizer, max_iter=3,tile_size=None):
    """自監督式迭代降噪

    以 IDR 策略在殘差仍具結構時降低 noise_level 進行多輪精修。
    參考 Noise2Self/Noise2Void 概念，透過殘差的局部變異數與梯度強度判斷是否需再迭代。

    Args:
        model: 去噪模型
        init_img (np.ndarray): 初步去噪後的影像 (H,W,C)
        device: 運算裝置
        base_noise_level (float): 初始 noise_level
        gpu_optimizer: GPUOptimizer 物件
        max_iter (int): 最大迭代次數

    Returns:
        np.ndarray: 迭代後的去噪結果
    """
    current = init_img
    noise_level = base_noise_level * 0.8

    for _ in range(max_iter):
        refined = tile_process_stable(
            model, current, device, noise_level, gpu_optimizer,
            enable_attention=False,
            enable_region_adaptive=False,
            tile_size=tile_size,
        )

        residual = cv2.absdiff(current, refined)
        residual_gray = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)
        local_var = cv2.Laplacian(residual_gray, cv2.CV_32F).var()
        residual_std = residual_gray.std() + 1e-6

        if local_var < residual_std * 0.1:
            current = refined
            break

        current = refined
        noise_level *= 0.8

    return current