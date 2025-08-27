import cv2
import numpy as np

from dual_stream import tile_process_stable


def adaptive_idr_denoise(
    model,
    init_img,
    device,
    base_noise_level,
    gpu_optimizer,
    max_iter=3,
    tile_size=None,
    tile_schedule=None,
):
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
        tile_size (tuple[int, int] | None): 若未提供 `tile_schedule` 時的固定分塊尺寸
        tile_schedule (list[tuple[int, int] | None] | None): 各輪使用的 `tile_size`

    Returns:
        np.ndarray: 迭代後的去噪結果
    """
    current = init_img
    noise_level = base_noise_level * 0.8
    prev_var = None

    for i in range(max_iter):
        iter_tile = (
            tile_schedule[i] if tile_schedule and i < len(tile_schedule) else tile_size
        )
        refined = tile_process_stable(
            model,
            current,
            device,
            noise_level,
            gpu_optimizer,
            enable_attention=False,
            enable_region_adaptive=False,
            tile_size=iter_tile,
        )

        residual = cv2.absdiff(current, refined)
        residual_gray = cv2.cvtColor(residual, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(residual_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(residual_gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(sobelx, sobely)
        edge_mask = cv2.normalize(gradient, None, 0, 1.0, cv2.NORM_MINMAX)
        thresh = max(0.1, residual_gray.std() / 255.0)
        _, edge_mask = cv2.threshold(edge_mask, thresh, 1.0, cv2.THRESH_BINARY)
        edge_mask = cv2.morphologyEx(
            edge_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )

        weighted_residual = residual_gray * edge_mask
        local_var = cv2.Laplacian(weighted_residual, cv2.CV_32F).var()
        residual_std = weighted_residual.std() + 1e-6

        current = refined
        if local_var < residual_std * 0.1:
            break

        if prev_var is not None:
            ratio = local_var / prev_var if prev_var > 0 else 1.0
            noise_level *= max(0.5, min(0.9, ratio))
            if ratio > 0.95:
                break
        else:
            noise_level *= 0.8
        prev_var = local_var

    return current
def pyramid_idr_denoise(
    model,
    init_img,
    device,
    base_noise_level,
    gpu_optimizer,
    levels=2,
    tile_size=None,
    tile_schedule=None,
):
    """金字塔式自監督 IDR

    先於低解析度執行 :func:`adaptive_idr_denoise` 去除大範圍噪聲，
    再逐層上採樣與精修，可透過 ``levels`` 調整金字塔層數。

    Args:
        levels (int): 金字塔層數。
        tile_size (tuple[int, int] | None): 固定分塊尺寸 (若無排程)
        tile_schedule (list[tuple[int, int] | None] | None): 傳遞給 ``adaptive_idr_denoise`` 的分塊排程
    """
    pyramids = [init_img]
    for _ in range(1, levels):
        pyramids.append(cv2.pyrDown(pyramids[-1]))

    result = pyramids[-1]
    for lvl in reversed(range(levels)):
        result = adaptive_idr_denoise(
            model,
            result,
            device,
            base_noise_level,
            gpu_optimizer,
            tile_size=tile_size,
            tile_schedule=tile_schedule,
        )
        if lvl > 0:
            up_size = (pyramids[lvl - 1].shape[1], pyramids[lvl - 1].shape[0])
            result = cv2.pyrUp(result, dstsize=up_size)

    return result