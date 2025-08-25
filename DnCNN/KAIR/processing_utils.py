import numpy as np
import cv2

try:
    from torch.amp import autocast as _autocast_base
    def autocast_if_amp(enabled):
        return _autocast_base('cuda', enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _autocast_base
    def autocast_if_amp(enabled):
        return _autocast_base(enabled=enabled)


def _prepare_image(img):
    """確保影像為(H,W,C)格式"""
    if len(img.shape) == 2:
        return img[:, :, np.newaxis]
    if len(img.shape) == 4:
        return img[0]
    return img

def estimate_noise_level(img, max_sigma=25.0, texture_threshold=0.02):
    """估計輸入影像的雜訊強度 (0~255)並避免因細節過多而高估

    Args:
        img: 輸入影像，支援 (H,W,C) 或 (H,W) 格式。
        max_sigma: 雜訊估計的上限，避免在紋理豐富的影像中被高估。
        texture_threshold: 拉普拉斯變異數閾值，超過表示細節豐富需限制估計值。

    Returns:
        float: 預估的雜訊強度。
    """
    img = _prepare_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    noise = gray - blur
    sigma = np.std(noise) * 255.0
     # 紋理檢測：拉普拉斯變異數越高代表細節越多
    lap_var = cv2.Laplacian(gray, cv2.CV_32F).var()
    if lap_var > texture_threshold:
        sigma = min(sigma, max_sigma)
    return float(sigma)
