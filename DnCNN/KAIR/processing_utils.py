import numpy as np

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
