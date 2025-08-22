import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils import utils_image as util
from processing_utils import autocast_if_amp

try:
    from enhanced_feature_extraction import enhanced_feature_based_fusion
except ImportError:
    enhanced_feature_based_fusion = None


def stable_dual_stream_processing(model, img_L, device, base_noise_level, gpu_optimizer, enable_attention=False, enable_region_adaptive=False):
    """
    穩定雙流處理 - 專注於可靠性
    """
    try:
        img_tensor = util.uint2tensor4(img_L).to(device)
        gray = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < 50:
            sigma_low_ratio, sigma_high_ratio = 0.2, 1.2
        elif laplacian_var > 500:
            sigma_low_ratio, sigma_high_ratio = 0.4, 1.8
        else:
            sigma_low_ratio, sigma_high_ratio = 0.3, 1.5

        sigma_low = base_noise_level * sigma_low_ratio
        sigma_high = base_noise_level * sigma_high_ratio
        print(f"    自適應參數: σ_low={sigma_low:.1f}, σ_high={sigma_high:.1f} (Laplacian_var={laplacian_var:.1f})")

        noise_level_low = sigma_low / 255.0
        noise_tensor_low = torch.full((1, 1, img_tensor.shape[2], img_tensor.shape[3]), noise_level_low).to(device)
        img_input_low = torch.cat([img_tensor, noise_tensor_low], dim=1)
        with torch.no_grad():
            with autocast_if_amp(gpu_optimizer.use_amp):
                o_low = model(img_input_low)

        noise_level_high = sigma_high / 255.0
        noise_tensor_high = torch.full((1, 1, img_tensor.shape[2], img_tensor.shape[3]), noise_level_high).to(device)
        img_input_high = torch.cat([img_tensor, noise_tensor_high], dim=1)
        with torch.no_grad():
            with autocast_if_amp(gpu_optimizer.use_amp):
                o_high = model(img_input_high)

        try:
            if enable_region_adaptive:
                print(f"    使用區域自適應處理...")
                fused_np = _apply_region_adaptive_processing(model, img_L, device, base_noise_level, gpu_optimizer, enable_attention)
                fused_tensor = util.uint2tensor4(fused_np).to(device)
                print(f"    區域自適應處理完成")
            elif gpu_optimizer.enhanced_mode and gpu_optimizer.feature_extractor:
                print(f"    使用增強版特徵融合...")
                try:
                    o_low_np = util.tensor2uint(o_low)
                    o_high_np = util.tensor2uint(o_high)
                    fused_np, edge_weight = enhanced_feature_based_fusion(
                        o_low_np, o_high_np, img_L, base_noise_level
                    )
                    fused_tensor = util.uint2tensor4(fused_np).to(device)
                    print(f"    增強版融合完成")
                except Exception as enhanced_e:
                    print(f"    增強版融合失敗: {str(enhanced_e)[:50]}..., 使用傳統方法")
                    fused_tensor, edge_weight = _traditional_edge_fusion_tensor(o_low, o_high, img_tensor)
            else:
                fused_tensor, edge_weight = _traditional_edge_fusion_tensor(o_low, o_high, img_tensor)
        except Exception as processing_e:
            print(f"    處理失敗: {str(processing_e)[:50]}..., 使用傳統方法")
            fused_tensor, edge_weight = _traditional_edge_fusion_tensor(o_low, o_high, img_tensor)

        if enable_attention and gpu_optimizer.attention_module is not None:
            try:
                print(f"    應用Self-Attention增強...")
                with torch.no_grad():
                    with autocast_if_amp(gpu_optimizer.use_amp):
                        attn_enhanced = gpu_optimizer.attention_module(fused_tensor)
                final_tensor = attn_enhanced
                print(f"    Self-Attention處理成功")
                del attn_enhanced
            except Exception as e:
                print(f"     Self-Attention處理失敗: {str(e)[:50]}..., 使用原結果")
                final_tensor = fused_tensor
        else:
            final_tensor = fused_tensor

        final_result = util.tensor2uint(final_tensor)
        del img_tensor, img_input_low, img_input_high, o_low, o_high
        return final_result

    except torch.cuda.OutOfMemoryError as e:
        print(f"    雙流處理記憶體不足: {str(e)[:50]}...")
        raise
    except Exception as e:
        print(f"    雙流處理失敗: {str(e)}, 使用後備處理")
        return fallback_single_processing(model, img_L, device, base_noise_level, gpu_optimizer)


def tile_process_stable(model, img_L, device, base_noise_level, gpu_optimizer, enable_attention=False, enable_region_adaptive=False, tile_size=None):
    """
    穩定分塊處理，必要時遞迴縮小分塊以避免記憶體不足
    """
    h, w, c = img_L.shape
    if tile_size is None:
        tile_h, tile_w = gpu_optimizer.get_optimal_tile_size((h, w))
    else:
        tile_h, tile_w = tile_size
    if h <= tile_h and w <= tile_w:
        return stable_dual_stream_processing(model, img_L, device, base_noise_level, gpu_optimizer, enable_attention, enable_region_adaptive)

    print(f"  穩定分塊處理: {h}x{w} -> {tile_h}x{tile_w}")
    overlap = 64
    h_tiles = (h - 1) // (tile_h - overlap) + 1
    w_tiles = (w - 1) // (tile_w - overlap) + 1
    print(f"  分塊數量: {h_tiles}x{w_tiles} = {h_tiles * w_tiles} 塊")

    result_img = np.zeros_like(img_L)
    for i in range(h_tiles):
        for j in range(w_tiles):
            start_h = i * (tile_h - overlap)
            start_w = j * (tile_w - overlap)
            end_h = min(start_h + tile_h, h)
            end_w = min(start_w + tile_w, w)
            tile = img_L[start_h:end_h, start_w:end_w, :]
            print(f"    處理塊 ({i+1}/{h_tiles}, {j+1}/{w_tiles}) - 尺寸: {tile.shape}")
            try:
                tile_result = stable_dual_stream_processing(model, tile, device, base_noise_level, gpu_optimizer, enable_attention, enable_region_adaptive)
            except torch.cuda.OutOfMemoryError:
                if tile_h <= 64 or tile_w <= 64:
                    print("    分塊已達最小限制，回傳原始分塊")
                    tile_result = tile
                else:
                    new_tile_h = max(tile_h // 2, 64)
                    new_tile_w = max(tile_w // 2, 64)
                    print(f"    分塊處理記憶體不足，縮小至 {new_tile_h}x{new_tile_w}")
                    tile_result = tile_process_stable(
                        model, tile, device, base_noise_level, gpu_optimizer,
                        enable_attention, enable_region_adaptive, (new_tile_h, new_tile_w)
                    )
            if i == 0 and j == 0:
                result_img[start_h:end_h, start_w:end_w, :] = tile_result
            else:
                actual_start_h = start_h + (overlap // 2 if i > 0 else 0)
                actual_start_w = start_w + (overlap // 2 if j > 0 else 0)
                tile_start_h = overlap // 2 if i > 0 else 0
                tile_start_w = overlap // 2 if j > 0 else 0
                result_img[actual_start_h:end_h, actual_start_w:end_w, :] = tile_result[tile_start_h:, tile_start_w:, :]
    gpu_optimizer.cleanup_memory()
    return result_img


def fallback_single_processing(model, img_L, device, base_noise_level, gpu_optimizer, tile_size=None):
    """後備單流處理方法"""
    try:
        print(f"    使用後備單流處理 (noise_level={base_noise_level})")
        img_tensor = util.uint2tensor4(img_L).to(device)
        noise_level_normalized = base_noise_level / 255.0
        noise_tensor = torch.full((1, 1, img_tensor.shape[2], img_tensor.shape[3]), noise_level_normalized).to(device)
        img_input = torch.cat([img_tensor, noise_tensor], dim=1)
        with torch.no_grad():
            with autocast_if_amp(gpu_optimizer.use_amp):
                img_output = model(img_input)
        result = util.tensor2uint(img_output)
        del img_tensor, img_input, img_output
        gpu_optimizer.cleanup_memory()
        return result
    except torch.cuda.OutOfMemoryError:
        print("    單流處理記憶體不足，改用分塊處理")
        return tile_process_stable(
            model, img_L, device, base_noise_level, gpu_optimizer, tile_size=tile_size
        )
    except Exception as e:
        print(f"    後備處理也失敗: {str(e)}, 返回原圖")
        return img_L


def _traditional_edge_fusion_tensor(o_low, o_high, img_tensor):
    """使用Torch實作的傳統邊緣融合"""
    gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
    kernel3 = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    edge3 = torch.abs(F.conv2d(gray, kernel3, padding=1))
    edge5 = torch.abs(F.conv2d(gray, kernel3, padding=2, dilation=2))
    combined_edge = 0.7 * edge3 + 0.3 * edge5
    tau = torch.quantile(combined_edge, 0.30)
    M = torch.quantile(combined_edge, 0.90)
    edge_weight = ((combined_edge - tau) / (M - tau + 1e-6)).clamp(0, 1)
    edge_weight = F.avg_pool2d(edge_weight, kernel_size=3, stride=1, padding=1)
    fused = edge_weight * o_low + (1 - edge_weight) * o_high
    return fused, edge_weight


def _apply_region_adaptive_processing(model, img_L, device, base_noise_level, gpu_optimizer, enable_attention):
    """
    區域自適應處理 - 解決去噪效果不均勻問題
    """
    print(f"      啟用區域自適應分析...")
    gray = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    kernel = np.ones((7, 7), np.float32) / 49
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    texture_complexity = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    texture_threshold = np.percentile(texture_complexity, 60)
    edge_threshold = np.percentile(edge_strength, 60)
    region_map = np.zeros((h, w), dtype=np.uint8)
    smooth_mask = (texture_complexity < texture_threshold) & (edge_strength < edge_threshold)
    region_map[smooth_mask] = 0
    detail_mask = (texture_complexity >= texture_threshold) & (edge_strength >= edge_threshold)
    region_map[detail_mask] = 1
    edge_mask = (edge_strength >= edge_threshold) & (texture_complexity < texture_threshold)
    region_map[edge_mask] = 2
    total_pixels = h * w
    smooth_ratio = np.sum(region_map == 0) / total_pixels * 100
    detail_ratio = np.sum(region_map == 1) / total_pixels * 100
    edge_ratio = np.sum(region_map == 2) / total_pixels * 100
    print(f"        區域分布: 平滑={smooth_ratio:.1f}%, 細節={detail_ratio:.1f}%, 邊緣={edge_ratio:.1f}%")
    img_tensor = util.uint2tensor4(img_L).to(device)
    region_results = []
    region_configs = [
        {'name': '平滑區域', 'sigma_low_ratio': 0.15, 'sigma_high_ratio': 0.9},
        {'name': '細節區域', 'sigma_low_ratio': 0.4, 'sigma_high_ratio': 2.0},
        {'name': '邊緣區域', 'sigma_low_ratio': 0.35, 'sigma_high_ratio': 1.6}
    ]
    for region_id, config in enumerate(region_configs):
        region_mask = region_map == region_id
        pixel_count = np.sum(region_mask)
        if pixel_count < total_pixels * 0.01:
            continue
        sigma_low = base_noise_level * config['sigma_low_ratio']
        sigma_high = base_noise_level * config['sigma_high_ratio']
        noise_level_low = sigma_low / 255.0
        noise_tensor_low = torch.full((1, 1, img_tensor.shape[2], img_tensor.shape[3]), noise_level_low).to(device)
        img_input_low = torch.cat([img_tensor, noise_tensor_low], dim=1)
        noise_level_high = sigma_high / 255.0
        noise_tensor_high = torch.full((1, 1, img_tensor.shape[2], img_tensor.shape[3]), noise_level_high).to(device)
        img_input_high = torch.cat([img_tensor, noise_tensor_high], dim=1)
        with torch.no_grad():
            with autocast_if_amp(gpu_optimizer.use_amp):
                o_low = model(img_input_low)
                o_high = model(img_input_high)
        o_low_np = util.tensor2uint(o_low)
        o_high_np = util.tensor2uint(o_high)
        if len(o_low_np.shape) == 2:
            o_low_np = o_low_np[:, :, np.newaxis]
        if len(o_high_np.shape) == 2:
            o_high_np = o_high_np[:, :, np.newaxis]
        region_results.append({'name': config['name'], 'mask': region_mask, 'o_low': o_low_np, 'o_high': o_high_np})
    print(f"        執行區域融合...")
    final_result = np.zeros_like(img_L, dtype=np.float32)
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)
    for result in region_results:
        region_mask = result['mask']
        region_edge_weight = _compute_region_edge_weights(gray, region_mask, result['name'])
        edge_weight_3d = region_edge_weight[..., np.newaxis]
        region_fused = edge_weight_3d * result['o_low'].astype(np.float32) + (1 - edge_weight_3d) * result['o_high'].astype(np.float32)
        region_mask_3d = region_mask[..., np.newaxis].astype(np.float32)
        final_result += region_fused * region_mask_3d
        weight_sum += region_mask_3d
    weight_sum = np.maximum(weight_sum, 1e-8)
    final_result = final_result / weight_sum
    return np.clip(final_result, 0, 255).astype(np.uint8)


def _compute_region_edge_weights(gray, region_mask, region_name):
    """計算區域特定的邊緣權重"""
    masked_gray = gray.copy().astype(np.float32)
    if np.sum(region_mask) > 0:
        masked_gray[~region_mask] = np.mean(masked_gray[region_mask])
    if '細節' in region_name:
        edge1 = cv2.Laplacian(masked_gray, cv2.CV_32F, ksize=3)
        edge2 = cv2.Laplacian(masked_gray, cv2.CV_32F, ksize=5)
        edge3 = cv2.Laplacian(masked_gray, cv2.CV_32F, ksize=7)
        combined_edge = 0.5 * np.abs(edge1) + 0.3 * np.abs(edge2) + 0.2 * np.abs(edge3)
        blur_kernel = (3, 3)
        blur_sigma = 0.3
    elif '平滑' in region_name:
        edge1 = cv2.Laplacian(masked_gray, cv2.CV_32F, ksize=3)
        edge2 = cv2.Laplacian(masked_gray, cv2.CV_32F, ksize=5)
        combined_edge = 0.7 * np.abs(edge1) + 0.3 * np.abs(edge2)
        blur_kernel = (7, 7)
        blur_sigma = 1.0
    else:
        edge1 = cv2.Laplacian(masked_gray, cv2.CV_32F, ksize=3)
        edge2 = cv2.Laplacian(masked_gray, cv2.CV_32F, ksize=5)
        combined_edge = 0.7 * np.abs(edge1) + 0.3 * np.abs(edge2)
        blur_kernel = (5, 5)
        blur_sigma = 0.5
    if np.sum(region_mask) > 0:
        region_edges = combined_edge[region_mask]
        if '細節' in region_name:
            tau = np.percentile(region_edges, 20)
            M = np.percentile(region_edges, 90)
        elif '平滑' in region_name:
            tau = np.percentile(region_edges, 35)
            M = np.percentile(region_edges, 75)
        else:
            tau = np.percentile(region_edges, 25)
            M = np.percentile(region_edges, 85)
    else:
        tau = np.mean(combined_edge) * 0.5
        M = np.mean(combined_edge) * 2.0
    if M - tau < 1e-6:
        M = tau + 1e-6
    edge_weight = np.clip((combined_edge - tau) / (M - tau), 0, 1)
    edge_weight = cv2.GaussianBlur(edge_weight, blur_kernel, blur_sigma)
    edge_weight[~region_mask] = 0
    return edge_weight