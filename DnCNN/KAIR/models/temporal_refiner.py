import torch
import torch.nn as nn
from .network_rvrt import SpyNet, flow_warp
from utils import utils_image as util

class TemporalRefiner(nn.Module):
    """簡易時域精修模組：使用光流對齊後平均多幀結果"""
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.spynet = SpyNet(return_levels=[5]).to(self.device)
        self.spynet.eval()
        for p in self.spynet.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def compute_flow(self, frames):
        n, t, c, h, w = frames.size()
        lqs1 = frames[:, :-1].reshape(-1, c, h, w)
        lqs2 = frames[:, 1:].reshape(-1, c, h, w)
        flows_forward = self.spynet(lqs2, lqs1).view(n, t - 1, 2, h, w)
        flows_backward = self.spynet(lqs1, lqs2).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    @torch.no_grad()
    def forward(self, frame_list):
        """輸入多張影像，輸出時域平均後的去噪結果"""
        if len(frame_list) == 1:
            return frame_list[0]
        tensor_list = [util.uint2tensor4(f).to(self.device) for f in frame_list]
        lqs = torch.stack(tensor_list, dim=1)  # (1,T,C,H,W)
        flows_f, flows_b = self.compute_flow(lqs)
        n, t, c, h, w = lqs.size()
        center_idx = t // 2
        center = lqs[:, center_idx]
        aligned = [center]
        for i in range(t):
            if i == center_idx:
                continue
            if i < center_idx:
                flow = flows_f[:, i]
            else:
                flow = flows_b[:, i - 1]
            warped = flow_warp(lqs[:, i], flow.permute(0, 2, 3, 1))
            aligned.append(warped)
        stacked = torch.stack(aligned, dim=1)
        refined = stacked.mean(dim=1)
        return util.tensor2uint(refined)