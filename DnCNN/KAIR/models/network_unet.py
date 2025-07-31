import torch
import torch.nn as nn
import models.basicblock as B
import numpy as np

'''
# ====================
# Residual U-Net
# ====================
citation:
@article{zhang2020plug,
title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
journal={arXiv preprint},
year={2020}
}
# ====================
'''

class NonLocalBlock(nn.Module):
    r"""
    Non-Local Self-Attention (Wang et al., 2018)
    ------------------------------------------------
    y_i = (1 / C(x)) * Σ_j softmax(θ(x_i)^T φ(x_j)) · g(x_j)
    z   = W(y) + x
    """
    def __init__(self, in_channels: int):
        super().__init__()
        inter_channels = max(1, in_channels // 2)
        self.g     = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.theta = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.phi   = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.W     = nn.Conv2d(inter_channels, in_channels, 1, bias=False)
        nn.init.zeros_(self.W.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, C, H, W = x.size()
        N = H * W
        g_x     = self.g(x).view(B_, -1, N).permute(0, 2, 1)
        theta_x = self.theta(x).view(B_, -1, N).permute(0, 2, 1)
        phi_x   = self.phi(x).view(B_, -1, N)
        affinity = torch.matmul(theta_x, phi_x)
        affinity = torch.softmax(affinity, dim=-1)
        y = torch.matmul(affinity, g_x)
        y = y.permute(0, 2, 1).contiguous().view(B_, -1, H, W)
        y = self.W(y)
        return x + y


class UNetRes(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True , use_nonlocal=True):
        super(UNetRes, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.non_local = NonLocalBlock(nc[3])
        
        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')

    def forward(self, x0):
#        h, w = x.size()[-2:]
#        paddingBottom = int(np.ceil(h/8)*8-h)
#        paddingRight = int(np.ceil(w/8)*8-w)
#        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x  = self.non_local(x)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)
#        x = x[..., :h, :w]

        return x


if __name__ == '__main__':
    x = torch.rand(1,3,256,256)
    net = UNetRes()
    net.eval()
    with torch.no_grad():
        y = net(x)
    print(y.size())

# run models/network_unet.py
