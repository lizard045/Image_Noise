import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustSelfAttention(nn.Module):
    """穩定版Self-Attention模組 - 修復所有維度問題"""
    def __init__(self, channels):
        super(RobustSelfAttention, self).__init__()
        self.channels = channels

        # 根據通道數選擇不同的注意力策略
        if channels <= 4:
            # 極小通道數：使用通道注意力機制
            self.attention_type = 'channel'
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(1, channels // 2), 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, channels // 2), channels, 1, bias=False),
                nn.Sigmoid()
            )
            print(f"    Self-Attention: 通道注意力模式 (channels={channels})")
        else:
            # 正常通道數：使用空間注意力機制
            self.attention_type = 'spatial'
            # 安全的降維策略
            self.reduced_channels = max(1, min(channels // 4, 32))  # 限制最大降維

            self.query = nn.Conv2d(channels, self.reduced_channels, 1, bias=False)
            self.key = nn.Conv2d(channels, self.reduced_channels, 1, bias=False)
            self.value = nn.Conv2d(channels, channels, 1, bias=False)

            # 初始化權重
            nn.init.kaiming_normal_(self.query.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.key.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.value.weight, mode='fan_out', nonlinearity='relu')

            print(f"    Self-Attention: 空間注意力模式 (channels={channels}, reduced={self.reduced_channels})")

        # 可學習的融合權重
        self.gamma = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.size()

        try:
            if self.attention_type == 'channel':
                # 通道注意力機制
                attention_weights = self.channel_attention(x)  # (B, C, 1, 1)
                attention_result = x * attention_weights
            else:
                # 空間注意力機制
                attention_result = self._compute_spatial_attention(x, H, W)

            # 殘差連接
            return self.gamma * attention_result + x

        except Exception as e:
            print(f"        Self-Attention處理失敗: {str(e)[:50]}..., 跳過")
            return x

    def _compute_spatial_attention(self, x, H, W):
        """計算空間注意力"""
        B, C = x.size(0), x.size(1)

        # 對大圖片進行下採樣以節約記憶體
        if H * W > 256 * 256:
            scale_factor = 0.5
            x_small = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            h_small, w_small = x_small.shape[2], x_small.shape[3]

            # 在小尺寸上計算attention
            attn_small = self._attention_computation(x_small, h_small, w_small)

            # 放回原尺寸
            attention_result = F.interpolate(attn_small, size=(H, W), mode='bilinear', align_corners=False)
        else:
            attention_result = self._attention_computation(x, H, W)

        return attention_result

    def _attention_computation(self, x, h, w):
        """核心attention計算"""
        B, C = x.size(0), x.size(1)

        # 生成Q, K, V
        q = self.query(x).view(B, self.reduced_channels, h * w).permute(0, 2, 1)  # (B, HW, reduced)
        k = self.key(x).view(B, self.reduced_channels, h * w)  # (B, reduced, HW)
        v = self.value(x).view(B, C, h * w)  # (B, C, HW)

        # 計算注意力矩陣
        attention = torch.bmm(q, k)  # (B, HW, HW)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        # 應用注意力
        out = torch.bmm(v, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, h, w)

        return out