import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models

# ===================== Fusion with Cross Attention + Class Embedding =====================

class CrossAttentionFusion(nn.Module):
    def __init__(self, latent_dim=4, embed_dim=256, num_heads=4, text_dim=768, dropout=0.1, num_layers=4, image_size=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.latent_proj = nn.Conv2d(latent_dim, embed_dim, kernel_size=1)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        latent_size = image_size // 8
        self.pos_embed = nn.Parameter(torch.randn(1, latent_size * latent_size, embed_dim))

        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_layers)
        ])

        self.out_proj = nn.Conv2d(embed_dim, latent_dim, kernel_size=1)

    def forward(self, latent, text_embeds):
        B, C, H, W = latent.shape

        x = self.latent_proj(latent)  # [B, embed_dim, H, W]

        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]

        x = x + self.pos_embed[:, :x.size(1), :]

        text_embed = self.text_proj(text_embeds)  # [B, T, C]
        kv = torch.cat([x, text_embed], dim=1)

        for layer in self.cross_attn_layers:
            attn_out, _ = layer['attn'](query=x, key=kv, value=kv)
            x = layer['norm1'](x + attn_out)
            x = layer['norm2'](x + layer['ffn'](x))

        fused_embed = x.transpose(1, 2).view(B, self.embed_dim, H, W).contiguous()
        fused_latent = self.out_proj(fused_embed)
        return fused_latent



class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        coords = torch.arange(kernel_size) - kernel_size // 2
        grid = coords.repeat(kernel_size).view(kernel_size, kernel_size)
        grid_x, grid_y = grid, grid.t()
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel[None, None, :, :])

    def forward(self, x):
        C = x.shape[1]
        kernel = self.kernel.repeat(C, 1, 1, 1)
        return F.conv2d(x, kernel, padding=self.kernel.shape[-1]//2, groups=C)

class LatentProjector(nn.Module):
    def __init__(self, in_channels=4, mid_channels=64, out_dim=768):
        super().__init__()
        self.blur = GaussianBlur(kernel_size=5, sigma=1.0)
        self.conv_down = nn.Conv2d(in_channels, mid_channels, kernel_size=4, stride=4)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(mid_channels * 4 * 4, out_dim),
            nn.LayerNorm(out_dim)
        )
    def forward(self, x):
        x = self.blur(x)                       # (B, 4, 64, 64)
        x = self.conv_down(x)                   # (B, 64, 16, 16)
        x = self.pool(x)                        # (B, 64, 4, 4)
        x = x.flatten(1)                        # (B, 1024)
        x = self.fc(x)                          # (B, 768)
        return x