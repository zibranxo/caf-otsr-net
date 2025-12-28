import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import math

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== EXACT MODEL ARCHITECTURE FROM NOTEBOOK ==========

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.activation(out)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        y = self.gap(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y

class LaplacianBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = self.residual(x)
        residual = self.upsample(residual)
        
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = x + residual
        
        return x

class EnhancedOTAlign(nn.Module):
    def __init__(self, optical_ch=11, thermal_ch=1):
        super().__init__()
        
        # Stage 1: Affine Transformation Predictor
        self.affine_net = nn.Sequential(
            nn.Conv2d(optical_ch + thermal_ch, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 6)
        )
        
        self.affine_net[-1].weight.data.zero_()
        self.affine_net[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        # Stage 2: Deformable Adjustment Network (NOTE: deformable_net not deform_net)
        self.deformable_net = nn.Sequential(
            nn.Conv2d(optical_ch + thermal_ch, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2, 3, padding=1),
            nn.Tanh()
        )
        
        # Learnable scaling factor for deformable offsets
        self.offset_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, optical, thermal):
        B, C, H, W = optical.shape
        
        # Stage 1: Affine Coarse Alignment
        affine_input = torch.cat([optical, thermal], dim=1)
        affine_params = self.affine_net(affine_input).view(-1, 2, 3)
        
        grid_affine = F.affine_grid(affine_params, optical.size(), align_corners=False)
        aligned_coarse = F.grid_sample(optical, grid_affine, align_corners=False, padding_mode='border')
        
        # Stage 2: Deformable Fine Adjustment
        deformable_input = torch.cat([aligned_coarse, thermal], dim=1)
        deformable_offsets = self.deformable_net(deformable_input) * self.offset_scale
        
        # Create coordinate grid
        coord_x, coord_y = torch.meshgrid(
            torch.linspace(-1, 1, W, device=optical.device),
            torch.linspace(-1, 1, H, device=optical.device),
            indexing='xy'
        )
        coord_x = coord_x.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        coord_y = coord_y.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        
        offset_x = deformable_offsets[:, 0:1, :, :]
        offset_y = deformable_offsets[:, 1:2, :, :]
        
        grid_deformable = torch.cat([
            coord_x + offset_x,
            coord_y + offset_y
        ], dim=1).permute(0, 2, 3, 1)
        
        aligned_final = F.grid_sample(aligned_coarse, grid_deformable, align_corners=False, padding_mode='border')
        
        return aligned_final, {
            "affine": affine_params,
            "deformable_offsets": deformable_offsets,
            "offset_scale": self.offset_scale
        }

class ThermalEncoder(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32)
        )
        self.block1 = ResidualBlock(32, 32, stride=1)
        self.block2 = ResidualBlock(32, 64, stride=2)
        self.block3 = ResidualBlock(64, 64, stride=2)
    
    def forward(self, x):
        x = self.initial(x)
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        return [f1, f2, f3]

class OpticalEncoder(nn.Module):
    def __init__(self, in_ch=11):
        super().__init__()
        self.vis_nir = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32)
        )
        self.swir = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(16)
        )
        self.indices = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(16)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            ChannelAttention(64)
        )
        self.down1 = ResidualBlock(64, 64, stride=1)
        self.down2 = ResidualBlock(64, 64, stride=2)
        self.down3 = ResidualBlock(64, 64, stride=2)
    
    def forward(self, x):
        vis_features = self.vis_nir(x[:, 0:6])
        swir_features = self.swir(x[:, 6:9])
        indices_features = self.indices(x[:, 9:11])
        combined = torch.cat([vis_features, swir_features, indices_features], dim=1)
        fused = self.fusion(combined)
        f1 = self.down1(fused)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        return [f1, f2, f3]

class TextureEncoder(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        return self.net(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        
        self.gate_net = nn.Sequential(
            nn.Conv2d(embed_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.attn_drop = nn.Dropout(0.1)
    
    def forward(self, thermal_feats, optical_feats):
        B, C, H, W = thermal_feats.shape
        
        Q = self.q_proj(thermal_feats).view(B, self.num_heads, self.head_dim, H * W)
        K = self.k_proj(optical_feats).view(B, self.num_heads, self.head_dim, H * W)
        V = self.v_proj(optical_feats).view(B, self.num_heads, self.head_dim, H * W)
        
        attn_weights = torch.matmul(Q.transpose(-1, -2), K) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        attended = torch.matmul(V, attn_weights.transpose(-1, -2))
        attended = attended.contiguous().view(B, C, H, W)
        attended = self.out_proj(attended)
        
        gate = self.gate_net(thermal_feats)
        output = thermal_feats + gate * attended
        
        return output, attn_weights

class TextureGuidance(nn.Module):
    def __init__(self, texture_ch=32, fused_ch=64):
        super().__init__()
        self.texture_expand = nn.Conv2d(texture_ch, fused_ch, 1)
        self.fusion_conv = nn.Conv2d(fused_ch * 2, fused_ch, 3, padding=1)
    
    def forward(self, fused_feats, texture_feats):
        B, _, H, W = fused_feats.shape
        texture_expanded = self.texture_expand(texture_feats).repeat(1, 1, H, W)
        combined = torch.cat([fused_feats, texture_expanded], dim=1)
        return self.fusion_conv(combined)

class TextureSafetyNet(nn.Module):
    def __init__(self, fused_ch=64):
        super().__init__()
        self.safety_net = nn.Sequential(
            nn.Conv2d(fused_ch, 32, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, fused_features):
        return self.safety_net(fused_features)

class ProgressiveLaplacianDecoder(nn.Module):
    def __init__(self, in_ch=64, scale_factor=2, use_4x=False):
        super().__init__()
        self.use_4x = use_4x
        self.level1 = LaplacianBlock(in_ch, 64, scale_factor=2)
        self.level2 = LaplacianBlock(64, 32, scale_factor=2)
        if use_4x:
            self.level3 = LaplacianBlock(32, 16, scale_factor=2)
    
    def forward(self, features, thermal_feats):
        x = self.level1(features)
        if len(thermal_feats) > 1:
            x = x + thermal_feats[1]
        x = self.level2(x)
        if len(thermal_feats) > 0:
            x = x + thermal_feats[0]
        if self.use_4x and hasattr(self, 'level3'):
            x = self.level3(x)
        return x

class SRHead(nn.Module):
    def __init__(self, decoder_ch=32, out_ch=1):
        super().__init__()
        self.final_conv = nn.Conv2d(decoder_ch, out_ch, 3, padding=1)
    
    def forward(self, x):
        return self.final_conv(x)

class EnhancedUncertaintyHead(nn.Module):
    def __init__(self, texture_ch=32, thermal_ch=64, fused_ch=64):
        super().__init__()
        self.uncertainty_net = nn.Sequential(
            nn.Conv2d(texture_ch + thermal_ch + fused_ch, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Softplus()
        )
    
    def forward(self, texture_feats, thermal_feats, fused_feats):
        B, _, H, W = fused_feats.shape
        texture_expanded = texture_feats.repeat(1, 1, H, W)
        combined = torch.cat([texture_expanded, thermal_feats, fused_feats], dim=1)
        return self.uncertainty_net(combined)

class EnhancedCAF_OTSRNet(nn.Module):
    def __init__(self, scale_factor=2, use_4x=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.use_4x = use_4x
        
        self.otalign = EnhancedOTAlign(optical_ch=11, thermal_ch=1)
        self.thermal_encoder = ThermalEncoder(in_ch=1)
        self.optical_encoder = OpticalEncoder(in_ch=11)
        self.texture_encoder = TextureEncoder(in_ch=2)
        self.cross_attention = CrossAttentionFusion(embed_dim=64, num_heads=8)
        self.texture_guidance = TextureGuidance(texture_ch=32, fused_ch=64)
        self.safety_net = TextureSafetyNet(fused_ch=64)
        self.decoder = ProgressiveLaplacianDecoder(in_ch=64, scale_factor=scale_factor, use_4x=use_4x)
        self.sr_head = SRHead(decoder_ch=32, out_ch=1)
        self.uncertainty_head = EnhancedUncertaintyHead(texture_ch=32, thermal_ch=64, fused_ch=64)
    
    def forward(self, x):
        thermal_lr = x[:, 0:1]
        optical_input = x[:, 2:13]
        texture_input = x[:, 13:15]
        
        aligned_optical, align_params = self.otalign(optical_input, thermal_lr)
        thermal_feats = self.thermal_encoder(thermal_lr)
        optical_feats = self.optical_encoder(aligned_optical)
        texture_feats = self.texture_encoder(texture_input)
        
        attention_fused, attn_maps = self.cross_attention(thermal_feats[2], optical_feats[2])
        texture_guided = self.texture_guidance(attention_fused, texture_feats)
        safety_scores = self.safety_net(texture_guided)
        safe_features = texture_guided * safety_scores
        
        decoder_features = self.decoder(safe_features, thermal_feats)
        sr_residual = self.sr_head(decoder_features)
        sr_output = thermal_lr + sr_residual
        
        uncertainty = self.uncertainty_head(texture_feats, thermal_feats[2], safe_features)
        
        return {
            'sr_output': sr_output,
            'uncertainty': uncertainty,
            'safety_scores': safety_scores,
            'attention_maps': attn_maps,
            'align_params': align_params,
            'sr_residual': sr_residual
        }

# Load model
print("Loading model...")
model = EnhancedCAF_OTSRNet(scale_factor=2, use_4x=False)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")

def process_npz(npz_file):
    try:
        data = np.load(npz_file.name)
        
        if 'data' in data:
            patch_data = data['data']
        elif 'array' in data:
            patch_data = data['array']
        elif 'patch' in data:
            patch_data = data['patch']
        else:
            first_key = list(data.keys())[0]
            patch_data = data[first_key]
        
        data.close()
        
        if patch_data.shape[0] == 16:
            patch_data = patch_data[:15]
        
        patch_tensor = torch.from_numpy(patch_data).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(patch_tensor)
        
        lr_thermal = patch_tensor[0, 0].cpu().numpy()
        hr_target = patch_tensor[0, 1].cpu().numpy()
        sr_output = output['sr_output'][0, 0].cpu().numpy()
        
        del patch_tensor, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        rmse = np.sqrt(np.mean((sr_output - hr_target)**2))
        psnr_value = 20 * np.log10(1.0 / rmse) if rmse > 0 else float('inf')
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(lr_thermal, cmap='hot')
        axes[0].set_title('Input: Low-Resolution Thermal', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(hr_target, cmap='hot')
        axes[1].set_title('Target: High-Resolution Thermal', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(sr_output, cmap='hot')
        axes[2].set_title(f'Output: Super-Resolved\nRMSE: {rmse:.4f} | PSNR: {psnr_value:.2f} dB', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        metrics_text = f"""
**Model Performance:**
- **RMSE:** {rmse:.4f} Kelvin
- **PSNR:** {psnr_value:.2f} dB
        """
        
        return Image.open(buf), metrics_text
        
    except Exception as e:
        import traceback
        return None, f"**Error:** {str(e)}\n``````"

demo = gr.Interface(
    fn=process_npz,
    inputs=gr.File(label="Upload .npz file", file_types=[".npz"]),
    outputs=[
        gr.Image(label="Visualization", type="pil"),
        gr.Markdown(label="Metrics")
    ],
    title="🔥 Thermal IR Super-Resolution",
    description="Upload a preprocessed .npz file to perform optical-guided super-resolution",
    cache_examples=False,
    delete_cache=(86400, 86400)
)

if __name__ == "__main__":
    demo.launch(show_error=True, max_file_size="10mb")