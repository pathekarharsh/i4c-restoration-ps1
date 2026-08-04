import torch
import torch.nn as nn

class LayerNorm2d(nn.Module):
    """
    Layer Normalization for 4D image tensors (B, C, H, W) normalized along the channel axis.
    """
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-6) * self.weight + self.bias

class SimpleGate(nn.Module):
    """
    SimpleGate splits channels in half and multiplies the two halves together.
    This acts as a nonlinear activation function without using explicit activations like ReLU.
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimplifiedChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA) block for weighting channel importance.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
        
    def forward(self, x):
        # Global average pooling
        pool = x.mean(dim=[-2, -1], keepdim=True)
        scale = self.conv(pool)
        return x * scale

class NAFBlock(nn.Module):
    """
    Nonlinear Activation-Free Block (NAFBlock) - the core computation block.
    """
    def __init__(self, channels):
        super().__init__()
        # Normalization and feature extraction
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=True)
        self.gate = SimpleGate()
        self.sca = SimplifiedChannelAttention(channels)
        
        # Linear block
        self.norm2 = LayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        
        # Skip connection scaling
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # First residual block pathway
        res1 = self.norm1(x)
        res1 = self.conv1(res1)
        res1 = self.gate(res1)
        res1 = self.sca(res1)
        # Apply scaling weight (initialized to 0) to ease early convergence
        x = x + res1 * self.beta
        
        # Second linear mapping pathway
        res2 = self.norm2(x)
        res2 = self.conv2(res2)
        x = x + res2 * self.gamma
        
        return x

class NAFNetLite(nn.Module):
    """
    Lightweight NAFNet-lite style super-resolution and denoising model.
    Accepts (B, 1, H, W) noisy inputs and outputs (B, 1, 2*H, 2*W) clean targets.
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=128, num_blocks=12):
        super().__init__()
        
        # 1. Shallow feature extraction
        self.intro = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True)
        
        # 2. Main deep feature extraction (NAF Blocks)
        self.body = nn.Sequential(*[
            NAFBlock(num_features) for _ in range(num_blocks)
        ])
        
        # 3. Upsampling layer: Learned 2x PixelShuffle upsampler
        up_features = num_features * 4  # 4x features for PixelShuffle(2) channel expansion
        self.up_conv = nn.Conv2d(num_features, up_features, kernel_size=3, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)  # 2x spatial upscaling
        
        # 4. Final reconstruction layer (maps from num_features to out_channels)
        self.out_conv = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()  # Restricts outputs strictly to [0.0, 1.0]

    def forward(self, x):
        # Extract features
        feat = self.intro(x)
        
        # Pass features through deep NAF blocks
        body_feat = self.body(feat)
        
        # Residual connection around the body
        feat = feat + body_feat
        
        # Upsample 2x
        feat = self.up_conv(feat)
        feat = self.pixel_shuffle(feat)
        
        # Map to final output channels
        out = self.out_conv(feat)
        out = self.sigmoid(out)
        
        return out

if __name__ == "__main__":
    # Test shape and model parameter count
    model = NAFNetLite(in_channels=1, out_channels=1, num_features=128, num_blocks=12)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized successfully!")
    print(f"Total Trainable Parameters: {total_params / 1e6:.3f}M")
    
    # Run a test forward pass
    test_input = torch.randn(4, 1, 128, 128)
    test_output = model(test_input)
    print(f"Input Shape:  {test_input.shape}")
    print(f"Output Shape: {test_output.shape}")
    assert test_output.shape == (4, 1, 256, 256), "Unexpected output shape!"
