import torch
import torch.nn as nn
import lpips
import ssl
from torchmetrics.functional.image import structural_similarity_index_measure

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 approximation) defined as:
    L(x, y) = sqrt((x - y)^2 + eps^2)
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2
        
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))

class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using pre-trained AlexNet weights from the `lpips` library.
    It expects inputs in the [0, 1] range, automatically maps them to [-1, 1],
    and repeats channels for 3-channel VGG/AlexNet compatiblity.
    """
    def __init__(self, net='alex'):
        super().__init__()
        # Temporary context shift to resolve macOS python SSL issues if any
        try:
            original_ctx = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception:
            original_ctx = None

        self.lpips_fn = lpips.LPIPS(net=net)
        
        # Restore original context if changed
        if original_ctx is not None:
            ssl._create_default_https_context = original_ctx

        # Freeze LPIPS parameters so we don't backprop into the pretrained trunk
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
            
    def forward(self, pred, target):
        # 1. Scale from [0, 1] to [-1, 1]
        pred_scaled = 2 * pred - 1
        target_scaled = 2 * target - 1
        
        # 2. Repeat grayscale channel 3 times for VGG/AlexNet input compatibility
        if pred_scaled.shape[1] == 1:
            pred_scaled = pred_scaled.repeat(1, 3, 1, 1)
            target_scaled = target_scaled.repeat(1, 3, 1, 1)
            
        # Ensure loss is computed on the correct device
        self.lpips_fn = self.lpips_fn.to(pred.device)
        
        loss = self.lpips_fn(pred_scaled, target_scaled)
        return torch.mean(loss)

class HybridLoss(nn.Module):
    """
    Hybrid Restoration Loss combining:
    L = 1.0 * Charbonnier + 0.3 * (1 - SSIM) + 0.05 * LPIPS
    """
    def __init__(self, w_charb=1.0, w_ssim=0.3, w_lpips=0.05):
        super().__init__()
        self.charb_fn = CharbonnierLoss()
        self.lpips_fn = PerceptualLoss(net='alex')
        self.w_charb = w_charb
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        
    def forward(self, pred, target):
        # 1. Charbonnier loss
        loss_charb = self.charb_fn(pred, target)
        
        # 2. SSIM loss (torchmetrics requires images to be in [0, 1] range)
        # We calculate structural similarity index measure and compute (1 - SSIM)
        ssim_val = structural_similarity_index_measure(pred, target, data_range=1.0)
        loss_ssim = 1.0 - ssim_val
        
        # 3. LPIPS loss
        loss_lpips = self.lpips_fn(pred, target)
        
        # Combined loss
        total_loss = (self.w_charb * loss_charb + 
                      self.w_ssim * loss_ssim + 
                      self.w_lpips * loss_lpips)
                      
        return total_loss, {
            "loss_charb": loss_charb.item(),
            "loss_ssim": loss_ssim.item(),
            "loss_lpips": loss_lpips.item(),
            "total_loss": total_loss.item()
        }

if __name__ == "__main__":
    # Test forward pass of loss functions
    loss_fn = HybridLoss()
    pred = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    total_loss, logs = loss_fn(pred, target)
    print("Loss calculation test: SUCCESS")
    print("Logs:", logs)
