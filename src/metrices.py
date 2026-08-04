import torch
import torch.nn as nn
from torchmetrics.functional.image import structural_similarity_index_measure
from src.losses import PerceptualLoss

class MetricsCalculator(nn.Module):
    """
    Computes PSNR, SSIM, and LPIPS metrics for image batches on the [0.0, 1.0] scale.
    """
    def __init__(self):
        super().__init__()
        self.lpips_fn = PerceptualLoss(net='alex')
        
    @torch.no_grad()
    def forward(self, pred, target):
        """
        pred, target: PyTorch tensors of shape (B, 1, H, W) normalized in [0.0, 1.0].
        """
        # Ensure values are strictly clamped to [0.0, 1.0] for standard metrics computation
        pred_clamped = torch.clamp(pred, 0.0, 1.0)
        target_clamped = torch.clamp(target, 0.0, 1.0)
        
        # 1. PSNR Calculation
        mse = torch.mean((pred_clamped - target_clamped) ** 2, dim=[-3, -2, -1])
        # Add epsilon to prevent log10(0)
        psnr = -10.0 * torch.log10(mse + 1e-8)
        mean_psnr = torch.mean(psnr).item()
        
        # 2. SSIM Calculation (using torchmetrics)
        # ssim requires data_range=1.0 as the tensors are clamped to [0.0, 1.0]
        ssim_val = structural_similarity_index_measure(pred_clamped, target_clamped, data_range=1.0).item()
        
        # 3. LPIPS Calculation (using our custom wrapper PerceptualLoss)
        lpips_val = self.lpips_fn(pred_clamped, target_clamped).item()
        
        return {
            "psnr": mean_psnr,
            "ssim": ssim_val,
            "lpips": lpips_val
        }

if __name__ == "__main__":
    # Quick verification
    calculator = MetricsCalculator()
    pred = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    metrics = calculator(pred, target)
    print("Metrics calculator test: SUCCESS")
    print("Metrics:", metrics)
