import random
import numpy as np
import cv2

def add_speckle_noise(img, sigma=None):
    """
    Adds multiplicative speckle noise to the image:
    I_noisy = I + I * n, where n ~ N(0, sigma^2)
    """
    if sigma is None:
        # Randomize noise severity level
        sigma = random.uniform(0.01, 0.12)
    
    # Generate zero-mean Gaussian noise matching image dimensions
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    
    # Multiplicative application
    noisy_img = img + img * noise
    return noisy_img

def add_gaussian_blur(img, kernel_size=None, sigma=None):
    """
    Applies random Gaussian blur to the image.
    """
    if kernel_size is None:
        # Choose odd kernel sizes
        kernel_size = random.choice([3, 5, 7])
    if sigma is None:
        # Randomize standard deviation
        sigma = random.uniform(0.5, 1.8)
        
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    return blurred

def downsample_image(img, method=None):
    """
    Downsamples the image by 2x using a randomized interpolation method.
    This helps the model build robustness against downsampling mismatches.
    """
    if method is None:
        # Randomize interpolation method
        method = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
    
    h, w = img.shape[:2]
    # Reduce size by 2x
    downscaled = cv2.resize(img, (w // 2, h // 2), interpolation=method)
    return downscaled

def degrade_image(gt_img):
    """
    Applies a random combination of 1-3 degradations to a clean HR (256x256) image,
    returning a degraded LR (128x128) version.
    """
    img = gt_img.copy()
    
    # 1. Apply Gaussian Blur (80% probability)
    apply_blur = random.random() < 0.8
    if apply_blur:
        img = add_gaussian_blur(img)
        
    # 2. Downsample (always downsample by 2x to go from 256x256 -> 128x128)
    img = downsample_image(img)
    
    # 3. Apply Speckle Noise (80% probability)
    apply_noise = random.random() < 0.8
    if apply_noise:
        img = add_speckle_noise(img)
        
    # Ensure values don't overflow/underflow to extreme unmanageable values
    # (Though we allow overshoot in normalize, capping to [-0.5, 2.0] keeps it stable)
    img = np.clip(img, -0.5, 2.0)
    
    return img
