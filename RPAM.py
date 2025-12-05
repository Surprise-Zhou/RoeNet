import torch
import torch.nn as nn
import time
import numpy as np
from torch.nn import functional as F

class RotorAttention(torch.nn.Module):
    """
    Rotor Physics-based Attention Mechanism for UAV Detection
    
    """
    def __init__(self, e_lambda=1e-4, symmetry_weight=0.5):
        super(RotorAttention, self).__init__()
        self.e_lambda = e_lambda
        self.symmetry_weight = symmetry_weight
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w = x.size()
        device = x.device
        dtype = x.dtype
        
        # 1️⃣ Base mechanism attention
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        base_attention = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        # 2️⃣ Symmetry-constrained attention
        
        center_h, center_w = h // 2, w // 2
        
        # Create radial distance matrix - Ensure type and device match
        y_coords = torch.arange(h, dtype=dtype, device=device).view(-1, 1) - center_h
        x_coords = torch.arange(w, dtype=dtype, device=device).view(1, -1) - center_w
        radial_dist = torch.sqrt(y_coords**2 + x_coords**2)
        
        # Create an angle matrix
        angles = torch.atan2(y_coords, x_coords)  # [-π, π]
        
        # 3️⃣ Radial blur perception of attention 
        # Attention weights based on the radial blur characteristics of the rotor
        # Using a Gaussian radial kernel to simulate the rotor blur effect
        sigma_r = torch.clamp(radial_dist.max() / 4, min=1.0).to(dtype=dtype)
        
        # Radial attentional nucleus
        radial_kernel = torch.exp(-radial_dist**2 / (2 * sigma_r**2))
        
        # Angle-symmetric nucleus
        angle_symmetry = torch.cos(4 * angles) + 1  # 四重对称性
        angle_symmetry = torch.clamp(angle_symmetry, min=0)
        
        # 4️⃣ Multiscale Radial Attention
        # Simulate the blurring effect under different rotational speeds
        multi_scale_attention = torch.zeros_like(radial_kernel, dtype=dtype, device=device)
        scales = [0.5, 1.0, 2.0]  
        weights = [0.3, 0.4, 0.3]  
        
        for scale, weight in zip(scales, weights):
            scaled_sigma = sigma_r * scale  
            scale_kernel = torch.exp(-radial_dist**2 / (2 * scaled_sigma**2))
            multi_scale_attention += weight * scale_kernel
        
        # 5️⃣ Integrated spatial attention based on physical priors
        # Combining radial blurring and symmetry constraints
        physics_attention = radial_kernel * angle_symmetry * multi_scale_attention
        physics_attention = physics_attention.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        physics_attention = physics_attention.expand(b, c, h, w)
        
        
        physics_attention = physics_attention.to(dtype=dtype, device=device)
        
        # 6️⃣ Temporal domain change perception
        # Simulate the temporal evolution using the channel dimension, and simulate the periodicity of the rotor rotation.
        time_phase = torch.arange(c, dtype=dtype, device=device) * 2 * torch.pi / c
        time_modulation = (torch.sin(time_phase) + 1) / 2  # [0, 1]
        time_modulation = time_modulation.view(1, c, 1, 1)
        
        # 7️⃣ Final calculation of attention weights
        # Integrating basic attention, physical priors and temporal variations
        final_attention = (
            base_attention * (1 - self.symmetry_weight) + 
            physics_attention * self.symmetry_weight
        ) * time_modulation
        
        # 8️⃣ Adaptive Normalization 
        final_attention = final_attention / (final_attention.mean(dim=[2, 3], keepdim=True) + self.e_lambda)
        
        return x * self.activation(final_attention)
