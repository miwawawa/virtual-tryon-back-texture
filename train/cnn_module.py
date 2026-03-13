import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models




class ImageEncoder(nn.Module):
    """
    Image encoder based on a modified ResNet-18 backbone.

    Input:
        x: (B, 3, H, W)

    Output:
        feat: (B, C, Hf, Wf)

    The encoder uses the convolutional part of ResNet-18 and removes
    the final average pooling and fully connected layers.
    Strides in layer2-4 are changed to preserve higher spatial resolution.
    """

    # ResNet-18
    def __init__(self):
        super().__init__()

        # Load ImageNet-pretrained ResNet-18
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Preserve spatial resolution by removing downsampling
        backbone.layer2[0].conv1.stride = (1, 1)
        backbone.layer2[0].downsample[0].stride = (1, 1)

        backbone.layer3[0].conv1.stride = (1, 1)
        backbone.layer3[0].downsample[0].stride = (1, 1)

        backbone.layer4[0].conv1.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)

        # Use only convolutional layers up to layer4
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # conv5_x まで
        
    def forward(self, x):
        return self.encoder(x)
    

class UVTexturePredictor(nn.Module):
    """
    Predict a UV texture map from front-view image features.

    Inputs:
        feat:          front-view image features, (B, C, Hf, Wf)
        pts_world:     3D coordinates for each UV texel, (B, Ht, Wt, 3)
        cameras_front: camera object used for projection
        img_size:      input image size
        valid_mask:    valid UV mask, (B, Ht, Wt)

    Outputs:
        tex_pred:      predicted UV texture, (B, 3, Ht, Wt)
        valid_mask_3:  expanded valid mask, (B, 1, Ht, Wt)
        grid:          sampling grid used for feature lookup, (B, Ht, Wt, 2)
    """
    def __init__(self, feat_channels):
        super().__init__()

        # Reduce the feature dimension to make the decoder lighter
        self.reduce = nn.Conv2d(feat_channels, 64, kernel_size=1)
        
        # Low-level branch for local detail refinement
        self.low = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Main decoder branch
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Fuse main features with low-level sampled features
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Predict RGB texture values
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat, pts_world, cameras_front, img_size, valid_mask):
        B, C, Hf, Wf = feat.shape
        _, Ht, Wt, _ = pts_world.shape
        
        # Reduce channel dimension of the front-view feature map
        feat = self.reduce(feat)     # (B,64,Hf,Wf)

        # Flatten UV texel coordinates for camera projection
        pts_world_flat = pts_world.reshape(B, -1, 3)   # (B, Ht*Wt, 3)


        # Project 3D UV texel coordinates to screen space
        pts_screen_flat = cameras_front.transform_points_screen(
            pts_world_flat, image_size=img_size
        )  # (B,Ht*Wt,3) 
        
        # Restore UV texture layout
        pts_screen = pts_screen_flat.reshape(B, Ht, Wt, 3)
        
        xs = pts_screen[..., 0]
        ys = pts_screen[..., 1]

        H_img, W_img = img_size

        # Normalize screen coordinates to [-1, 1] for grid_sample
        x_norm = 2.0 * (xs / (W_img - 1.0)) - 1.0
        
        y_norm = 2.0 * (ys / (H_img - 1.0)) - 1.0
        
        # Build the sampling grid in UV space
        grid = torch.stack([x_norm, y_norm], dim=-1)  # (B,Ht,Wt,2)
        
        # Sample front-view features at projected UV texel locations
        feat_tex = F.grid_sample(
            feat,                # (B,64,Hf,Wf)
            grid,                # (B,Ht,Wt,2)
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )  # (B,64,Ht,Wt)
        
        # Extract low-level features and sample them in UV space
        low_feat = self.low(feat)  # (B,32,Hf,Wf)
        low_tex = F.grid_sample(
            low_feat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )  # (B,32,Ht,Wt)

        # Decode sampled features into UV texture
        f = self.dec1(feat_tex)              # (B,128,Ht,Wt)
        f = torch.cat([f, low_tex], dim=1)   # (B,128+32=160,Ht,Wt)
        f = self.dec2(f)                     # (B,64,Ht,Wt)

        # Predict RGB values for each UV texel
        tex_pred = self.final(f)             # (B,3,Ht,Wt)
        
        # Apply valid UV mask to remove invalid regions
        valid_mask_3 = valid_mask.unsqueeze(1) # (B,1,Ht,Wt)
        
        tex_pred = tex_pred * valid_mask_3

        return tex_pred, valid_mask_3, grid
    
