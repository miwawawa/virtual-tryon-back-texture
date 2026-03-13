import torch


# decide the proportion of the front image versus the predicted image to be used in the front area of the 3D model
def blend_pred_and_front(tex_pred, tex_front, valid_mask_3, front_mask, k=6.0):
    
    # difference between the predicted UV texture and the front texture
    diff = (tex_pred - tex_front).abs().mean(dim=1, keepdim=True) 
    
    # Convert the difference into a blending weight
    alpha = torch.sigmoid(k * diff)
    alpha = alpha * front_mask
    alpha = alpha * valid_mask_3
    
    
    # Blend predicted texture and front texture in front-facing regions.
    # Back-facing regions use only the predicted texture.
    tex_final = tex_pred * (1 - front_mask) + front_mask * ((1 - alpha)* tex_pred +  alpha* tex_front)
    
    return tex_final.clamp(0,1)
