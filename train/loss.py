import torch


def texture_l1_loss(tex_pred, tex_gt, valid_mask, face_uv_mask, cloth_uv_mask, front_mask,pts_world, w_face=10.0, alpha_color=0.3):

    # Compute the masked pixel-wise difference
    diff = (tex_pred - tex_gt) * valid_mask

    # Normalization term for valid pixels
    denom = valid_mask.sum() * tex_pred.shape[1] + 1e-8

    # Create a mask for the backside region
    back_mask=valid_mask*(1.0-front_mask)
    denom_back=back_mask.sum()*tex_pred.shape[1] + 1e-8

    # Base L1 loss over all valid regions
    l1_base = diff.abs().sum() / denom

     # L1 loss for the backside region
    l1_back = (
        (tex_pred - tex_gt).abs()
        * back_mask
    ).sum() / denom_back

    # Assign higher weights to colored regions in the ground-truth texture
    with torch.no_grad():
        gray = tex_gt.mean(dim=1, keepdim=True)
        weight_color = torch.where(
            gray > 0.05,
            torch.ones_like(gray),
            0.5 * torch.ones_like(gray)
        )

    # Weighted L1 loss that emphasizes colored regions
    l1_color = ((tex_pred - tex_gt).abs()
                * valid_mask
                * weight_color).sum() / denom
    
    # Increase the loss weight for face regions
    face_weight = 1.0 + face_uv_mask * (w_face - 1.0)
    l1_face = ((tex_pred - tex_gt).abs()
               * valid_mask
               * face_weight).sum() / denom
    
    # Increase the loss weight for clothing regions
    cloth_weight = 1.0 + cloth_uv_mask * (w_face - 1.0)
    l1_cloth = ((tex_pred - tex_gt).abs()
               * valid_mask
               * cloth_weight).sum() / denom
    # Final weighted loss
    loss = (
          0.1 * l1_base
        + alpha_color * l1_color
        + 2 * l1_back
        + 0.01 * l1_face
        + 0.1 * l1_cloth        
    )
    return loss
