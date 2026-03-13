
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.functional as F


def make_front_uv_texture(img_tensor, grid, valid_mask_3, front_mask,save_path=None):

    # Sample RGB values from the input image according to the UV→image mapping grid.
    # Each UV texel retrieves the corresponding pixel color from the front image.
    tex = F.grid_sample(
        img_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False
    ) 

    # Keep only texels that are:
    # 1) valid on the mesh surface (valid_mask_3)
    # 2) facing the camera (front_mask)
    mask = valid_mask_3 * front_mask
    tex = tex * mask
    tex=tex.clamp(0, 1)
    if save_path is not None:
        to_pil = transforms.ToPILImage()

        tex_img = to_pil(tex[0].cpu())
        tex_img.save(save_path)

        print(f"[OK] Saved front-only UV → {save_path}")
    return tex
