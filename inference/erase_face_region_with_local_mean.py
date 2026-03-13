
import torch.nn.functional as F

# Apply the blur only to the back hair.
def erase_overlap_with_local_mean(tex, front_mask, face_uv_mask, radius=10):

    tex = tex.clone()

    # Create a locally averaged texture using a large mean filter.
    blurred = F.avg_pool2d(
        tex,
        kernel_size=radius*2+1,
        stride=1,
        padding=radius
    )

    # Apply 'blurred' to the back hair only
    tex = (1 - front_mask)*face_uv_mask*blurred + (1 - face_uv_mask)*tex + front_mask*face_uv_mask*tex


    return tex
