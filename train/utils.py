import torch
from pytorch3d.structures import Meshes
import config
from pytorch3d.renderer import (
    OrthographicCameras,
)


def make_face_mask_from_bbox(bbox, H, W, device):
    mask = torch.zeros((H, W), device=device)
    x1,y1,x2,y2 = bbox
    mask[y1:y2, x1:x2] = 1.0
    return mask

# Define the front-view camera
def build_front_camera():
    cameras_front = OrthographicCameras(
        device=config.DEVICE,
        R=torch.tensor(
            [[[-1., 0., 0.],
              [ 0., 1., 0.],
              [ 0., 0., -1.]]],
            device=config.DEVICE
        ),
        T=torch.tensor([[0., 0., 1.]], device=config.DEVICE),
        in_ndc=True,
    )
    return cameras_front

# Apply manual mesh alignment for THuman3.0
def adjust_mesh_like_training(mesh: Meshes):
    verts = mesh.verts_padded()
    verts = verts * 0.8
    verts[..., 1] -= (1.01) / 512.0
    verts[..., 0] -= 0.146  / 512.0
    mesh = mesh.update_padded(verts)
    return mesh


def load_bbox(bbox_path):
    with open(bbox_path, "r") as f:
        line = f.read().strip()
    parts = line.replace(",", " ").split()
    x1, y1, x2, y2 = map(int, parts)
    return x1, y1, x2, y2

