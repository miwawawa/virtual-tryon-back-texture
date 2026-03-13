import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
import matplotlib.pyplot as plt

import torch.nn.functional as F
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
    OrthographicCameras,
)

def save_models(encoder, tex_predictor, save_dir="./checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model_latest.pth")

    torch.save({
        "encoder": encoder.state_dict(),
        "tex_predictor": tex_predictor.state_dict(),
    }, save_path)

    print(f"Saved model → {save_path}")
