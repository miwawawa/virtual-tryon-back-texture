
import torch
import torch.nn.functional as F
import config
from torchvision import  transforms
from PIL import Image
from train.cnn_module import ImageEncoder
from train.cnn_module import UVTexturePredictor
from train.relation import compute_uv_to_3d_points
from train.utils import make_face_mask_from_bbox
from train.utils import build_front_camera
from train.utils import adjust_mesh_like_training
from train.utils import load_bbox
from make_front_uv_texture import make_front_uv_texture
from blend_pred_and_front import blend_pred_and_front
from erase_face_region_with_local_mean import erase_overlap_with_local_mean
from pytorch3d.io import load_objs_as_meshes



transform_img = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

transform_img_no_norm = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),  
])


def infer_uv_texture(front_img_path, obj_path, bbox_path, model_path,
                     save_path="tex_pred_1.png"):
    
    # Build model architecture
    encoder       = ImageEncoder().to(config.DEVICE)
    tex_predictor = UVTexturePredictor(feat_channels=512).to(config.DEVICE)

    # Load trained weights
    ckpt = torch.load(model_path, map_location=config.DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    tex_predictor.load_state_dict(ckpt["tex_predictor"])

    encoder.eval()
    tex_predictor.eval()

    # Load input image and apply preprocessing
    img = Image.open(front_img_path).convert("RGB")
    img_tensor = transform_img(img).unsqueeze(0).to(config.DEVICE)  
    img_tensor_raw = transform_img_no_norm(img).unsqueeze(0).to(config.DEVICE)

    # Load mesh and apply the same geometric normalization as training
    mesh = load_objs_as_meshes([obj_path], device=config.DEVICE)[0]
    mesh = adjust_mesh_like_training(mesh)

    # Build the front camera
    cameras_front = build_front_camera()

    # UV → 3D
    pts_world, valid_mask, normals = compute_uv_to_3d_points(mesh, cameras_front, config.TEX_SIZE)

    # Compute camera viewing direction
    cam_dir_camcoords = torch.tensor([0., 0., 1.], device=config.DEVICE).view(1,3,1)

    # Transform view direction to world coordinates
    view_dir = cameras_front.R.transpose(1,2) @ cam_dir_camcoords  

    view_dir = view_dir.view(1,1,1,3)  


    # Compute dot product between surface normals and camera direction
    dot = (normals * view_dir).sum(dim=-1, keepdim=True)   

    # Negative value indicates the surface
    front_mask = (dot < 0).float()  
    front_mask = front_mask.permute(0,3,1,2)  
    # Extract image features
    with torch.no_grad():
        feat = encoder(img_tensor)  

        # Predict UV texture
        tex_pred, valid_mask_3, grid = tex_predictor(
            feat,
            pts_world,
            cameras_front,
            img_size=(config.IMG_SIZE, config.IMG_SIZE),
            valid_mask=valid_mask
        )
        # Generate UV texture from the input image
        tex_front = make_front_uv_texture(img_tensor_raw, grid, valid_mask_3, front_mask, save_path="front_only_uv_from_front.png")

        # Blend predicted texture with front-view texture adaptively
        tex_final = blend_pred_and_front(tex_pred, tex_front, valid_mask_3, front_mask,k=6.0)

    # Build face mask 
    face_mask = make_face_mask_from_bbox(load_bbox(bbox_path), config.IMG_SIZE, config.IMG_SIZE, config.DEVICE)
    face_mask_img = face_mask.unsqueeze(0).unsqueeze(0).float()

    # Project the face mask from image space to UV space
    face_uv_mask = F.grid_sample(
                    face_mask_img,  
                    grid,           
                    mode='nearest',
                    padding_mode='zeros',
                    align_corners=False
    )

    # valid UV pixels
    face_uv_mask = face_uv_mask * valid_mask_3

    # A local mean smoothing
    tex_final = erase_overlap_with_local_mean(tex_final, front_mask, face_uv_mask, radius=7)
    
    # save
    tex = tex_final[0].cpu().clamp(0, 1)
    tex_img = transforms.ToPILImage()(tex)
    tex_img.save(save_path)

    print(f"[OK] Saved UV texture → {save_path}")