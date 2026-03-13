import torch
import torch.nn.functional as F
import config
import time
from torchvision import transforms
from relation import compute_uv_to_3d_points
from loss import texture_l1_loss
from utils import make_face_mask_from_bbox
from utils import build_front_camera
from utils import adjust_mesh_like_training
from save_model import save_models
from pytorch3d.renderer import (
    OrthographicCameras,
)

# Save training state
def save_checkpoint(path, epoch, encoder, tex_predictor, optimizer, loss_log):
    checkpoint = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "tex_predictor": tex_predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_log": loss_log,
    }
    torch.save(checkpoint, path)
    print(f"[Checkpoint] saved -> {path}")


# Restore model
def load_checkpoint(path, encoder, tex_predictor, optimizer, device):
    checkpoint = torch.load(path, map_location=device)

    encoder.load_state_dict(checkpoint["encoder"])
    tex_predictor.load_state_dict(checkpoint["tex_predictor"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    encoder.to(device)
    tex_predictor.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    start_epoch = checkpoint["epoch"] + 1
    loss_log = checkpoint["loss_log"]
    print(f"[Checkpoint] loaded -> {path} (restart from epoch {start_epoch})")
    return start_epoch, loss_log



def train_model(dataloader, encoder, tex_predictor, optimizer, resume_path=None):
   
    loss_log = []
    start_epoch = 0

    # Number of mini-batches used for gradient accumulation
    ACCUM = 8

    # Resume training
    if resume_path is not None:
        start_epoch, loss_log = load_checkpoint(resume_path, encoder, tex_predictor, optimizer, config.DEVICE)
        print("Resuming training...")

    # Main training loop
    for epoch in range(start_epoch,config.NUM_EPOCHS):
        epoch_start_time = time.time()
        accum_counter=0
        optimizer.zero_grad(set_to_none=True)
        judge=True
        for batch_idx, (img_tensor, meshes, tex_gt, bboxes, cloth_mask) in enumerate(dataloader):
            
            B = img_tensor.shape[0]

            img_tensor = img_tensor.to(config.DEVICE)
            tex_gt = tex_gt.to(config.DEVICE)

            # Extract front-view image
            feat = encoder(img_tensor)   # (B, C, Hf, Wf)

            total_loss = 0.0
            
            # Process each sample individually
            for b in range(B):
                mesh = meshes[b].to(config.DEVICE)

                # Define the front-view camera
                cameras_front = build_front_camera()
                
                # Apply manual mesh alignment for THuman3.0
                mesh = adjust_mesh_like_training(mesh)

                # Compute the 3D point corresponding to each UV texel
                pts_world, valid_mask,normals = compute_uv_to_3d_points(mesh, cameras_front,config.TEX_SIZE)

                # Compute the view direction in world coordinates
                cam_dir_camcoords = torch.tensor([0., 0., 1.], device=config.DEVICE).view(1,3,1)
                view_dir = cameras_front.R.transpose(1,2) @ cam_dir_camcoords
                view_dir = view_dir.view(1,1,1,3)

                # Create a front-facing mask from the normal-view direction dot product
                dot = (normals * view_dir).sum(dim=-1, keepdim=True)  
                front_mask = (dot < 0).float()
                front_mask = front_mask.permute(0,3,1,2)
                
                # Select the current sample
                feat_b = feat[b:b+1]          # (1,C,Hf,Wf)
                tex_gt_b = tex_gt[b:b+1]      # (1,3,Ht,Wt)

                # Create a face mask
                bbox = bboxes[b].cpu().tolist()   # (x1,y1,x2,y2)
                face_mask = make_face_mask_from_bbox(bbox, config.IMG_SIZE, config.IMG_SIZE, config.DEVICE)
                face_mask_img = face_mask.unsqueeze(0).unsqueeze(0).float()

                # Predict the UV texture
                tex_pred, valid_mask_3, grid= tex_predictor(
                    feat_b,
                    pts_world,
                    cameras_front,
                    img_size=(config.IMG_SIZE, config.IMG_SIZE),
                    valid_mask=valid_mask
                )
                
                if isinstance(cloth_mask, list):
                    cloth_mask = cloth_mask[0]
                cloth_mask = cloth_mask.to(config.DEVICE) 
                cloth_mask_img = cloth_mask.unsqueeze(0).float()

                # Project the face region into UV space
                face_uv_mask = F.grid_sample(
                    face_mask_img,   
                    grid,            
                    mode='nearest',
                    padding_mode='zeros',
                    align_corners=False
                )

                # Project the clothing region into UV space
                cloth_uv_mask = F.grid_sample(
                    cloth_mask_img,   
                    grid,            
                    mode='nearest',
                    padding_mode='zeros',
                    align_corners=False
                )

                # valid UV texels
                face_uv_mask = face_uv_mask * valid_mask_3
                cloth_uv_mask=cloth_uv_mask*valid_mask_3

                # Save an intermediate prediction image
                if((epoch+1)%50==0 and  batch_idx == 0 and b == 0):
                    with torch.no_grad():
                        tex_img = tex_pred[0].detach().cpu().clamp(0,1)
                        to_pil = transforms.ToPILImage()
                        pil_img = to_pil(tex_img)
                        pil_img.save(f"pred_epoch{epoch+1}_5.png")
                        print(f"Saved: pred_epoch{epoch+1}_5.png")
                
                valid_uv_mask = valid_mask_3

                # Compute the loss
                loss_l1 = texture_l1_loss(
                    tex_pred, tex_gt_b,
                    valid_uv_mask,
                    face_uv_mask,
                    cloth_uv_mask,
                    front_mask,
                    pts_world,
                    w_face=15.0,
                )

                # Accumulate sample loss within the batch
                total_loss = total_loss + loss_l1

            total_loss = total_loss / B           
            original_loss=total_loss

            # Normalize loss for gradient accumulation
            total_loss = total_loss / ACCUM       

            total_loss.backward()                 
            accum_counter+=1

            # Update parameters after ACCUM steps
            if (batch_idx + 1) % ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_counter=0
            
        # Handle remaining gradients
        if accum_counter > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        
        print(f"[{epoch+1}] loss={original_loss.item():.4f}")
        
        if(judge==True):
            loss_log.append(original_loss.item())
            judge=False

        # Save a checkpoint
        if (epoch + 1) % 1 == 0:
            save_checkpoint(
                path=f"checkpoints/epoch_all_{epoch+1}_1.pth",
                epoch=epoch,
                encoder=encoder,
                tex_predictor=tex_predictor,
                optimizer=optimizer,
                loss_log=loss_log
            )
        epoch_time = time.time() - epoch_start_time
        minutes = epoch_time / 60
        print(f"[{epoch+1}] epoch time: {epoch_time:.2f} sec ({minutes:.2f} min)")
    save_models(encoder, tex_predictor)
    return loss_log