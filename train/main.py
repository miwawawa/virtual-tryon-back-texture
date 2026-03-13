import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pytorch3d.io import load_objs_as_meshes
import config

from IN import load_samples_from_folder
from cnn_module import ImageEncoder,UVTexturePredictor
from train import train_model

class SingleViewTextureDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
               
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["image"]  
        img = Image.open(sample["image"]).convert("RGB")
        img_tensor = self.transform(img) 
        
        mesh = load_objs_as_meshes([sample["obj"]], device="cpu")[0]

        tex_img = Image.open(sample["uv"]).convert("RGB")
        tex_img = tex_img.resize((config.TEX_SIZE, config.TEX_SIZE))
        tex_gt = transforms.ToTensor()(tex_img).unsqueeze(0)

        with open(sample["bbox"], "r") as f:
            line = f.read().strip()
        parts = line.replace(",", " ").split()
        nums = [int(round(float(p))) for p in parts[:4]]
        x1, y1, x2, y2 = nums
        bbox_tensor = torch.tensor([x1, y1, x2, y2], dtype=torch.int32)

        cloth_img = Image.open(sample["cloth_mask"]).convert("L")
        cloth_img = cloth_img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.NEAREST)
        cloth_mask = (np.array(cloth_img) > 128).astype(np.float32)
        cloth_mask = torch.from_numpy(cloth_mask).unsqueeze(0) 

        return img_tensor, mesh, tex_gt, bbox_tensor, cloth_mask

def mesh_collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    meshes = [b[1] for b in batch]   
    tex_gts = torch.cat([b[2] for b in batch], dim=0)
    bboxes = torch.stack([b[3] for b in batch], dim=0)
    img_paths = [b[4] for b in batch]  
    return imgs, meshes, tex_gts, bboxes,img_paths


if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    root_dir = "./THuman3.0"   
    samples = load_samples_from_folder(root_dir)
    dataset = SingleViewTextureDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=8,  
        persistent_workers=True,
        collate_fn=mesh_collate_fn   
    )
    
    encoder = ImageEncoder().to(config.DEVICE)
    
    tex_predictor = UVTexturePredictor(feat_channels=512).to(config.DEVICE)

    for name, p in tex_predictor.named_parameters():
        if 'decoder' in name and 'weight' in name:
            print("name_and_p: ",name, p.mean(), p.std())



    params = list(encoder.parameters()) + list(tex_predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=config.LR)
    scheduler = None 

    history=train_model(dataloader, encoder, tex_predictor, optimizer)

    # If you want to resume training, uncomment the code below
    
    # history=train_model(
    # dataloader,
    # encoder,
    # tex_predictor,
    # optimizer,
    # resume_path="checkpoints/epoch_all_375_1.pth"
    # )

    with open("loss_log.txt", "w") as f:
        for value in history:
            f.write(f"{value}\n")