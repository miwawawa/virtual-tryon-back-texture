import torch

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE  = 512        
TEX_SIZE  = 1024       
MEAN      = [0.485, 0.456, 0.406]
STD       = [0.229, 0.224, 0.225]