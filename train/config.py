import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
TEX_SIZE = 1024
BATCH_SIZE = 1
LR = 1e-5
NUM_EPOCHS = 376