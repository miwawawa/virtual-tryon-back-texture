import os
from train.utils import load_bbox
from infer_uv_texture import infer_uv_texture




if __name__ == "__main__":
    
    # input
    front_img_path = "/virtual-tryon-back-texture/train/dataset/THuman3.0/00001_1/00001_0003/00001_0003.png"
    obj_path       = "/virtual-tryon-back-texture/train/dataset/THuman3.0/00001_1/00001_0003/mesh.obj"
    bbox_path      = "/virtual-tryon-back-texture/train/dataset/THuman3.0/00001_1/00001_0003/bbox.txt"
    model_path     = "/virtual-tryon-back-texture/checpoints/epoch_all_210_1.pth"
    save_path      = "tex_pred.png"


    if os.path.exists(bbox_path):
        x1, y1, x2, y2 = load_bbox(bbox_path)
        print("Loaded bbox:", x1, y1, x2, y2)
    else:
        print("WARNING: bbox.txt not found, but proceeding without it.")

    infer_uv_texture(
        front_img_path=front_img_path,
        obj_path=obj_path,
        bbox_path=bbox_path,
        model_path=model_path,
        save_path=save_path
    )
