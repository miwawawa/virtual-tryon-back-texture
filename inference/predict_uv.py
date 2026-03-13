import os
from train.utils import load_bbox
from infer_uv_texture import infer_uv_texture




if __name__ == "__main__":
    
    # input
    '''front_img_path = "./dataset/00001_1/00001_0003/00001_0003.png"
    obj_path       = "./dataset/00001_1/00001_0003/mesh.obj"
    bbox_path      = "./dataset/00001_1/00001_0003/bbox.txt"
    model_path     = "./checkpoints/model_latest_200epochs_18dataset_face_cloth_loss_except39.pth"
    save_path      = "tex_pred_5.png"'''

    front_img_path = "./00005_0018/00005_0018.png"
    obj_path       = "./00005_0018/mesh.obj"
    bbox_path      = "./00005_0018/bbox.txt"
    model_path     = "./checkpoints/model_latest_200epochs_18dataset_face_cloth_loss_except39.pth"
    save_path      = "tex_pred_5.png"


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
