import os

def load_samples_from_folder(root_dir):
    
    # root_dir/
    #   └── 00001_1/
    #         ├── 00001_0003/
    #         │     ├── 00001_0003.png  ← image
    #         │     ├── mesh.obj        ← mesh
    #         │     ├── tex.png         ← uv
    #         │     └── tex.mtl         ← ignore
    #         ├── 00001_0004/
    #         ...
    

    samples = []

    for group in sorted(os.listdir(root_dir)):
        group_dir = os.path.join(root_dir, group)
    
        if not os.path.isdir(group_dir):
            continue

        for sample_folder in sorted(os.listdir(group_dir)):
            sample_dir = os.path.join(group_dir, sample_folder)
            if not os.path.isdir(sample_dir):
                continue

            image_path = None
            uv_path = None
            obj_path = None
            bbox_path=None

            for f in os.listdir(sample_dir):
                fpath = os.path.join(sample_dir, f)
                f_low = f.lower()

                # OBJ
                if f_low == "mesh.obj":
                    obj_path = fpath
                    continue

                # UV texture
                if f_low == "tex_down_1024.png":
                    uv_path = fpath
                    continue
                if f_low == "cloth_mask.png":
                    cloth_path = fpath
                    continue

                # MTL ignore
                if f_low.endswith(".mtl"):
                    continue

                # (00001_0003.png）
                if f_low.endswith(".png"):
                    if f_low != "tex_down_1024.png":
                        image_path = fpath
                        continue
                if f_low=="bbox.txt":
                    bbox_path=fpath
            # Check that all required files exist
            if image_path and obj_path and uv_path and bbox_path and cloth_path:
                samples.append({
                    "image": image_path,
                    "obj": obj_path,
                    "uv": uv_path,
                    "bbox": bbox_path,
                    "cloth_mask": cloth_path
                })
            else:
                print(f"   Skipped incomplete sample: {sample_dir}")
                print(f"   image={image_path}, obj={obj_path}, uv={uv_path}, bbox={bbox_path}")

    print(f"Loaded {len(samples)} samples from {root_dir}")
    return samples