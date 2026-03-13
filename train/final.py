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
#epochs#grid
#train_model#grid_debug.png#decoder#mesh_colate#mesh_collate_fn
# =========================OrthographicCameras
# 0. 設定
# =========================
#RasterizationSettings
#self.conv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512          # 正面画像の解像度
TEX_SIZE = 1024         # UVテクスチャの解像度（正解に合わせて変更）
BATCH_SIZE = 1          # まずは 1 でOK（拡張可能）
LR = 1e-5
NUM_EPOCHS = 376

# 損失の重み
LAMBDA_L1 = 1.0
LAMBDA_SMOOTH = 0.1
LAMBDA_SYM = 0.1
#getitem
#UVTexturePredictor#decoder#simplify
# =========================
# 1. 正面画像用 Dataset の雛形
# =========================
#rasterize_meshes#align_corners
#compute_uv_to_3d_points
class SingleViewTextureDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),#0~255が0~1のfloatに変換。形状は(C,H,W)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],#x'=(x-mean)/stdで変換して標準化
                std=[0.229, 0.224, 0.225]#標準化する理由は大きさがバラバラだと勾配計算でうまくいかない
                #標準化で平均0,分散1付近になる#三つ値があるのはRGBに対応
            )
        ])

    def __len__(self):
        return len(self.samples)#データ数を返す

    def __getitem__(self, idx):
        sample = self.samples[idx]
        #samplesはidx=0でimage,obj,uvをすべてセットで持つ

        # --- 正面画像 ---
        img_path = sample["image"]  
        img = Image.open(sample["image"]).convert("RGB")#色をRGBに変える#imageじゃない場合に備えifが正しい
        img_tensor = self.transform(img)  # (3,H,W)#__init__のtransformを用いる

        # --- メッシュ ---

        
        mesh = load_objs_as_meshes([sample["obj"]], device="cpu")[0]#0で最初のmeshをとりだす
        # load_objs_as_meshesはpytorch3Dの関数で3D座標やf,(u,v)座標をとりだす
        #print("verts:", mesh.verts_padded().shape)
        #print("faces:", mesh.faces_padded().shape)


        # --- 正解UV (教師データとして使う) ---
        tex_img = Image.open(sample["uv"]).convert("RGB")#uvのデータを取りだす
        tex_img = tex_img.resize((TEX_SIZE, TEX_SIZE))
        tex_gt = transforms.ToTensor()(tex_img).unsqueeze(0)  # (1,3,H,W)
        #print("tex_gt_mean: ",tex_gt.mean(dim=[0,2,3]))
        #ToTensorで0~1floatにする。unsqueeze(0)は先頭に次元を1つ追加->(1,3,H,W)

        with open(sample["bbox"], "r") as f:
            line = f.read().strip()
        parts = line.replace(",", " ").split()
        nums = [int(round(float(p))) for p in parts[:4]]
        x1, y1, x2, y2 = nums
        bbox_tensor = torch.tensor([x1, y1, x2, y2], dtype=torch.int32)

        # --- cloth_mask ---
        cloth_img = Image.open(sample["cloth_mask"]).convert("L")
        cloth_img = cloth_img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        cloth_mask = (np.array(cloth_img) > 128).astype(np.float32)
        cloth_mask = torch.from_numpy(cloth_mask).unsqueeze(0)  # shape (1, H, W)

        return img_tensor, mesh, tex_gt, bbox_tensor, cloth_mask 
#//////////////////////////追加する////////////////////////////////
#mesh_collate_fn
from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    RasterizationSettings,
    PointLights,
)
#//////////////////////////////////////////NDC変換確認用/////////////////////////////////////////////いらない//////////////////////////
def debug_uv_mask_from_pix_to_face(pix_to_face, filename="uv_mask.png"):
    """
    pix_to_face: (H, W)  (-1 or face index)
    """
    import numpy as np
    from PIL import Image

    mask = (pix_to_face.cpu().numpy() >= 0).astype(np.uint8)  # 0 or 1
    # 1→白, 0→黒
    mask_img = (mask * 255).astype(np.uint8)
    mask_rgb = np.stack([mask_img]*3, axis=-1)
    Image.fromarray(mask_rgb).save(filename)
    print("Saved:", filename)


#///////////////////////////////////////////////////////////////////いらない/////////////////////////////////////////////////////////////////
def render_front_view(mesh, cameras_front, image_size=512):

    # ラスタライズ設定（線画でもOK）
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # シェーダは何でもOK、色付けが見えれば十分
    lights = PointLights(device=device, location=[[0, 0, 3]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras_front,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras_front,
            lights=lights
        )
    )
    '''scale = 0.8  
    verts = mesh.verts_padded() * scale
    mesh = mesh.update_padded(verts)'''


    # メッシュの正面画像を生成
    img = renderer(mesh)
    return img  # shape = (1,H,W,3)
#///////////////////////////////////////////////////////////いらない///////////////////////////////////
def visualize_feature_map(feat, prefix="feat"):
    """
    feat: (B, C, H, W)
    各チャネルをヒートマップ表示・保存
    """
    import matplotlib.pyplot as plt
    import os
    os.makedirs(prefix, exist_ok=True)

    B, C, H, W = feat.shape
    fmap = feat[0].detach().cpu()

    for i in range(min(C, 16)):  # 16チャネルだけ保存
        fm = fmap[i]
        plt.imshow(fm, cmap="viridis")
        plt.axis("off")
        plt.savefig(f"{prefix}/channel_{i}.png", bbox_inches="tight")
        plt.close()

    #print(f"Saved first 16 channels to folder ./{prefix}")

#////////////////////////////////////////////////////experiments/////////////////////////////////////////////////////////////////////

def compute_uv_to_3d_points(mesh_cpu: Meshes, cameras, tex_size=TEX_SIZE):
    #Meshesはpytorch3Dの型の種類の一つ
    #tex_sizeは出力するUVマップの解像度
    """
    - 入力: メッシュ(mesh_cpu)
    - 出力:
        pts_world: (1, Ht, Wt, 3)  各UV texel に対応する 3D 座標
        valid_mask: (1, Ht, Wt)    その texel がメッシュ上かどうか
    """

    # ===== 1. verts / faces (3D 側) =====
    mesh = mesh_cpu.to(device)
    verts = mesh.verts_padded()   # (1, V, 3)->各3D頂点の(x,y,z)
    faces = mesh.faces_padded()   # (1, F, 3)->各fの三頂点(v1,v2,v3)


    # ===== 2. UV verts / faces =====
    # verts_uvs: (1, Vt, 2) in [0,1]
    # faces_uvs: (1, F, 3)
    verts_uvs = mesh.textures.verts_uvs_padded()[0].to(device)  # (Vt,2)
    faces_uvs = mesh.textures.faces_uvs_padded()[0].to(device)  # (F,3)
    #texturesは(u,v)座標が入っている.[0]でバッチの0番目の要素の(Vt,2)が代入される
    #facesはUV上の三角形の頂点なので(F,3)
    #print("UV min/max:", float(verts_uvs.min()), float(verts_uvs.max()))

    # ===== 3. うまくいったコードと同じように UV → NDC 変換 =====
    uv = verts_uvs.clone()#[0,1]
    #verts_uvs は [0,1] 範囲の (u,v)。
    # PyTorch3D のラスタライザは、画面座標を NDC (Normalized Device Coordinates) の [-1,1] 範囲で扱う。
    # そのため、UV を NDC に変換している

    # u（横）は左右反転してから [-1,1]
    #///////////////////変更してみた//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #uv[:, 0] = 1.0 - uv[:, 0]#こっちをするとlossが上がる
    uv[:, 0] = uv[:, 0] * 2.0 - 1.0

    # v（縦）は上下反転してから [-1,1]
    #///////////////////変更してみた//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #uv[:, 1] = 1.0 - uv[:, 1]
    uv[:, 1] = uv[:, 1] * 2.0 - 1.0
    # v (縦方向) は上下反転 → 画像系に揃える
#uv_map V min/max
    '''z = torch.ones_like(uv[:, :1])*1.0 
    uv_verts_3d = torch.cat([uv, z], dim=-1)'''
    uv_verts_3d = torch.cat(#z=0の座標を付け加えている
        [uv, torch.zeros_like(uv[:, :1])],#uv[:, :1]は(Vt,1)
        dim=-1#これがuv[:, :1]を最後の次元に結合するよういってる
    )  # (Vt,3)

    #uv は (Vt,2) → これに z=0 の次元を追加して (Vt,3) にする。
    # つまり、UV平面を「Z=0 の平面に置かれた 3Dメッシュ」として扱う。
    # この uv_verts_3d をラスタライズすると、「UV空間での三角形」が 2D 画像に焼かれる。

    #print("uv_verts_3d min/max:", float(uv_verts_3d.min()), float(uv_verts_3d.max()))

    # ===== 4. UV メッシュを Meshes 型にまとめる =====
    uv_mesh = Meshes(
        verts=[uv_verts_3d],   # リストにする
        faces=[faces_uvs],
    )

    # ===== 5. OrthographicCameras + MeshRasterizer で UV 平面をラスタライズ =====
    '''cameras = OrthographicCameras(#OrthographicCamerasは平行投影カメラ
        device=device,
        R=torch.tensor([[#R：回転行列（3x3）
            #y と z を反転している（軸の向きの違いを補正）
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., 0., -1.],
        ]], device=device),
        T=torch.tensor([[0., 0., 1.]], device=device),
        #平行移動（カメラ位置）
        # [0,0,1] → カメラは [0,0,1]から原点に向いているイメージ
    )'''
    '''rendered = render_front_view(mesh, cameras, image_size=IMG_SIZE)
    rendered_img = rendered[0, ..., :3].detach().cpu()

    from torchvision.utils import save_image
    save_image(rendered_img.permute(2,0,1), "front_render.png")
    print("Saved front_render.png")'''
    #pts_view = cameras.get_world_to_view_transform().transform_points(uv_verts_3d)
    #print("view z-range_compute_uv_to_3d_points", pts_view[...,2].min(), pts_view[...,2].max())
#ラスタライズは「三角形メッシュを、ピクセル単位の“画像”に落とし込む処理」
    raster_settings = RasterizationSettings(
        #これらは「このメッシュをどう画面に焼くか」の条件設定
        image_size=tex_size,
        faces_per_pixel=1,
        #1つのピクセルについて「何枚までの三角形を記録するか」。
        # 1 → 一番手前の三角形だけ
        # 5 → 前から順に5枚まで（重なりを考慮するときに使う）
        blur_radius=0.0,
        #三角形のエッジをぼかすかどうか。
        # 0.0 なら、「完全に内側＝ヒット、外＝ノーヒット」というシャープな判定。
    )
#grid
    rasterizer = MeshRasterizer(
        #MeshRasterizer はカメラ（投影の仕方）
        # ラスタライズ設定（解像度、faces_per_pixelなど）
        # を持っていて、uv_mesh を画面に焼き込む「レンダリングの前半部分」を担当します。
        cameras=cameras,
        raster_settings=raster_settings,
    )
    #uv_mesh の各三角形を、カメラ座標 → 画面座標 (x,y) に投影
    # 画面の各ピクセルについて「このピクセルの中心点は、どの三角形の内部に入るか？」
    # 「入るなら、その三角形のどの位置か？」（バリセントリック座標）
    # その結果を fragments にまとめて返す
    fragments = rasterizer(uv_mesh)
    #fragments は PyTorch3D が返してくる「ラスタライズ結果まとめオブジェクト」です。
    # 中には代表的にこんな情報が入っています：
    # fragments.pix_to_face :
    # 各ピクセルが「どの三角形(face index)」に対応しているか
    # fragments.bary_coords :
    # そのピクセルが三角形内部のどの位置にいるか（バリセントリック座標）
    # fragments.zbuf :そのピクセルの深度 (z) 情報
    pix_to_face = fragments.pix_to_face[0, ..., 0]#(H,W)元々は(N, H, W, K)
    #pix_to_face[h, w] = f→ 画素 (h,w) は、「faces[f] 番の三角形」に属している。
    # pix_to_face[h, w] = -1→ 何の三角形にもヒットしていない（背景）
    bary_coords = fragments.bary_coords[0, ..., 0, :]#(H,W,3)元々は(N, H, W, K, 3)
    #末尾の 3 が「三角形の3頂点に対応するバリセントリック座標 (w0, w1, w2)」
    #これはp=w0​v0​+w1​v1​+w2​v2​,w0​+w1​+w2​=1
    H_tex, W_tex = pix_to_face.shape
    #print("UV rasterized size:", H_tex, W_tex)
    #debug_uv_mask_from_pix_to_face(pix_to_face, "uv_mask_from_pix_to_face.png")

    '''from utils_uv_debug import debug_uv_to_image

    debug_uv_to_image(
        verts_uvs=verts_uvs,          # (Vt,2)
        faces_uvs=faces_uvs,          # (F,3)
        tex_size=1024,
        filename="debug_uv_3dpoints.png",
        device="cuda"
    )'''

    valid_mask = pix_to_face >= 0                      # (H,W)
    #先ほどのpix_to_faceの条件を用いている
    #True → メッシュのどこかに対応しているUVピクセル
    # False → メッシュ外（空白部分）

    #「ラスタライズ」とは何をする処理か（まとめ）
    # グラフィックス的に言うとラスタライズは：
    # 連続的な三角形メッシュ（連続空間）を、離散的なピクセルグリッド（画像）上に割り当てる処理
    # もっと具体的には：
    # 三角形メッシュをカメラから見た画面へ射影する
    # 画面上の各ピクセルごとに
    # どの三角形がそこを覆っているか？（→ pix_to_face）
    # その三角形の中のどの位置か？（→ bary_coords）
    # どれくらい手前/奥か？（→ zbuf）
    # を計算してくれる処理です。
    num_valid = int(valid_mask.sum())#有効なピクセル数
    #print("UV pix_to_face valid pixels:", num_valid, "/", H_tex * W_tex)

    if num_valid == 0:
        # 一応全部0で返す（デバッグ用）
        pts_world_dummy = torch.zeros((1, H_tex, W_tex, 3), device=device)
        return pts_world_dummy, valid_mask.unsqueeze(0)

    # ===== 6. 各 UV ピクセルに対応する 3D 頂点座標を求める =====
    safe_face_idx = pix_to_face.clone()#copyと同じ
    safe_face_idx[~valid_mask] = 0#「~」はnotの意味boolが反転
    #valid=False）の画素に対しても face index を 0 にしておく。
    # こうしないと後のインデックス参照でエラーになるので「仮の値」を入れている。
    #fがないところは-1だからそれを0にした？->正しい、次のコードで配列の引数に入れるため
    
    # faces: (1,F,3) → faces[0]: (F,3)
    faces_idx = faces[0][safe_face_idx]      # (H,W,3)
    #safe_face_idx: (H,W) → 各ピクセルが「どの三角形」に属するかで配列のvalueがfなので、
    #facesの(F,3)のFに対応するだからどの(H,W)が(v1,v2,v3)に対応するかがわかった
    face_verts = verts[0][faces_idx]         # (H,W,3,3)
    #vertsはvの(x,y,z)をもつので、faces_idxのもつv1~v3の頂点を入れるとその座標がわかる
    #これで頂点の(x,y,z)と(u,v)がつなげた(まだ3Dピクセルとはつなげられてない)
    verts_normals = mesh.verts_normals_padded()[0]  # (V,3)
    #////////////////////////////////////////////////正面と背面をわける
    face_norms = verts_normals[faces_idx]
    normals = torch.sum(
        bary_coords.unsqueeze(-1) * face_norms,
        dim=-2
    )  # (Ht,Wt,3)
    #//////////////////////////////////////////////正面と背面をわける
    pts_world = torch.sum(
        bary_coords.unsqueeze(-1) * face_verts,   # (H,W,3,1) * (H,W,3,3)
        #unsqueeze(-1)で最後に次元を追加(H,W,3)->(H,W,3,1)
        #p=w0​v0​+w1​v1​+w2​v2​を計算
        #v0とかは頂点にばらされている？->正しい
        # なんで(x,y,z)と(u,v)の対応分かったのにpって求める必要ある？
        #->求める必要がある。pは3D座標のピクセルにあたるから
        #p=w0​v0​+w1​v1​+w2​v2​,w0​+w1​+w2​=1でpが必ずこの三角形内であると保証される
        #pは必ずしも3D座標上のピクセルの中心ではない。あくまで内部であることが保証される
        dim=-2#頂点方向らしいがなんで？->どのように和を取るかを決めている
        #p=w0​v0​+w1​v1​+w2​v2なので頂点でとればいい
        #掛け算の結果は(H,W,3,3)で、右から2番目の要素が頂点、右端が座標、よって-2で右から二番目を選ぶ
    )  # (H,W,3)

    pts_world = pts_world.unsqueeze(0)           # (1,H,W,3)
    valid_mask = valid_mask.unsqueeze(0)  
    normals = normals.unsqueeze(0)    
    '''print("pts_world:", pts_world.shape)
    print("valid_mask:", valid_mask.shape)
    print("UV rasterized valid pixels:", int(valid_mask.sum()), "/", H_tex * W_tex)'''
    #print("pts_world device:", pts_world.device)

    #単にunsqueeze(0)で先頭にバッチ次元を追加しただけ
    return pts_world, valid_mask, normals

#tex_pred

#verts
#y_norm
# =========================
# 2. CNN エンコーダ
# =========================

class ImageEncoder(nn.Module):#nn.Modileを継承
    """
    ResNet18 の最後の conv フィーチャーマップを使う
      入力: (B,3,H,W)
      出力: (B,C,Hf,Wf)

      Resnetの構造

      conv1 (最初の 7x7 畳み込み)
      bn1 (BatchNorm)
      relu
      maxpool
      layer1 (残差ブロックのグループ1)
      layer2
      layer3
      layer4 ← ここまでが「畳み込み部分」
      avgpool (AdaptiveAvgPool2d)
      fc (全結合層: 512 → 1000)
    """
    #resnet18
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        #torchvision.models.resnet18を用いて学習済みResNet18をつくる
        # weights=models.ResNet18_Weights.IMAGENET1K_V1
        # → ImageNet（1000クラス分類）の事前学習済みパラメータを読み込む指定です。

        # layer2
        backbone.layer2[0].conv1.stride = (1, 1)
        backbone.layer2[0].downsample[0].stride = (1, 1)#stride=(1,1)本来は(2,2)で画像サイズが1/2になるのだが
        #細かい特徴を失うので(1,1)にして同じ大きさを保っている

        # layer3
        backbone.layer3[0].conv1.stride = (1, 1)
        backbone.layer3[0].downsample[0].stride = (1, 1)

        # layer4
        backbone.layer4[0].conv1.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)

        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # conv5_x まで
        #backbone.children()
        # ResNet18 を構成している「サブモジュール」を順番に返すイテレータ。
        # 上で挙げた conv1, bn1, ..., layer4, avgpool, fc が順に出てくる。
        #[:-2]
        # 「最後の2つを除いた部分だけ取る」という意味。
        # ResNet18 では最後の2つは avgpool と fc なので、それを削除するイメージ。
        # 残るのは conv1 〜 layer4 まで（= 最後の畳み込みブロック conv5_x 相当）
    def forward(self, x):
        return self.encoder(x)
    #forward は PyTorch の「順伝播の定義」。
    # x は画像バッチ (B,3,H,W)。


# =========================
# 3. UV 空間の各 texel に対応する 3D 座標を計算する関数
# =========================
#////////////////////////////////////////////////////////////////////////////////////////いらない//////////////////////////////
def compute_front_uv_mask(mesh_cpu: Meshes, tex_size=TEX_SIZE, img_size=IMG_SIZE):
    """
    3Dメッシュを「正面カメラ」でレンダリングしたときに
    見えている面(face)だけを UV 空間に対応させたマスクを返す。

    return: front_mask_uv (1,1,Ht,Wt) float (0 or 1)
    """

    mesh = mesh_cpu.to(device)
    verts = mesh.verts_padded()   # (1,V,3)
    faces = mesh.faces_padded()   # (1,F,3)
    F = faces.shape[1]#面の数(整数スカラー)
    cameras_front = OrthographicCameras(
                device=device,
                R=torch.tensor([[
                [-1., 0., 0.],
                [0., 1., 0.],
                [0., 0., -1.],
                ]], device=device),
                T=torch.tensor([[0., 0., 1.]], device=device)
            )
    '''rendered = render_front_view(mesh, cameras_front, image_size=IMG_SIZE)
    rendered_img = rendered[0, ..., :3].detach().cpu()

    from torchvision.utils import save_image
    save_image(rendered_img.permute(2,0,1), "front_render_1.png")
    print("Saved front_render_1.png")'''

    raster_settings_img = RasterizationSettings(
        image_size=img_size,
        faces_per_pixel=1,
        blur_radius=0.0,
    )
    rasterizer_img = MeshRasterizer(
        cameras=cameras_front,
        raster_settings=raster_settings_img
    )

    fragments_img = rasterizer_img(mesh)
    pix_to_face_img = fragments_img.pix_to_face[0, ..., 0]  # (H_img, W_img)

    # 画像上で実際に見えている face index の集合
    visible_faces = pix_to_face_img[pix_to_face_img >= 0].unique()
    #Trueの位置の値を抽出して1次元配列にする->そしてuniqueで重複削除
    #print("visible_faces count:", visible_faces.numel())

    # ========= 2. UV 空間のラスタライズ =========
    verts_uvs = mesh.textures.verts_uvs_padded()[0].to(device)  # (Vt,2)
    faces_uvs = mesh.textures.faces_uvs_padded()[0].to(device)  # (F,3)

    # UV→NDC（前にうまくいったやつ）
    uv = verts_uvs.clone()
    #///////////////////変更してみた//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #uv[:, 0] = 1.0 - uv[:, 0]
    uv[:, 0] = uv[:, 0] * 2.0 - 1.0
    #////////////////////////////////////////////////変更してみた//////////////////////////////////////////////////////////////////////////////////////////////////
    #uv[:, 1] = 1.0 - uv[:, 1]
    uv[:, 1] = uv[:, 1] * 2.0 - 1.0

    #z = torch.ones_like(uv_ndc[:, :1]) * 1.0
    #uv_verts_3d = torch.cat([uv_ndc, z], dim=-1)

    uv_verts_3d = torch.cat(
        [uv, torch.zeros_like(uv[:, :1])],
        dim=-1
    )  # (Vt,3)#(Vt,2)から(Vt,3)にした

    uv_mesh = Meshes(
        verts=[uv_verts_3d],
        faces=[faces_uvs],
    )

    cameras_uv = OrthographicCameras(
        device=device,
        R=torch.tensor([[
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., 0., -1.],
        ]], device=device),
        T=torch.tensor([[0., 0., 1.]], device=device),
    )
    '''rendered = render_front_view(mesh, cameras_uv, image_size=IMG_SIZE)
    rendered_img = rendered[0, ..., :3].detach().cpu()

    from torchvision.utils import save_image
    save_image(rendered_img.permute(2,0,1), "front_render_2.png")
    print("Saved front_render_2.png")'''

    raster_settings_uv = RasterizationSettings(
        image_size=tex_size,
        faces_per_pixel=1,
        blur_radius=0.0,
    )
    rasterizer_uv = MeshRasterizer(
        cameras=cameras_uv,
        raster_settings=raster_settings_uv
    )

    fragments_uv = rasterizer_uv(uv_mesh)
    pix_to_face_uv = fragments_uv.pix_to_face[0, ..., 0]  # (Ht,Wt)
    #そのピクセルに対応するfが入ってる

    Ht, Wt = pix_to_face_uv.shape
    #print("UV rasterized size:", Ht, Wt)

    # ========= 3. visible_faces に属する face だけ 1 にする =========
    # faceごとに「正面から見えているか」のフラグ
    visible_flags = torch.zeros(F, dtype=torch.bool, device=device)#初期化はすべてFalse
    if visible_faces.numel() > 0:
        visible_flags[visible_faces] = True
    #numel(): 要素数を返す（number of elements）

    # pix_to_face_uv が -1 のところは背景なので 0 にしておく
    safe_face_idx_uv = pix_to_face_uv.clone()
    safe_face_idx_uv[safe_face_idx_uv < 0] = 0

    front_mask_bool = visible_flags[safe_face_idx_uv] & (pix_to_face_uv >= 0)
    front_mask_uv = front_mask_bool.float().unsqueeze(0).unsqueeze(0)  # (1,1,Ht,Wt)
    #学習時に Loss のマスクとして：正面（可視部分）＝教師あり（GTを使う）
    # 背面（不可視）＝予測させる部分（Loss を入れない）という区別が必要。
    # そのために、“正面から見えている face の UV 位置だけ 1 のマスク”
    # を作る必要がある。これがまさに compute_front_uv_mask() の目的です。

    #print("front_mask_uv sum:", front_mask_uv.sum().item())


    return front_mask_uv
#train_model
# =========================
# 4. UV テクスチャ予測ネットワーク
# =========================

class UVTexturePredictor(nn.Module):
    """
    - 正面画像特徴 feat: (B,C,Hf,Wf)
    - UV texelの3D位置 pts_world: (B,Ht,Wt,3)
    - カメラを使って pts_world -> 画面座標 -> feature map 上の座標へ
    - grid_sampleで feat から対応する特徴を取得
    - その特徴を Conv で RGB に変換 -> UVテクスチャ (B,3,Ht,Wt)
    """
    #featはResNet などで抽出した 正面画像の特徴マップ
    #UV texelはUvテクスチャマップの1ピクセルのこと
    #pts_worldは3Dモデルの “頂点そのもの” ではなく、
    # UV テクスチャ画像の各 texel（ピクセル）に対応する 3D 座標
    #pts_world[b, i, j] = (X, Y, Z)  ← 3D空間座標
    def __init__(self, feat_channels):
        super().__init__()

        # =============================
        # 【変更①】C を 64 に downsample
        # =============================
        self.reduce = nn.Conv2d(feat_channels, 64, kernel_size=1)
        #もともと、C=512だった。それを64にした(軽くするため)
        # =============================
        # 【変更②】decoder を軽量化
        # =============================
#tex_gt_b
        '''self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )'''
        self.low = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )



        
#tex
    def forward(self, feat, pts_world, cameras_front, img_size, valid_mask):
        B, C, Hf, Wf = feat.shape
        _, Ht, Wt, _ = pts_world.shape
        #Ht, Wt: UVテクスチャ画像の解像度
        #print("forward R:", cameras_front.R[0].detach().cpu().numpy())
        #print("forward T:", cameras_front.T[0].detach().cpu().numpy())

        # ---- CNN特徴を 64ch に削減 ----
        feat = self.reduce(feat)     # (B,64,Hf,Wf)

        # ------------------------------------
        # pts_world → 画面座標への transform
        # ------------------------------------
        pts_world_flat = pts_world.reshape(B, -1, 3)   # (B, Ht*Wt, 3)
        #print("//////////////////////pts_world_flat",pts_world_flat.shape)
        pts_screen_flat = cameras_front.transform_points_screen(
            pts_world_flat, image_size=img_size
        )  # (B,Ht*Wt,3) 
        #R,Tは実は他の部分で自分で決めていて、今回はその値を持ったcameras_frontを引数に持ってるから指定してない
        #3D点 (X, Y, Z)
        # ↓ カメラ座標系に変換 (R, T)
        # (Xc, Yc, Zc)
        # ↓ 投影（正射影/透視投影）
        # (x_ndc, y_ndc)
        # ↓ 画像サイズへ変換
        # (x_screen, y_screen)
        #このcameraでやりたいのは3DモデルやUVテクスチャマップと入力の正面単一画像をつなぐこと
        #3Dobjを正面からとったものが入力の正面画像と座標が一致するとして対応させている

        pts_screen = pts_screen_flat.reshape(B, Ht, Wt, 3)
        #print(pts_screen[:10])
        #pts_screenは単に深度マップ(x,y,depth)
        #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        '''with torch.no_grad():
    # pts_screen: (B, Ht, Wt, 3)
            pts = pts_screen[0].detach().cpu().numpy()   # (Ht, Wt, 3)
            import numpy as np

    # 正規化して可視化しやすくする
            vis = (pts - pts.min()) / (pts.max() - pts.min() + 1e-6)
            vis = (vis * 255).astype(np.uint8)

            from PIL import Image
            Image.fromarray(vis).save("debug_pts_screen.png")
            print("Saved debug_pts_screen.png")

    # ついでに y だけヒートマップに
            y_map = pts[..., 1]
            y_norm = (y_map - y_map.min()) / (y_map.max() - y_map.min() + 1e-6)
            y_img = (y_norm * 255).astype(np.uint8)
            Image.fromarray(y_img).save("debug_pts_y.png")
            print("Saved debug_pts_y.png")

            print("pts_screen x-range:", pts[...,0].min(), pts[...,0].max())
            print("pts_screen y-range:", pts[...,1].min(), pts[...,1].max())
            print("pts_screen z-range:", pts[...,2].min(), pts[...,2].max())'''
        #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        xs = pts_screen[..., 0]#pts_screen[..., 0] = pts_screen[:, :, :, 0]と同じ
        ys = pts_screen[..., 1]#shapeは(B, Ht, Wt)

        H_img, W_img = img_size

        x_norm = 2.0 * (xs / (W_img - 1.0)) - 1.0
        #[-1,1]にしてる
        y_norm = 2.0 * (ys / (H_img - 1.0)) - 1.0
        #shapeは(B, Ht, Wt)


        #ここまでで：
        # UVの各 (i,j) texel が、正面特徴マップ上のどの (x,y) を見ればいいか
        # が分かった状態。
        grid = torch.stack([x_norm, y_norm], dim=-1)  # (B,Ht,Wt,2)最後の2は0で正面画像のx座標、1でy座標
        #print("grid x min/max:", grid[...,0].min().item(), grid[...,0].max().item())
        #print("grid y min/max:", grid[...,1].min().item(), grid[...,1].max().item())
# b=0 だけ保存（重いので）
        #stackの動作dim=-1で最後に新しい次元を追加して結合
        #print("grid min/max:", float(grid.min()), float(grid.max()))

        #ここまでで：
        # UVの各 (i,j) texel が、正面特徴マップ上のどの (x,y) を見ればいいか
        # が分かった状態。
        #print("grid min/max:", grid.min().item(), grid.max().item())

        #/////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # ================================
#  BACKSIDE grid scatter debug
# ================================
        '''with torch.no_grad():
        # 1. UV → front への grid (すでに計算済み)
    #    grid: (B, Ht, Wt, 2)
            g = grid[0].detach().cpu().numpy()    # (Ht, Wt, 2)
            vm = valid_mask[0].detach().cpu().numpy()   # (Ht, Wt)

    # 2. backside: valid_mask=0 の texel を抽出
            back_mask = (vm == 0)

    # 3. grid の x,y を抽出
            gx = g[..., 0]
            gy = g[..., 1]

    # 4. 背面 texel の grid の範囲
            print("=== BACKSIDE grid range ===")
            print("back grid x min/max:", gx[back_mask].min(), gx[back_mask].max())
            print("back grid y min/max:", gy[back_mask].min(), gy[back_mask].max())

    # 5. scatter で保存
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,6))
            plt.scatter(gx[back_mask], gy[back_mask], s=1, alpha=0.5)
            plt.xlim([-2, 2])
            plt.ylim([ 2,-2])   # 画面の上を上にしたければ反転
            plt.title("Backside texel grid distribution")
            plt.savefig("debug_backside_grid_scatter.png", dpi=200)
            plt.close()

            print("Saved debug_backside_grid_scatter.png")'''

        #/////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # grid visualization
        # ======== grid visualization ======== 
        '''with torch.no_grad():
            grid_vis = (grid[0].detach().cpu() + 1) / 2  # [-1,1] → [0,1]
            from torchvision import transforms
            to_pil = transforms.ToPILImage()
    # (H,W,2) → (2,H,W)
            pil_img = to_pil(grid_vis.permute(2,0,1))
            pil_img.save("grid_debug.png")
            print("Saved grid_debug.png")'''
# ====================================#mask
        feat_tex = F.grid_sample(
            feat,                # (B,64,Hf,Wf)
            grid,                # (B,Ht,Wt,2)
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )  # (B,64,Ht,Wt)
        #grid_sample の意味：
        # 入力 feat を「ソース」と見て、
        # grid[b,i,j] = (x_norm, y_norm) の位置から 補間した特徴ベクトル (64次元) を取り出し、
        # (B,64,Ht,Wt) として並べたものが feat_tex
        # 直感的には：
        # 「正面画像から見た特徴マップを、UV空間に '引き伸ばす / 引き写す' 処理」

        # ---- デコードして RGB に ----
        low_feat = self.low(feat)            # (B,32,Hf,Wf)
        low_tex = F.grid_sample(
            low_feat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )  # (B,32,Ht,Wt)

        f = self.dec1(feat_tex)              # (B,128,Ht,Wt)
        f = torch.cat([f, low_tex], dim=1)   # (B,128+32=160,Ht,Wt)
        f = self.dec2(f)                     # (B,64,Ht,Wt)

        tex_pred = self.final(f)             # (B,3,Ht,Wt)
        #ここで得られるのは：
        # 「正面画像の情報を元にした '予測UVテクスチャ' 」
        valid_mask_3 = valid_mask.unsqueeze(1) # (B,1,Ht,Wt)
        #valid_mask: たぶん (B,Ht,Wt) で
        # 1 → 有効なUV texel
        # 0 → 無効（メッシュ外 or 欠損領域）
        tex_pred = tex_pred * valid_mask_3
        #テクスチャの有効範囲だけ色を残す 処理。
        '''print("tex_pred:", tex_pred.shape)
        print("valid_mask_dtype: ",valid_mask.dtype)
        print("valid_mask_3_dtype: ",valid_mask_3.dtype)
        print("tex_pred_mean: ",tex_pred.mean(dim=[0,2,3]))
        print("tex_pred min/max:", float(tex_pred.min()), float(tex_pred.max()))'''
        # 例: 有効な UV texel だけ見て min/max を調べる
        '''valid = valid_mask[0] > 0
        gy = grid[0, ..., 1][valid]
        gx = grid[0, ..., 0][valid]

        print("valid grid x:", gx.min().item(), gx.max().item())
        print("valid grid y:", gy.min().item(), gy.max().item())'''



        return tex_pred, valid_mask_3, grid
#深度マップ
#/////////////////////////////////////////////NDC変換が正しいか判定する//////////////////////////////////////////////////////////
import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
    OrthographicCameras,
)
from PIL import Image

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

def load_checkpoint(path, encoder, tex_predictor, optimizer, device):
    checkpoint = torch.load(path, map_location=device)

    encoder.load_state_dict(checkpoint["encoder"])
    tex_predictor.load_state_dict(checkpoint["tex_predictor"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # ★ GPU に配置（これ必須）
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



#///////////////////////////////////////////////////////////////////////////いらない//////////////////////
def uv_updown_test(mesh, tex_size=512, filename="uv_updown_test.png"):
    import numpy as np
    from PIL import Image
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        MeshRasterizer, RasterizationSettings, OrthographicCameras
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("device:", device)

    # === GPU に載せる ===
    mesh = mesh.to(device)

    verts_uvs = mesh.textures.verts_uvs_padded()[0]     # (Vt,2)
    faces_uvs = mesh.textures.faces_uvs_padded()[0]     # (F,3)

    # UV → NDC#/////////////////////////////////////ここ変える///////////////////
    uv_ndc = torch.zeros_like(verts_uvs)
    uv_ndc[:, 0] = 1 - verts_uvs[:, 0] * 2
    #uv_ndc[:, 0] = verts_uvs[:, 0] * 2 - 1
    #uv_ndc[:, 1] = 1 - verts_uvs[:, 1] * 2
    uv_ndc[:, 1] = verts_uvs[:, 1] * 2 - 1
    '''uv_verts_3d = torch.cat(
        [uv_ndc, torch.zeros_like(uv_ndc[:, :1])],
        dim=-1
    )'''
    z = torch.ones_like(uv_ndc[:, :1])*1.0
    uv_verts_3d = torch.cat([uv_ndc, z], dim=-1)

#uv_verts_3d
    # === GPU に Meshes を再構築 ===
    uv_mesh = Meshes(
        verts=[uv_verts_3d.to(device)],
        faces=[faces_uvs.to(device)]
    )

    # === GPU カメラ ===
    cameras = OrthographicCameras(device=device)
#Saved front_render.png
    # Rasterize
    settings = RasterizationSettings(
        image_size=tex_size,
        faces_per_pixel=1
    )

    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=settings
    )

    fragments = rasterizer(uv_mesh)

    pix_to_face = fragments.pix_to_face[0,:,:,0].cpu().numpy()
    bary = fragments.bary_coords[0,:,:,0,:].cpu().numpy()

    verts_uvs_np = verts_uvs.cpu().numpy()
    faces_uvs_np = faces_uvs.cpu().numpy()

    H, W = tex_size, tex_size
    out_img = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            f = pix_to_face[y, x]
            if f < 0:
                continue
            v0, v1, v2 = faces_uvs_np[f]
            b0, b1, b2 = bary[y, x]

            uv_v = (
                verts_uvs_np[v0][1] * b0 +
                verts_uvs_np[v1][1] * b1 +
                verts_uvs_np[v2][1] * b2
            )
            out_img[y, x] = uv_v

    out_rgb = np.stack([out_img]*3, axis=-1)
    out_rgb = (out_rgb * 255).clip(0,255).astype(np.uint8)
    Image.fromarray(out_rgb).save(filename)
    pts_view = cameras.get_world_to_view_transform().transform_points(uv_verts_3d)
    print("view z-range_uv_updown_test", pts_view[...,2].min(), pts_view[...,2].max())
    print("pix_to_face unique:", np.unique(pix_to_face))
    print("uv_verts_3d min/max:", uv_verts_3d.min().item(), uv_verts_3d.max().item())

    zbuf = fragments.zbuf[0,:,:,0].cpu().numpy()
    print("zbuf min/max:", zbuf.min(), zbuf.max())

    print(f"Saved: {filename}")

#valid_mask_3print("pix_to_face unique:", np.unique(pix_to_face))

import torch


#/////////////////////////////////////////////////////////////////////////いらない/////////////////////
def generate_face_uv_mask_from_grid(img_face_mask, grid, valid_mask):
    """
    img_face_mask: (1,1,H_img,W_img)  0 or 1（外部で作成された顔セグメント）
    grid:          (1,Ht,Wt,2)        UV→正面画像マッピング（grid_sampleと同じ）
    valid_mask:    (1,1,Ht,Wt)        UVの有効領域

    return face_uv_mask: (1,1,Ht,Wt)
    """
    device = grid.device

    # 正面画像の解像度
    _, _, H_img, W_img = img_face_mask.shape

    # === NDC (-1~1) → pixel index ===
    # x_img = (-1~1) → (0 ~ W_img-1)
    x_img = ((grid[..., 0] + 1.0) * 0.5 * (W_img - 1))
    y_img = ((grid[..., 1] + 1.0) * 0.5 * (H_img - 1))

    # long tensor に変換
    x_img = x_img.long()
    y_img = y_img.long()

    # 出力マスクの初期化
    Ht, Wt = x_img.shape[1], x_img.shape[2]
    face_uv_mask = torch.zeros((1, 1, Ht, Wt), dtype=torch.float32, device=device)

    # === 各 UV texel について、対応する正面画像のピクセルを参照 ===
    for i in range(Ht):
        for j in range(Wt):
            xi = x_img[0, i, j].item()
            yi = y_img[0, i, j].item()

            # 範囲外は無視
            if xi < 0 or xi >= W_img or yi < 0 or yi >= H_img:
                continue

            # 顔領域なら1
            if img_face_mask[0, 0, yi, xi] > 0.5:
                face_uv_mask[0, 0, i, j] = 1.0

    # UVで有効な領域だけに限定
    face_uv_mask = face_uv_mask * valid_mask

    return face_uv_mask


# =========================
# 5. 損失関数（L1, smoothness, symmetry）
# =========================


import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image

#uv_map
def debug_check_face_uv_projection(img_path, face_mask, uv_map, face_uv_mask, tex_gt, tag="debug"):
    import numpy as np
    from PIL import Image

    #--------------------------------------------
    # 画像の可視化用に512x512に揃える
    #--------------------------------------------
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img)

    #--------------------------------------------
    # face_mask (512×512) 可視化
    #--------------------------------------------
    face_mask_np = face_mask > 0
    face_vis = img_np.copy()
    face_vis[face_mask_np] = [255, 0, 0]
    Image.fromarray(face_vis).save(f"{tag}_image_face_mask.png")

    #--------------------------------------------
    # UV マスク可視化
    #--------------------------------------------
    uv_mask_np = face_uv_mask[0,0].detach().cpu().numpy()
    uv_mask_img = (uv_mask_np * 255).astype(np.uint8)
    Image.fromarray(uv_mask_img).save(f"{tag}_face_uv_mask.png")

    #--------------------------------------------
    # UV マスクを tex_gt に適用
    #--------------------------------------------
    tex_np = tex_gt[0].detach().cpu().numpy().transpose(1,2,0)
    tex_np = (tex_np * 255).astype(np.uint8)

    mask_t = uv_mask_np[:, :, None]
    tex_face = (tex_np * mask_t).astype(np.uint8)

    Image.fromarray(tex_face).save(f"{tag}_tex_face_only.png")

    print(f"Saved debug images for {tag}")
#uv_map
#/////////////////////////////////////////////////////////////////////////////////


#//////////////////////////////////////////////////////////////////////////////////いらない////////////////////////
def face_mask_to_uv_mask(mesh, face_mask, cameras_front, tex_size=TEX_SIZE):
    """
    mesh: 1サンプルの Meshes
    face_mask: (1,1,H,W) 正面画像の顔マスク (0/1)
    cameras_front: 正面用 Orthographic カメラ
    return: (1,1,tex_size,tex_size) の UV 空間の顔マスク
    """
    device = face_mask.device
    H, W = face_mask.shape[-2:]

    # === 1. 正面から rasterize (高解像度に合わせて1024でもOK)
    rasterizer = MeshRasterizer(
        cameras=cameras_front,
        raster_settings=RasterizationSettings(
            image_size=H,       # IMG_SIZE=512
            blur_radius=0.0,
            faces_per_pixel=1
        )
    )

    fragments = rasterizer(mesh)
    pix_to_face = fragments.pix_to_face[0]        # (H,W)
    bary = fragments.bary_coords[0]              # (H,W,3)
    print("pix_to_face_front min/max:", pix_to_face.min().item(), pix_to_face.max().item())
    print("pix_to_face_front coverage:", (pix_to_face>=0).sum().item())


    # === 2. face_mask と掛けて、顔に対応する pixels だけ残す
    face_pix = (pix_to_face >= 0) & (face_mask[0,0] > 0.5)
    if face_pix.sum() == 0:
        print("WARNING: no face pixels found!")
        return torch.zeros((1,1,tex_size,tex_size), device=device)

    face_index = pix_to_face[face_pix]          # (N,) 顔に映っているFace ID
    bary_face = bary[face_pix]                  # (N,3)

    # === 3. UV 座標を face_id から復元
    verts_uvs = mesh.textures.verts_uvs_padded()[0]   # (Vt,2)
    faces_uvs = mesh.textures.faces_uvs_padded()[0]   # (F,3)
    face_uvs = verts_uvs[faces_uvs]                  # (F,3,2)

    uv = (bary_face[:, None, :] * face_uvs[face_index]).sum(dim=2)   # (N,2)
    uv = uv.clamp(0,1)

    # === 4. UV空間 → ピクセルへ (2048x2048)
    uv_px = (uv * (tex_size - 1)).long()
    mask_uv = torch.zeros((tex_size, tex_size), device=device)

    mask_uv[uv_px[:,1], uv_px[:,0]] = 1.0
    mask_uv = mask_uv[None,None,:,:]   # (1,1,Ht,Wt)
    print("face_mask sum:", face_mask.sum().item())
    face_pix = (pix_to_face >= 0) & (face_mask[0,0] > 0.5)
    print("face_pix sum:", face_pix.sum().item())

    return mask_uv

#pix_to_face
#///////////////////////////////////////////////////////////////////////////////////

import matplotlib.pyplot as plt
#////////////////////////////////////////////////////////////////////////////debugでいらないかも//////////////////////////////
def visualize_back_mask(back_mask, save_path="debug_back_mask.png"):
    """
    back_mask: (1,1,Ht,Wt)
    """
    mask_np = back_mask[0,0].detach().cpu().numpy()

    plt.figure(figsize=(4,4))
    plt.imshow(mask_np, cmap="gray")
    plt.title("Back UV Mask (white=back)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[OK] Saved back_mask visualization -> {save_path}")

from torchvision import transforms
from PIL import Image

import torch
from torchvision import transforms


#////////////////////////////////////////////////////////////////////入れたいならいれてもok////////////////////
def save_human_lr_debug(tex_pred, left_mask,save_prefix="human_lr"):
    """
    人体の左右（3D基準）を可視化
    """
    device = tex_pred.device

    debug = torch.zeros_like(tex_pred)
    debug[:, 0:1] = left_mask  # Rチャンネル
    to_pil = transforms.ToPILImage()
    img = to_pil(debug[0].detach().cpu())
    img.save(f"{save_prefix}_left_body_1.png")

    print(f"[OK] Saved {save_prefix}_left_body_1.png")



#////////////////////////////////////////////////いらない/////////////////////////////////
def flip_symmetry_loss(
    tex_pred,
    tex_flip,
    back_left_mask,
    eps=1e-8
):
    """
    背面 × 人体左側のみで左右対称性を課す loss
    """
    denom = back_left_mask.sum() * tex_pred.shape[1] + eps

    loss = (
        (tex_pred - tex_flip).abs()
        * back_left_mask
    ).sum() / denom

    return loss




def texture_l1_loss(tex_pred, tex_gt, valid_mask, face_uv_mask, cloth_uv_mask, front_mask,pts_world, w_face=10.0, alpha_color=0.3, alpha_mean=0.2, alpha_std=0.1):
    """
    tex_pred, tex_gt: (B,3,Ht,Wt)
    valid_mask      : (B,1,Ht,Wt)
    face_uv_mask    : (B,1,Ht,Wt)  ← 顔領域のみ1
    """
    '''with torch.no_grad():
        tex_pred_np = tex_gt[0].detach().cpu().numpy().transpose(1,2,0)
        Image.fromarray((tex_pred_np*255).clip(0,255).astype("uint8")).save("vis_tex_gt_1.png")'''
    #========================================================
    # 0) 基本の L1
    #========================================================
    diff = (tex_pred - tex_gt) * valid_mask
    denom = valid_mask.sum() * tex_pred.shape[1] + 1e-8
    back_mask=valid_mask*(1.0-front_mask)
    denom_back=back_mask.sum()*tex_pred.shape[1]+1e-8
    l1_base = diff.abs().sum() / denom
    #visualize_back_mask(back_mask)
    l1_back = (
        (tex_pred - tex_gt).abs()
        * back_mask
    ).sum() / denom_back
    #center_x = pts_world[..., 0].mean()
    left_body_mask = (pts_world[..., 0:1] < 0).float()   # X < 0
    left_body_mask = left_body_mask.permute(0,3,1,2)
    left_body_mask = left_body_mask * valid_mask

    # 背面 × 左側
    back_left_mask = back_mask * left_body_mask
    tex_flip = torch.flip(tex_pred, dims=[3])

    loss_sym = flip_symmetry_loss(
        tex_pred,
        tex_flip,
        back_left_mask
    )
    #save_human_lr_debug(tex_pred,back_left_mask)
    # 左半分を赤く

#debug_tex_pred_face
    #========================================================
    # 1) 色のついている部分を強めに見る L1（既存）
    #========================================================
    with torch.no_grad():
        gray = tex_gt.mean(dim=1, keepdim=True)
        weight_color = torch.where(
            gray > 0.05,
            torch.ones_like(gray),
            0.5 * torch.ones_like(gray)
        )

    l1_color = ((tex_pred - tex_gt).abs()
                * valid_mask
                * weight_color).sum() / denom

    #========================================================
    # 2) 前景領域の平均色合わせ（既存）
    #========================================================
    gt_fg = tex_gt * valid_mask
    pred_fg = tex_pred * valid_mask

    gt_mean = gt_fg.sum(dim=(2,3)) / (valid_mask.sum(dim=(2,3)) + 1e-8)
    pred_mean = pred_fg.sum(dim=(2,3)) / (valid_mask.sum(dim=(2,3)) + 1e-8)

    loss_mean = (gt_mean - pred_mean).abs().mean()

    #========================================================
    # 3) 前景領域の標準偏差合わせ（既存）
    #========================================================
    gt_std = gt_fg.std(dim=(2,3))
    pred_std = pred_fg.std(dim=(2,3))
    loss_std = (gt_std - pred_std).abs().mean()

    #========================================================
    # 4) ★ 顔領域を強調する weighted L1（新追加）
    #========================================================
    # 顔だけweightを大きくする: 例 w_face = 10〜30


    face_weight = 1.0 + face_uv_mask * (w_face - 1.0)
    l1_face = ((tex_pred - tex_gt).abs()
               * valid_mask
               * face_weight).sum() / denom
    #///////////////////////////////////////////////服領域を強調//////////////////////
    cloth_weight = 1.0 + cloth_uv_mask * (w_face - 1.0)
    l1_cloth = ((tex_pred - tex_gt).abs()
               * valid_mask
               * cloth_weight).sum() / denom
    loss = (
        0.1*l1_base
        + alpha_color * l1_color
        #+ alpha_mean * loss_mean
        #+ alpha_std * loss_std
        + 2*l1_back
        #+ 0.005*loss_sym
        + 0.01 * l1_face
        + 0.1*l1_cloth        
    )
    #///////////////////////////////////////////////////////////////////////////////////////////////////
    '''with torch.no_grad():

    # ----- 0. Face UV Mask (white-black) -----
        mask_np = face_uv_mask[0,0].cpu().numpy()
        Image.fromarray((mask_np*255).astype("uint8")).save("vis_face_uv_mask_bw.png")

    # ----- 1. tex_gt の顔領域 -----
        tex_gt_np = tex_gt[0].detach().cpu().numpy().transpose(1,2,0)
        tex_gt_face = tex_gt_np * mask_np[:,:,None]
        Image.fromarray((tex_gt_face*255).clip(0,255).astype("uint8")).save("vis_tex_gt_face.png")

    # ----- 2. tex_pred の顔領域 -----
        tex_pred_np = tex_pred[0].detach().cpu().numpy().transpose(1,2,0)
        tex_pred_face = tex_pred_np * mask_np[:,:,None]
        Image.fromarray((tex_pred_face*255).clip(0,255).astype("uint8")).save("vis_tex_pred_face.png")

    # ----- 3. tex_gt × mask overlay（赤塗り） -----
        overlay = (tex_gt_np * 255).clip(0,255).astype("uint8")
        overlay[mask_np > 0.5] = [255, 0, 0]
        Image.fromarray(overlay).save("vis_tex_gt_overlay_face.png")

    # ----- 4. Colormap visualization -----
        plt.imsave("vis_face_uv_mask_colormap.png", mask_np, cmap="jet")'''
    #///////////////////////////////////////////////////////////////////////////////////////////////////
    return loss


#/////////////////////////////////////////////////////////////いらない/////////////////////
def texture_smoothness_loss(tex_pred, valid_mask):
    """
    近傍画素間の差分をペナルティ（全体 smoothness）
    """
    #tex_pred, tex_gt: (B,3,Ht,Wt)
    B, C, H, W = tex_pred.shape
    vm = valid_mask

    dx = (tex_pred[:, :, 1:, :] - tex_pred[:, :, :-1, :]) * vm[:, :, 1:, :]
    #tex_pred[:, :, 1:, :]これは 高さ方向を 1 から H-1 まで切り取ったもの。
    #つまりこれは、(i=1, j), (i=2, j), ..., (i=H-1, j)の画素が並んでいます。
    #一方tex_pred[:, :, :-1, :]これは 高さ方向を 0 から H-2 まで切り取ったもの。
    #よってインデックスが1つズレているので(i,j)と(i-1,j)の差になる
    dy = (tex_pred[:, :, :, 1:] - tex_pred[:, :, :, :-1]) * vm[:, :, :, 1:]
    #なぜこれが滑らかさになるかというと、差を取っていてこの差が小さいほど変化が小さい
    #つまりなめらかである

    loss = dx.abs().mean() + dy.abs().mean()#小さいほど滑らか
    return loss


#//////////////////////////////////////////////////////////////いらない//////////////////////////////////////////
def texture_symmetry_loss(tex_pred, valid_mask):
    """
    横方向（U方向）の対称性を簡易的に仮定して L1。
    実際は人体UVの構造に合わせてマスクを細かく設計した方がよい。
    """
    #tex_pred, tex_gt: (B,3,Ht,Wt)
    #横方向に反転したテクスチャを作る
    tex_flip = torch.flip(tex_pred, dims=[3])  # W方向を反転
    vm = valid_mask
    diff = (tex_pred - tex_flip) * vm
    denom = vm.sum() * tex_pred.shape[1] + 1e-8
    return diff.abs().sum() / denom

#//////////////////////////////////////////////////////////////////////顔座標のみとりだし///////////////////////////////////////////////////////////////

def make_face_mask_from_bbox(bbox, H, W, device):
    # bbox: (x1,y1,x2,y2), pixel座標
    mask = torch.zeros((H, W), device=device)
    x1,y1,x2,y2 = bbox
    mask[y1:y2, x1:x2] = 1.0
    return mask

#////////////////////////////////////////////////////////////////////////////////

#////////////////////////////服領域抽出/////////////////////////////////////////////////////////////////
import numpy as np
from PIL import Image
import torch
#//////////////////////////////////////////////////////////////いらない////////////////////////////
def load_binary_mask_png(path, device="cuda"):
    # === PNG画像を読み込む ===
    img = np.array(Image.open(path))

    # === shape確認 ===
    # 服maskがRGB(3ch)の場合はグレーにする
    if img.ndim == 3:
        # 白(255,255,255)以外 → mask とする
        img_gray = img.mean(axis=2)   # (H,W)
    else:
        img_gray = img                # (H,W)

    # === 0〜1 のバイナリmaskに変換 ===
    mask_np = (img_gray < 128).astype(np.float32)  # 白以外を1にしたいならここ変更

    # === torch tensor に変換 ===
    mask_t = torch.from_numpy(mask_np).to(device)

    return mask_t

#////////////////////////////服領域抽出/////////////////////////////////////////////////////////////////

#///////////////////////////////////////////////////////////////いらない/////////////////////////////////
def compute_image_to_uv_map(mesh, cameras, img_size):
    device = mesh.device
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
#uv_valid
    fragments = rasterizer(mesh)
    pix_to_face = fragments.pix_to_face[0]      # (H,W,1)
    bary = fragments.bary_coords[0]             # (H,W,1,3)

    H, W = pix_to_face.shape[:2]

    verts_uvs = mesh.textures.verts_uvs_padded()[0]
    faces_uvs = mesh.textures.faces_uvs_padded()[0]

    face_uvs = verts_uvs[faces_uvs]                 # (F,3,2)

    # valid pixels
    valid_pix = (pix_to_face[..., 0] >= 0)          # (H,W)

    # ---------------
    # 全ピクセル分まとめて計算
    # ---------------
    face_idx = pix_to_face[..., 0][valid_pix]       # (N_valid,)
    bary_valid = bary[valid_pix].squeeze(1)         # (N_valid,3)

    uv_valid = (bary_valid.unsqueeze(-1) * face_uvs[face_idx]).sum(dim=1)
    # (N_valid,2)
    # uv_valid: (N_valid,2) in [0,1]
    u = uv_valid[:, 0]
    v = uv_valid[:, 1]

# 左右反転
    u = 1.0 - u

# 上下反転
    v = 1.0 - v

    uv_valid = torch.stack([u, v], dim=1)


    # 結果を貼り付け
    uv_map = torch.zeros((H, W, 2), device=device)
    uv_map[valid_pix] = uv_valid

    return uv_map

#grid = torch.stack([x_norm, y_norm], dim=-1)
#/////////////////////////////////////////////////////////////////////この関数はいらない/////////////////////////
def project_face_to_uv(face_mask, uv_map, tex_size, device):
    Ht, Wt = tex_size
    uv_mask = torch.zeros((Ht, Wt), device=device)

    ys, xs = torch.where(face_mask > 0)

    u = uv_map[ys, xs, 0]
    v = uv_map[ys, xs, 1]

    ui = (u * (Wt - 1)).long().clamp(0, Wt - 1)
    vi = (v * (Ht - 1)).long().clamp(0, Ht - 1)

    uv_mask[vi, ui] = 1.0
    return uv_mask.float().unsqueeze(0).unsqueeze(0)

#face_uv_mask
#//////////////////////////////////////////////////////////////////////顔座標のみとりだし///////////////////////////////////////////////////////////////

# =========================
# 6. 学習ループ
# =========================#project_face_to_uv
def mesh_collate_fn(batch):
    """
    batch: list of (img_tensor, mesh, tex_gt)
    return:
        imgs: Tensor(B,3,H,W)
        meshes: list of Meshes (len=B)
        tex_gts: Tensor(B,3,Ht,Wt)
    """
    #batchの構造を分けている
    imgs = torch.stack([b[0] for b in batch], dim=0)#画像だけスタックして (B,3,H,W) にする
    # img1 → shape = (3,H,W)
    # img2 → shape = (3,H,W)
    # img3 → shape = (3,H,W)
    #imgs =[
    # img1
    # img2
    # img3
    #]
    #stackはまとめるイメージ
    meshes = [b[1] for b in batch]     # ← Meshes は list として保持
    tex_gts = torch.cat([b[2] for b in batch], dim=0)
    bboxes = torch.stack([b[3] for b in batch], dim=0)
    img_paths = [b[4] for b in batch]  
    return imgs, meshes, tex_gts, bboxes,img_paths


import time##save_models
def train_model(dataloader, encoder, tex_predictor, optimizer, resume_path=None):
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
#debug_uv_from_front#uv_map_debug
    loss_log = []
    start_epoch = 0
    ACCUM = 8
#MeshRenderer
    if resume_path is not None:
        start_epoch, loss_log = load_checkpoint(resume_path, encoder, tex_predictor, optimizer, device)
        '''print(next(encoder.parameters()).device)
        print(next(tex_predictor.parameters()).device)
        for k, v in optimizer.state.items():
            for kk, vv in v.items():
                if torch.is_tensor(vv):
                    print("opt state device:", vv.device)
                break
            break'''

        print("Resuming training...")

    for epoch in range(start_epoch,NUM_EPOCHS):
        epoch_start_time = time.time()
        accum_counter=0
        optimizer.zero_grad(set_to_none=True)
        #prev_loss_raw=0
        judge=True
        loss_loss=0
        for batch_idx, (img_tensor, meshes, tex_gt, bboxes, cloth_mask) in enumerate(dataloader):
            
            B = img_tensor.shape[0]

            img_tensor = img_tensor.to(device)
            tex_gt = tex_gt.to(device)

            # CNN特徴を全バッチで一度に計算
            feat = encoder(img_tensor)   # (B, C, Hf, Wf)

            total_loss = 0.0
            
            
            for b in range(B):
                #mesh_cpu = meshes[b] 
                #from debug_polygon import visualize_uv_with_colors
                #visualize_uv_with_colors(mesh_cpu.cpu(), out="uv_debug.png")
            

                mesh = meshes[b].to(device)
                cameras_front = OrthographicCameras(
                    device=device,
                    R=torch.tensor([[[-1.,0.,0.],
                                     [ 0.,1.,0.],
                                     [ 0.,0.,-1.]]], device=device),
                    T=torch.tensor([[0.,0.,1.]], device=device),
                    #focal_length=((-scale_x, -scale_y),),
                    in_ndc=True,
                )
                
                # 3D座標補正（あなたの実験）
                verts = mesh.verts_padded()
                verts = verts * 0.8
                verts[...,1] -= (1.01)/512
                verts[...,0] -= 0.146/512
                mesh = mesh.update_padded(verts)
                verts2d = cameras_front.transform_points(verts)[0]

                '''rendered = render_front_view(mesh, cameras_front, image_size=IMG_SIZE)
                rendered_img = rendered[0, ..., :3].detach().cpu()
                from torchvision.utils import save_image
                print("#/////////////////////////",rendered_img.shape)
                save_image(rendered_img.permute(2,0,1), "front_render_100.png")'''
                #print("train R:", cameras_front.R[0].detach().cpu().numpy())
               # print("train T:", cameras_front.T[0].detach().cpu().numpy())


                # ========== UV→3D ==========
                pts_world, valid_mask,normals = compute_uv_to_3d_points(mesh, cameras_front,TEX_SIZE)
                # shape: (1,Ht,Wt,3)
                cam_dir_camcoords = torch.tensor([0., 0., 1.], device=device).view(1,3,1)

# world coords の方向ベクトル
                view_dir = cameras_front.R.transpose(1,2) @ cam_dir_camcoords   # (1,3,1)

                view_dir = view_dir.view(1,1,1,3)   # (1,1,1,3)#


    #view_dir = torch.tensor([0., 0., -1.], device=device).view(1,1,1,3)

# 内積 → 正面なら値が正
                dot = (normals * view_dir).sum(dim=-1, keepdim=True)   # (1,Ht,Wt,1)

# front-facing マスク作成
                front_mask = (dot < 0).float()  # (1,Ht,Wt,1)
                front_mask = front_mask.permute(0,3,1,2)  # (1,1,Ht,Wt)
                
                # ========== 1サンプルの特徴を取り出し ==========
                feat_b = feat[b:b+1]          # (1,C,Hf,Wf)
                tex_gt_b = tex_gt[b:b+1]      # (1,3,Ht,Wt)
#face_mask_to_uv_mask
#transfer_face_color
                bbox = bboxes[b].cpu().tolist()   # (x1,y1,x2,y2)
                # cloth_mask が list の場合は文字列を取り出す
                if isinstance(cloth_mask, list):
                    cloth_mask = cloth_mask[0]

                
                face_mask = make_face_mask_from_bbox(bbox, IMG_SIZE, IMG_SIZE, device)#(512,512)正面画像で顔部分に1が立っている
        
                #print("face_mask sum:", face_mask.sum().item())
                #uv_map = compute_image_to_uv_map(mesh, cameras_front, TEX_SIZE)

                #print("uv_map U min/max:", uv_map[...,0].min().item(), uv_map[...,0].max().item())
                #print("uv_map V min/max:", uv_map[...,1].min().item(), uv_map[...,1].max().item())
            
                '''from Visualize_uv import visualize_uv_map
                print("uv_map shape:", uv_map.shape)
                visualize_uv_map(uv_map, out="uv_map_debug.png")'''
                #uv_face_color
                '''from debug_uv_mapping import debug_uv_mapping
                if epoch == 0 and batch_idx == 0 and b == 0:
                    debug_uv_mapping(mesh, img_tensor, cameras_front, TEX_SIZE, device)'''

        

                # ========== UVテクスチャ予測 ==========
                tex_pred, valid_mask_3, grid= tex_predictor(
                    feat_b,
                    pts_world,
                    cameras_front,
                    img_size=(IMG_SIZE, IMG_SIZE),
                    valid_mask=valid_mask
                )
                '''from transfer_face_color_to_uv_with_grid_sample import transfer_face_color_to_uv_with_grid_sample
                if epoch == 0 and batch_idx == 0 and b == 0:
                    transfer_face_color_to_uv_with_grid_sample(grid, face_mask, img_tensor[b:b+1], TEX_SIZE)

                print("DEBUG face_mask shape:", face_mask.shape)
                print("DEBUG grid shape:", grid.shape)'''
                face_mask_img = face_mask.unsqueeze(0).unsqueeze(0).float()

                cloth_mask = cloth_mask.to(device) 
                cloth_mask_img = cloth_mask.unsqueeze(0).float()
                #print(cloth_mask_img.shape)

                face_uv_mask = F.grid_sample(
                    face_mask_img,   # (1,1,H_img,W_img)
                    grid,            # (1,Ht,Wt,2)
                    mode='nearest',
                    padding_mode='zeros',
                    align_corners=False
                )
#face_u
                cloth_uv_mask = F.grid_sample(
                    cloth_mask_img,   # (1,1,H_img,W_img)
                    grid,            # (1,Ht,Wt,2)
                    mode='nearest',
                    padding_mode='zeros',
                    align_corners=False
                )
                face_uv_mask = face_uv_mask * valid_mask_3
                cloth_uv_mask=cloth_uv_mask*valid_mask_3

                #print("face_uv_mask shape:", face_uv_mask.shape)
                '''if epoch == 0 and batch_idx == 0 and b == 0:
                    print("face_uv_mask sum:", face_uv_mask.sum().item())
                    print("cloth_uv_mask sum:", cloth_uv_mask.sum().item())'''

                #////////////////////////////////////////////////////////////////////////////////////////////
                '''if epoch == 0 and batch_idx == 0 and b == 0:
                    with torch.no_grad():
                        # UV上の顔マスクを可視化
                        mask_np = cloth_uv_mask[0,0].cpu().numpy().astype("float32")
                        from PIL import Image
                        Image.fromarray((mask_np*255).astype("uint8")).save("debug_face_uv_mask_from_grid.png")

                        tex_gt_img = tex_gt_b[0].detach().cpu().numpy().transpose(1,2,0)
                        overlay = (tex_gt_img * 255).astype("uint8")
                        overlay[mask_np > 0.5] = [255, 0, 0]
                        Image.fromarray(overlay).save("debug_tex_gt_face_overlay.png")'''
                #///////////////////////////////////////////////////////////////////////////////////////////////

                # ========== front mask ==========
                #/////////////////////////////////////////////front_maskは一旦消す//////////////////////////////////////
                #front_mask = compute_front_uv_mask(mesh, TEX_SIZE, IMG_SIZE).to(device)
                #mask = front_mask * valid_mask_3
                #///////////////////////////////////一旦front_maskは無視する////////////////////////////////////////////////////
                valid_uv_mask = valid_mask_3
                if((epoch+1)%50==0 and  batch_idx == 0 and b == 0):
                    with torch.no_grad():
                        tex_img = tex_pred[0].detach().cpu().clamp(0,1)
                        from torchvision import transforms
                        to_pil = transforms.ToPILImage()
                        pil_img = to_pil(tex_img)
                        pil_img.save(f"pred_epoch{epoch+1}_5.png")
                        print(f"Saved: pred_epoch{epoch+1}_5.png")
                '''with torch.no_grad():
                    masked_gt = tex_gt_b * mask
                    masked_pred = tex_pred * mask
                    print("gt std:", masked_gt.std(dim=[0,2,3]))
                    print("pred std:", masked_pred.std(dim=[0,2,3]))'''

                #print("front_mask UV coverage:", front_mask.sum().item())
                if epoch == 0 and batch_idx == 0 and b == 0:
                    print("valid_mask coverage:", valid_uv_mask.sum().item())
                #print("mask coverage:", (front_mask*valid_mask_3).sum().item())
                #os.system("nvidia-smi")
                # ========== 損失 ==========
                '''if epoch==0 and batch_idx==0:
                    with torch.no_grad():
                        zero_pred = torch.zeros_like(tex_pred)
                        baseline_loss = texture_l1_loss(zero_pred, tex_gt_b, mask)
                        print("all-black baseline L1:", baseline_loss.item())'''
                
                # ▼ 画素マスクを生成
                

                loss_l1 = texture_l1_loss(
                    tex_pred, tex_gt_b,
                    valid_uv_mask,
                    face_uv_mask,
                    cloth_uv_mask,
                    front_mask,
                    pts_world,
                    w_face=15.0,
                )
                #save_human_lr_debug(tex_pred,pts_world,valid_mask, save_prefix="lr_sym_check")
                #////////////////////////debug//////////////////////////////////
                #////////////////////////debug/////////////////////
                '''if epoch == 0 and batch_idx == 0:
                    debug_check_face_uv_projection(
                    img_path=img_paths[b],
                    face_mask=face_mask.cpu().numpy(),
                    uv_map=uv_map.cpu().numpy(),
                    face_uv_mask=face_uv_mask,
                    tex_gt=tex_gt_b,
                    tag=f"sample{b}"
                    )'''

                '''if epoch > 1:
                    print("Δloss:", loss_l1.item() - prev_loss_raw)
                prev_loss_raw = loss_l1.item()'''
                
#decoder#grid
                loss_smooth = texture_smoothness_loss(tex_pred, valid_uv_mask)
                loss_sym = texture_symmetry_loss(tex_pred, valid_uv_mask)

                #loss_b = loss_l1 + 0.1*loss_smooth + 0.1*loss_sym
                #////////////////////////////一旦lossはl1だけでやってみる////////////////////////////////////////////////////////
                
                total_loss = total_loss + loss_l1
                
            # ===== バッチ全体の loss =====
#append
            total_loss = total_loss / B           # もともとの平均
            original_loss=total_loss
            total_loss = total_loss / ACCUM       # ★accumulation 分で割る

            total_loss.backward()                 # ★勾配を「足す」
            accum_counter+=1
# ACCUM 回たまったら更新
            if (batch_idx + 1) % ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_counter=0
            #///////////////////////schedulerは一旦無効///////////////////////////////////////////////////////////////
        #scheduler.step()
        #print(f"LR now: {scheduler.get_last_lr()}")
        #//////////////////////////////////////////////////////////////

        '''torch.cuda.reset_peak_memory_stats()
        print("allocated (real):", torch.cuda.max_memory_allocated() / 1024**3, "GB")'''
        #print("reserved (cached):", torch.cuda.max_memory_reserved() / 1024**3, "GB")
        if accum_counter > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        #//////////////////////////////////////////////////////////////
        print(f"[{epoch+1}] loss={original_loss.item():.4f}")
        #print(f"[{epoch+1}] loss={total_loss.item():.4f}")
        
        if(judge==True):
            loss_log.append(original_loss.item())
            judge=False
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
    from save_model import save_models
    save_models(encoder, tex_predictor)
    return loss_log
    

#debug_face_uv_mask_from_grid
def load_samples_from_folder(root_dir):
    """
    root_dir/
      └── 00001_1/
            ├── 00001_0003/
            │     ├── 00001_0003.png  ← image
            │     ├── mesh.obj        ← mesh
            │     ├── tex.png         ← uv
            │     └── tex.mtl         ← ignore
            ├── 00001_0004/
            ...
    """

    samples = []

    # 直下のフォルダ（例：00001_1）
    for group in sorted(os.listdir(root_dir)):
        group_dir = os.path.join(root_dir, group)
    
        if not os.path.isdir(group_dir):
            continue

        # その下のフォルダ（例：00001_0003）
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

                # 正面画像（例：00001_0003.png）
                if f_low.endswith(".png"):
                    # tex.png ではない
                    if f_low != "tex_down_1024.png":
                        if f_low != "tex_down_2048.png":
                            if f_low != "input_512.png":
                                if f_low != "tex_down_512.png":
                                    if f_low != "tex.png":
                                        if f_low != "cloth_mask.png":
                                            image_path = fpath
                                            continue
                if f_low=="bbox.txt":
                    bbox_path=fpath
            # 必須すべて揃っているか？
            if image_path and obj_path and uv_path and bbox_path and cloth_path:
                samples.append({
                    "image": image_path,
                    "obj": obj_path,
                    "uv": uv_path,
                    "bbox": bbox_path,
                    "cloth_mask": cloth_path
                })
            else:
                print(f"⚠ Skipped incomplete sample: {sample_dir}")
                print(f"   image={image_path}, obj={obj_path}, uv={uv_path}, bbox={bbox_path}")

    print(f"Loaded {len(samples)} samples from {root_dir}")
    return samples

#grid#2048
# =========================meshes[0]
# 7. メイン実行例
# =========================

if __name__ == "__main__":
    #Python ファイルが 「スクリプトとして直接実行されたときだけ」 中身を動かすためのおまじない。
    # 逆に、このファイルを他のファイルから import したときには、ここ以下は実行されない。
    # 「このファイルを main プログラムとして実行したときだけ、学習を始める」
    # という意味。
    torch.cuda.empty_cache()

    # 例として1サンプルだけ
    
    root_dir = "./THuman3.0"   # ←データフォルダのパスを書くだけ
    samples = load_samples_from_folder(root_dir)
#print#bbox
    #image → img_tensor (3,H,W)
    # obj → mesh (Meshes)
    # uv → tex_gt (1,3,Ht,Wt) か (3,Ht,Wt)
    dataset = SingleViewTextureDataset(samples)
    #samples の情報を元に __getitem__ で (img_tensor, mesh, tex_gt) を返す。
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,  # ← ここ変更
        persistent_workers=True,
        collate_fn=mesh_collate_fn     # ← ← ← これが無いと必ずエラー
    )
    '''print("num_workers:", dataloader.num_workers)
    print("batch size:", dataloader.batch_size)
    print("len dataset:", len(dataset))
    print("len dataloader:", len(dataloader))'''

    #image → img_tensor (3,H,W)
    # obj → mesh (Meshes)
    # uv → tex_gt (1,3,Ht,Wt) か (3,Ht,Wt)
    #tex_predictor
    #grid
    #mask
    #////////////////////////////for_NDC_test////////////////////////////////////////////
    '''img, meshes, tex_gt = next(iter(dataloader))
    mesh = meshes[0]  # Meshes オブジェクト

    # --- ★ UV 上下チェックテスト ★ ---
    uv_updown_test(mesh, tex_size=TEX_SIZE, filename="uv_updown_test_False_version.png")'''

    #////////////////////////////for_NDC_test////////////////////////////////////////////
    
    # モデル
    encoder = ImageEncoder().to(device)
    # enqcoder 出力チャンネル数（ResNet18 最終 conv は 512ch）
    
    tex_predictor = UVTexturePredictor(feat_channels=512).to(device)

    for name, p in tex_predictor.named_parameters():
        if 'decoder' in name and 'weight' in name:
            print("name_and_p: ",name, p.mean(), p.std())



    params = list(encoder.parameters()) + list(tex_predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    scheduler = None  # 必要なら StepLR 等を追加
    #エポックごとに学習率を下げたいなら、ここに StepLR, CosineAnnealingLR などを入れる。
    # 今は None なので学習率一定。
    #profile_one_step(dataloader, encoder, tex_predictor)
    #exit()


    #history=train_model(dataloader, encoder, tex_predictor, optimizer)

    #////////////////////////////以下は途中から学習開始(上のtrain消す)///////////////////////////////////////////////////////////////

    history=train_model(
    dataloader,
    encoder,
    tex_predictor,
    optimizer,
    resume_path="checkpoints/epoch_all_375_1.pth"
    )

    with open("loss_log.txt", "w") as f:
        for value in history:
            f.write(f"{value}\n")

#to(device)
#camera#pts_world
#train_model
#tex_gt
#tex_predictor
#valid_mask_3 loss_raw#nvidia-smi
#tex_pred
#with open(sample["bbox"], "r") as f:#compute_uv_map
#1024#align_corners#verts: #tex_gt_mean#load_checkpoint

#未使用   00028_30, 00026_0039, 00008_0068, 00008_0015, 00007_0021, 00006_0013, 00005_0048, 00001_0003\