import torch
import config
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
)


def compute_uv_to_3d_points(mesh_cpu: Meshes, cameras, tex_size=config.TEX_SIZE):
    """
    Compute the 3D world coordinate corresponding to each UV texel.

    Args:
        mesh_cpu: PyTorch3D mesh object.
        cameras: Camera used for rasterizing the UV mesh.
        tex_size: Resolution of the UV texture map.

    Returns:
        pts_world:  (1, Ht, Wt, 3), 3D point for each UV texel
        valid_mask: (1, Ht, Wt), whether each texel lies on the mesh
        normals:    (1, Ht, Wt, 3), interpolated surface normals
    """

    # 3D mesh vertices and face indices
    mesh = mesh_cpu.to(config.DEVICE)
    verts = mesh.verts_padded()   # (1, V, 3)
    faces = mesh.faces_padded()   # (1, F, 3)


    # UV verts / faces
    verts_uvs = mesh.textures.verts_uvs_padded()[0].to(config.DEVICE)  # (Vt,2)
    faces_uvs = mesh.textures.faces_uvs_padded()[0].to(config.DEVICE)  # (F,3)
    

    # UV → NDC
    uv = verts_uvs.clone()
    uv[:, 0] = uv[:, 0] * 2.0 - 1.0
    uv[:, 1] = uv[:, 1] * 2.0 - 1.0

    uv_verts_3d = torch.cat(
        [uv, torch.zeros_like(uv[:, :1])],
        dim=-1
    )  # (Vt,3)

    # Build a mesh in UV space
    uv_mesh = Meshes(
        verts=[uv_verts_3d],
        faces=[faces_uvs],
    )


    raster_settings = RasterizationSettings(
        image_size=tex_size,
        faces_per_pixel=1,
        blur_radius=0.0,
    )

    # Rasterize the UV mesh
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings,
    )
    
    fragments = rasterizer(uv_mesh)
    
    # For each UV pixel:
    #   pix_to_face gives the face index covering that pixel
    #   bary_coords gives the barycentric coordinates within that face
    pix_to_face = fragments.pix_to_face[0, ..., 0]
    bary_coords = fragments.bary_coords[0, ..., 0, :]
    H_tex, W_tex = pix_to_face.shape

    # Valid UV texels
    valid_mask = pix_to_face >= 0   # (H,W)
    num_valid = int(valid_mask.sum())

    # If no valid texels exist, return dummy outputs
    if num_valid == 0:
        pts_world_dummy = torch.zeros((1, H_tex, W_tex, 3), device=config.DEVICE)
        return pts_world_dummy, valid_mask.unsqueeze(0)
        
    # Replace invalid face indices with 0
    safe_face_idx = pix_to_face.clone()
    safe_face_idx[~valid_mask] = 0

    # Get the 3D vertex indices of the corresponding face for each UV texel
    faces_idx = faces[0][safe_face_idx]   

    # Gather the 3D coordinates of the three vertices of each face
    face_verts = verts[0][faces_idx]   
    
    verts_normals = mesh.verts_normals_padded()[0]  
    
    # vertex normals and barycentric coordinates
    face_norms = verts_normals[faces_idx]
    normals = torch.sum(
        bary_coords.unsqueeze(-1) * face_norms,
        dim=-2
    )  # (Ht,Wt,3)
    
    # Interpolate the 3D world coordinate of each UV texel
    pts_world = torch.sum(
        bary_coords.unsqueeze(-1) * face_verts,   # (H,W,3,1) * (H,W,3,3)
        dim=-2
    )  # (H,W,3)

    pts_world = pts_world.unsqueeze(0)
    valid_mask = valid_mask.unsqueeze(0)  
    normals = normals.unsqueeze(0)    

    return pts_world, valid_mask, normals