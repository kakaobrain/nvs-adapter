from typing import Union, Tuple

import numpy as np
import torch


def make_4x4_matrix(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    from 3x4 matrix to 4x4 matrix by appending [0, 0, 0, 1] to the bottom row
    """
    if isinstance(matrix, torch.Tensor):
        output = torch.zeros([4, 4], dtype=matrix.dtype)
    elif isinstance(matrix, np.ndarray):
        output = np.zeros([4, 4], dtype=matrix.dtype)
    else:
        raise TypeError(f"{type(matrix)} is unsupported")

    output[:3,:4] = matrix[:3,:4]
    output[3, 3] = 1.0
    return output


def make_intrinsic_matrix(fov_rad: float, h: int, w: int) -> torch.Tensor:
    """
    make intrinsic matrix from fov, height, width
    """
    focal_x = w / (2 * np.tan(fov_rad))
    focal_y = h / (2 * np.tan(fov_rad))

    intrinsic = torch.tensor([[focal_x, 0.0, w * 0.5, 0.0],
                               [0.0, focal_y, h * 0.5, 0.0],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]],
                             dtype=torch.float32)
    return intrinsic


def normalize(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    vec /= (vec.norm(dim=-1) + eps)
    return vec


def lookat(eye: torch.Tensor,
           center: torch.Tensor,
           up: torch.Tensor) -> torch.Tensor:
    forward = center - eye
    forward = normalize(forward)

    right = torch.cross(up, forward)
    right = normalize(right)

    up = torch.cross(right, forward)
    up = normalize(up)

    matrix = torch.tensor([[right[0], up[0], -forward[0], eye[0]],
                           [right[1], up[1], -forward[1], eye[1]],
                           [right[2], up[2], -forward[2], eye[2]],
                           [0.0, 0.0, 0.0, 1.0]], dtype=eye.dtype)
    matrix[:3, :3] *= -1
    return matrix


def spherical_to_cartesian(azimuth: float, elevation: float, radius: float, dtype=torch.float32) -> torch.Tensor:
    if isinstance(azimuth, torch.Tensor):
        return torch.tensor([radius * torch.cos(elevation) * torch.cos(azimuth),
                             radius * torch.cos(elevation) * torch.sin(azimuth),
                             radius * torch.sin(elevation)],
                            dtype=dtype)
    else:
        return torch.tensor([radius * np.cos(elevation) * np.cos(azimuth),
                             radius * np.cos(elevation) * np.sin(azimuth),
                             radius * np.sin(elevation)],
                            dtype=dtype)


def make_view_matrix(azimuth, elevation, dist, dtype=torch.float32) -> torch.Tensor:
    """
    make view matrix from elevation, azimuth, distance
    """
    position = spherical_to_cartesian(azimuth, elevation, dist, dtype=dtype)
    center = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    up = torch.tensor([0.0, 0.0, -1.0], dtype=dtype)
    matrix = lookat(eye=position, center=center, up=up)
    return matrix


def inverse_intrinsic_matrix(intrinsic: torch.Tensor) -> torch.Tensor:
    """
    efficient computation of inverse intrinsic matrix
    """
    inv_intrinsics = torch.zeros_like(intrinsic)
    inv_intrinsics[..., [0, 1], [0, 1]] = 1.0 / intrinsic[..., [0, 1], [0, 1]]
    inv_intrinsics[..., [0, 1], [2]] = - intrinsic[..., [0, 1], [2]] * inv_intrinsics[..., [0, 1], [0, 1]]
    inv_intrinsics[..., [2, 3], [2, 3]] = 1.0
    return inv_intrinsics


def compute_inverse_transform(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    efficient computation of inverse transform matrix
    """
    if isinstance(matrix, torch.Tensor):
        output = torch.zeros([4, 4], dtype=matrix.dtype)
    elif isinstance(matrix, np.ndarray):
        output = np.zeros([4, 4], dtype=matrix.dtype)
    else:
        raise TypeError(f"{type(matrix)} is unsupported")

    output[..., :3, :3] = matrix[..., :3, :3].T
    output[..., :3, 3:] = - output[..., :3, :3] @ matrix[..., :3, 3:]
    output[..., 3, 3] = 1.0
    return output


def get_rays(intrinsics: torch.Tensor,
             c2ws: torch.Tensor,
             image_size: Tuple[int, int],
             stride: int = 1
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    from intrinsics, c2ws, image_size, stride, compute rays, 
    get rays origin and rays direction
    """
    batch_size = c2ws.shape[0]
    intrinsics_stride = intrinsics.clone()
    intrinsics_stride[..., [0, 0, 1, 1], [0, 2, 1, 2]] /= stride
    inv_intrinsics = inverse_intrinsic_matrix(intrinsics_stride)[..., :3, :3]
    h, w = image_size
    h, w = int(h // stride), int(w // stride)

    w_arange = torch.arange(w).type_as(intrinsics)
    h_arange = torch.arange(h).type_as(intrinsics)

    mesh_grid = (torch.stack(torch.meshgrid(w_arange, h_arange, indexing="xy"), -1) + 0.5).type_as(intrinsics)
    mesh_grid = torch.repeat_interleave(mesh_grid.unsqueeze(0), batch_size, dim=0)

    i_coords, j_coords = torch.split(mesh_grid, [1, 1], dim=-1)
    pixel_homo_coords = torch.cat([i_coords, j_coords, torch.ones_like(i_coords)], dim=-1)

    dirs = torch.einsum("bnij, bhwj -> bnhwi", inv_intrinsics, pixel_homo_coords)
    rays_offset = c2ws[:, :, None, None, :3, 3].repeat((1, 1, h, w, 1))
    rays_direction = torch.einsum("bnhwi, bnji -> bnhwj", dirs, c2ws[..., :3, :3])

    return rays_offset, rays_direction
