import os
from typing import List, Optional
import json

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from sgm.geometry import make_view_matrix, make_intrinsic_matrix


def decode_image(path: str, color: List, has_alpha: bool = True) -> np.array:
    img = Image.open(path)
    img = np.array(img, dtype=np.float32)
    if has_alpha:
        img[img[:, :, -1] == 0.0] = color
    return Image.fromarray(np.uint8(img[:, :, :3]))


class SingleImageDataset(Dataset):
    fov_rad = np.deg2rad(49.1)  # for objaverse rendering dataset
    color_background = [255.0, 255.0, 255.0, 255.0]

    def __init__(self, image_path: str,
                 support_elevation: float, support_azimuth: float, support_dist: float,
                 elevations: list, azimuths: list, dists: list,
                 resolution: int, num_query: int, use_relative: bool = True):
        super().__init__()

        assert len(elevations) == len(azimuths) == len(azimuths), \
            f"{len(elevations)=} == {len(azimuths)=} == {len(azimuths)=}"

        self.image_path = image_path
        self.num_query = num_query
        self.elevations = np.array(elevations).reshape([-1, self.num_query])
        self.azimuths = np.array(azimuths).reshape([-1, self.num_query])
        self.dists = np.array(dists).reshape([-1, self.num_query])
        self.resolution = resolution
        self.use_relative = use_relative

        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (self.resolution, self.resolution),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])
        self.support_rgb = image_transform(decode_image(self.image_path, self.color_background))

        self.support_c2w = make_view_matrix(azimuth=np.deg2rad(support_azimuth), elevation=np.deg2rad(support_elevation), dist=support_dist)
        self.support_c2w[:3, :3] *= -1

        self.intrinsic = make_intrinsic_matrix(fov_rad=self.fov_rad, h=self.resolution, w=self.resolution)

    def __len__(self):
        return len(self.elevations)

    def __getitem__(self, index):
        num_views_each = [1, self.num_query]

        intrinsics = [self.intrinsic]
        c2ws = [self.support_c2w]

        for azimuth, elevation, dist in zip(self.azimuths[index], self.elevations[index], self.dists[index]):
            c2w = make_view_matrix(azimuth=np.deg2rad(azimuth), elevation=np.deg2rad(elevation), dist=dist)
            c2w[:3, :3] *= -1

            intrinsics.append(self.intrinsic)
            c2ws.append(c2w)

        intrinsics, c2ws = map(lambda x: torch.stack(x), (intrinsics, c2ws))

        support_rgbs = self.support_rgb[None, ...]
        support_intrinsics, query_intrinsics = torch.split(intrinsics, num_views_each)
        support_c2ws, query_c2ws = torch.split(c2ws, num_views_each)

        if self.use_relative:
            inverse_support_c2ws = torch.inverse(support_c2ws)
            support_c2ws = inverse_support_c2ws @ support_c2ws
            query_c2ws = inverse_support_c2ws @ query_c2ws

        return dict(
            support_rgbs=support_rgbs,
            support_intrinsics=support_intrinsics,
            support_c2ws=support_c2ws,
            query_intrinsics=query_intrinsics,
            query_c2ws=query_c2ws,
        )
