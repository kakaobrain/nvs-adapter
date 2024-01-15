import os
from typing import List, Optional
import json

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from PIL import Image

from sgm.geometry import make_4x4_matrix, compute_inverse_transform, make_intrinsic_matrix


def load_json_data(path: str):
    with open(path, "r") as file:
        json_data = json.load(file)
    return json_data


def decode_image(path: str, color: List, has_alpha: bool = True) -> np.array:
    img = Image.open(path)
    img = np.array(img, dtype=np.float32)
    if has_alpha:
        img[img[:, :, -1] == 0.0] = color
    return Image.fromarray(np.uint8(img[:, :, :3]))


class DirDataset(Dataset):

    color_background = [255.0, 255.0, 255.0, 255.0]

    def __init__(self, ds_root_path: str, ds_list_json_path: str, num_total_views: int, resolution: int, use_relative: bool):
        super().__init__()

        self.ds_root_path = ds_root_path
        self.ds_list_json_path = ds_list_json_path
        self.num_total_views = num_total_views

        ds_list = load_json_data(ds_list_json_path)
        self.ds_list = list(dict.fromkeys(["_".join(ds.split("_")[:-1]) for ds in ds_list]))
        self.resolution = resolution
        self.use_relative = use_relative

    def __len__(self):
        return len(self.ds_list)

    def __getitem__(self, index):
        ds_path = self.ds_list[index]

        path_list = []

        for idx in range(self.num_total_views):
            postfix = str(idx + 1).zfill(3)
            if idx == 0:
                path_list.append((
                    os.path.join(self.ds_root_path, ds_path + "_" + postfix, "source.png"),
                    os.path.join(self.ds_root_path, ds_path + "_" + postfix, "source.npy")
                ))

            path_list.append((
                os.path.join(self.ds_root_path, ds_path + "_" + postfix, "target.png"),
                os.path.join(self.ds_root_path, ds_path + "_" + postfix, "target.npy")
            ))

        fov_rad = np.deg2rad(49.1)  # for objaverse rendering dataset

        num_views_each = [1, self.num_total_views]

        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (self.resolution, self.resolution),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])

        rgbs, intrinsics, c2ws = [], [], []

        for png_path, npy_path in path_list:
            image = image_transform(decode_image(png_path, self.color_background))

            w2c = np.load(npy_path).astype(np.float32)
            w2c = make_4x4_matrix(torch.tensor(w2c))
            c2w = compute_inverse_transform(w2c)
            c2w[..., :3, :3] *= -1

            intrinsic = make_intrinsic_matrix(fov_rad=fov_rad, h=image.shape[1], w=image.shape[2])

            rgbs.append(image)
            intrinsics.append(intrinsic)
            c2ws.append(c2w)

        rgbs, intrinsics, c2ws = map(lambda x: torch.stack(x), (rgbs, intrinsics, c2ws))

        support_rgbs, query_rgbs = torch.split(rgbs, num_views_each)
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
            query_rgbs=query_rgbs,
            query_intrinsics=query_intrinsics,
            query_c2ws=query_c2ws
        )


class DirDataModule(LightningDataModule):

    def __init__(
        self,
        ds_root_path: str,
        ds_list_json_path: str,
        num_total_views: int,
        batch_size: int,
        num_workers: int,
        resolution: int,
        use_relative: bool,
    ):
        super(DirDataModule, self).__init__()

        self.ds_root_path = ds_root_path
        self.ds_list_json_path = ds_list_json_path
        self.num_total_views = num_total_views
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.use_relative = use_relative

        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: str) -> None:
        if stage == "test" or stage is None:
            self.test_dataset = DirDataset(self.ds_root_path, self.ds_list_json_path, self.num_total_views,
                                           resolution=self.resolution, use_relative=self.use_relative)
        else:
            raise f"DirDataLoader only support test"

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
