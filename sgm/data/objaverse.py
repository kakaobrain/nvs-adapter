import os
from typing import Dict, Union, Tuple
from functools import partial

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import webdataset as wds
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from sgm.data.utils import decode_image
from sgm.geometry import make_4x4_matrix, compute_inverse_transform, make_intrinsic_matrix


def create_dataset(urls, length, batch_size, resampled, stage, postprocess_fn, world_size, force_no_shuffle=False):
    
    shardshuffle = 100 if stage == "train" and not force_no_shuffle else None

    dloader_length = length // (batch_size * world_size)
    dataset_length = dloader_length * batch_size

    dataset = (
        wds.WebDataset(
            urls,
            nodesplitter=wds.split_by_node,
            shardshuffle=shardshuffle,
            detshuffle=True, # only for training
            resampled=resampled,
            handler=wds.ignore_and_continue,
        )
        .shuffle(
            size=(1 if shardshuffle is None else shardshuffle * 10),
            initial=(0 if shardshuffle is None else 100),
        )
        .map(postprocess_fn)
        .with_length(dataset_length)
    )
    
    return dataset


class ObjaverseDataLoader(LightningDataModule):

    color_background = [255.0, 255.0, 255.0, 255.0]

    def __init__(
        self,
        train_config: DictConfig,
        val_config: DictConfig,
        test_config: DictConfig | None,
        batch_size: int,
        num_workers: int,
        load_all_views: bool = False,
    ):
        super(ObjaverseDataLoader, self).__init__()
        if test_config is None:
            test_config = val_config

        self.train_config = train_config
        self.val_config = val_config
        self.test_config = test_config

        self.train_postprocess_fn = partial(self.postprocess_fn, config=train_config)
        self.val_postprocess_fn = partial(self.postprocess_fn, config=val_config)
        self.test_postprocess_fn = partial(self.postprocess_fn, config=test_config)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_all_views = load_all_views


    def setup(self, stage: str) -> None:        

        # Adjusting the length of an epoch for multi-node training
        world_size = 1
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            try:
                import torch.distributed

                if (
                        torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                ):
                    group = torch.distributed.group.WORLD
                    world_size = torch.distributed.get_world_size(group=group)
            except ModuleNotFoundError:
                pass

        if stage == "fit" or stage is None:
            self.train_dataset = create_dataset(self.train_config.urls, self.train_config.length, self.batch_size,
                                                True, "train", self.train_postprocess_fn, world_size)
            self.val_dataset = create_dataset(self.val_config.urls, self.val_config.length, self.batch_size,
                                              True, "val", self.val_postprocess_fn, world_size)
        
        if stage == "test" or stage is None:
            self.test_dataset = create_dataset(self.test_config.urls, self.test_config.length, self.batch_size,
                                               True, "val", self.test_postprocess_fn, world_size, force_no_shuffle=True)

    def postprocess_fn(self, sample: Dict, config: DictConfig) -> Union[Dict, None]:

        fov_rad = np.deg2rad(49.1)  # for objaverse rendering dataset

        num_views_each = [config.num_support_views, config.num_query_views]
        num_selected_views = sum(num_views_each)

        assert num_selected_views <= config.total_views, \
            f"num_selected_views={num_selected_views} > total_view={config.total_views}"

        if self.load_all_views:
            indices = np.concatenate([np.array([0], dtype=np.int32), np.arange(config.total_views-1, dtype=np.int32) + 1])
            num_views_each = (num_views_each[0], config.total_views)
        elif not config.deterministic:
            # training
            indices = np.random.choice(config.total_views, num_selected_views, replace=False)
        else:
            # evaluation
            indices = np.arange(num_selected_views)

        image_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize(
                (config.resolution, config.resolution), 
                interpolation=transforms.InterpolationMode.BICUBIC, 
                antialias=True
            ),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])

        rgbs, intrinsics, c2ws = [], [], []

        for index in indices:
            index_str = f"{index:03d}"
            png = sample[f"png_{index_str}"]
            npy = sample[f"npy_{index_str}"]
            image = image_transform(decode_image(png, self.color_background))

            w2c = wds.autodecode.npy_loads(npy).astype(np.float32)
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

        if config.use_relative:
            inverse_support_c2ws = torch.inverse(support_c2ws)
            support_c2ws = inverse_support_c2ws @ support_c2ws
            query_c2ws = inverse_support_c2ws @ query_c2ws

        return dict(
            support_rgbs=support_rgbs,
            support_intrinsics=support_intrinsics,
            support_c2ws=support_c2ws,
            query_rgbs=query_rgbs,
            query_intrinsics=query_intrinsics,
            query_c2ws=query_c2ws,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
