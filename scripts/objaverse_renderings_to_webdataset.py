from typing import List
import os
import math
import json
from glob import glob
from tqdm import tqdm

import fire
import webdataset as wds


def paths_to_webdataset(paths: List[str],
                        basepath: str,
                        target_filepath_pattern: str,
                        maxcount_per_file: int = 100000,
                        maxsize_in_bytes_per_file: float = 1e9):
    
    shard_writer = wds.ShardWriter(pattern=target_filepath_pattern, 
                                   maxcount=maxcount_per_file,         # default 100,000
                                   maxsize=maxsize_in_bytes_per_file)  # default 3e9, in bytes    
    total_view = 12
    
    for path in tqdm(paths):
        basename = path
        sample = {
            "__key__": basename
        }
        for view_index in range(total_view):
            view_index_str = f"{view_index:03d}"
            
            full_basepath = os.path.join(basepath, basename, view_index_str)

            with open(f"{full_basepath}.png", "rb") as stream:
                image = stream.read()
            with open(f"{full_basepath}.npy", "rb") as stream:
                data = stream.read()
                
            sample[f"png_{view_index_str}"] = image
            sample[f"npy_{view_index_str}"] = data
        shard_writer.write(sample)
    shard_writer.close()

    
def objaverse_renderings_to_webdataset(paths_json_path: str,
                                       basepath: str = "./",
                                       maxcount_per_file: int = 100000,
                                       maxsize_in_bytes_per_file: float = 1e9):
    assert os.path.exists(paths_json_path), f"{paths_json_path} not exits."
    with open(os.path.join(paths_json_path)) as file:
        paths = json.load(file)

    total_objects = len(paths)
    assert total_objects > 0, f"total objects: {total_objects}, no valid objects exits."

    split_index = math.floor(total_objects * 0.99)

    train_paths = paths[:split_index]  # first 99 % as training
    valid_paths = paths[split_index:]  # last 1 % as validation

    assert len(train_paths) > 0, f"{len(train_paths)}, no train path exits."
    assert len(valid_paths) > 0, f"{len(valid_paths)}, no valid path exits."

    paths_to_webdataset(paths=train_paths,
                        basepath=basepath,
                        target_filepath_pattern="objaverse_rendering_train_%06d.tar",
                        maxcount_per_file=maxcount_per_file,
                        maxsize_in_bytes_per_file=maxsize_in_bytes_per_file)

    paths_to_webdataset(paths=valid_paths,
                        basepath=basepath,
                        target_filepath_pattern="objaverse_rendering_valid_%06d.tar",
                        maxcount_per_file=maxcount_per_file,
                        maxsize_in_bytes_per_file=maxsize_in_bytes_per_file)


if __name__ == "__main__":
    fire.Fire(objaverse_renderings_to_webdataset)
