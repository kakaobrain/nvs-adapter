from typing import List
import io

from PIL import Image
import numpy as np


def decode_image(data, color: List, has_alpha: bool = True) -> np.array:
    img = Image.open(io.BytesIO(data))
    img = np.array(img, dtype=np.float32)
    if has_alpha:
        img[img[:, :, -1] == 0.0] = color
    return Image.fromarray(np.uint8(img[:, :, :3]))
