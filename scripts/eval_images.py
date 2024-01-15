import os

import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob

import torch
from skimage.io import imread
from piqa.ssim import SSIM
from piqa.lpips import LPIPS
from piqa.psnr import PSNR


def compute_psnr_float(img_gt, img_pr):
    img_gt = img_gt.reshape([-1, 3]).astype(np.float32)
    img_pr = img_pr.reshape([-1, 3]).astype(np.float32)
    mse = np.mean((img_gt - img_pr) ** 2, 0)
    mse = np.mean(mse)
    psnr = 10 * np.log10(1 / mse)
    return psnr


def color_map_forward(rgb):
    dim = rgb.shape[-1]
    if dim==3:
        return rgb.astype(np.float32)/255
    else:
        rgb = rgb.astype(np.float32)/255
        rgb, alpha = rgb[:,:,:3], rgb[:,:,3:]
        rgb = rgb * alpha + (1-alpha)
        return rgb


def main():
    """
    input_dir
      - folder_0
        - pred.png
        - target.png
      - folder_1
        - pred.png
        - target.png
      ...
    """
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    args = parser.parse_args()

    output_log_path = os.path.join(args.input_dir, "metric.txt")

    target_path_list = glob(os.path.join(args.input_dir, "**", "target.png"), recursive=True)

    psnr_fn = PSNR().cuda()
    ssim_fn = SSIM().cuda()
    lpips_fn = LPIPS(network="vgg").cuda()

    psnrs, ssims, lpipss = [], [], []
    for target_path in tqdm(target_path_list):
        pred_path = target_path.replace("target.png", "pred.png")

        img_gt_int = imread(target_path)
        img_pr_int = imread(pred_path)

        img_gt = color_map_forward(img_gt_int)
        img_pr = color_map_forward(img_pr_int)

        with torch.no_grad():
            img_gt_tensor = torch.from_numpy(img_gt.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
            img_pr_tensor = torch.from_numpy(img_pr.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()

            ssims.append(ssim_fn(img_pr_tensor, img_gt_tensor).cpu().numpy())
            lpipss.append(lpips_fn(img_pr_tensor, img_gt_tensor).cpu().numpy())
            psnrs.append(psnr_fn(img_pr_tensor, img_gt_tensor).cpu().numpy())

    msg=f'psnr: {np.mean(psnrs):.5f}\nssim: {np.mean(ssims):.5f}\nlpips {np.mean(lpipss):.5f}'
    print(msg)
    with open(output_log_path,'w') as f:
        f.write(msg+'\n')


if __name__=="__main__":
    main()
