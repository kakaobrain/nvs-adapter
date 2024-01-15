import argparse
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sgm.util import instantiate_from_config
from sgm.data.dirdataset import DirDataModule


def evaluate(args):
    pl.seed_everything(args.seed)

    name = args.name
    if name is None:
        name = "noname"

    expname = os.path.splitext(os.path.basename(args.config_path))[0]

    save_dir = Path(args.logdir, expname)
    save_dir.mkdir(exist_ok=True)

    with open(args.config_path) as fp:
        config = OmegaConf.load(fp)

    for cfg_path in args.additional_configs:
        with open(cfg_path) as fp:
            config = OmegaConf.merge(config, OmegaConf.load(fp))

    model_config = config.model
    model_config.params.use_ema = args.use_ema
    model_config.params.sd_ckpt_path = None
    model_config.params.ckpt_path = args.ckpt_path

    data_config = config.data

    if args.cfg_scale is not None:
        model_config.params.sampler_config.params.guider_config.params.scale = args.cfg_scale
        cfg_scale = args.cfg_scale
    else:
        cfg_scale = model_config.params.sampler_config.params.guider_config.params.scale

    dirname = f"{name}_step_{ckpt_step}_cfg_scale_{cfg_scale}_use_ema_{args.use_ema}_seed_{args.seed}"
    if args.split_idx is not None:
        dirname = dirname + "_" + f"{args.split_idx}"
    
    save_dir = save_dir.joinpath(dirname)
    save_dir.mkdir(exist_ok=True)

    litmodule = instantiate_from_config(model_config)
    litmodule.save_dir = save_dir

    datamodule = DirDataModule(
        ds_root_path=args.ds_root_path,
        ds_list_json_path=args.ds_list_json_path,
        num_total_views=args.num_total_views,
        batch_size=args.batch_size,
        num_workers=data_config.params.num_workers,
        resolution=data_config.params.val_config.resolution,
        use_relative=data_config.params.val_config.use_relative,
    )

    trainer = pl.Trainer(devices=1)
    trainer.test(litmodule, dataloaders=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None, required=True, help="path to config of trained model")
    parser.add_argument("--ckpt_path", type=str, default=None, required=True, help="path to checkpoint of trained model")
    parser.add_argument("-n", "--name", type=str, default=None, help="name of the visualization")
    parser.add_argument("--logdir", type=str, default="./logs_sampling", help="path to save the visualization")
    parser.add_argument("--use_ema", action="store_true", default=False, help="whether to use EMA model")
    parser.add_argument("--cfg_scale", type=float, default=None, help="scale for classifier free guidance")
    parser.add_argument("--ds_name", type=str, default="objaverse", help="the name of dataset")
    parser.add_argument("--ds_root_path", type=str, help="path to dataset for test", required=True)
    parser.add_argument("--ds_list_json_path", type=str, help="json path for list of dataset", required=True)
    parser.add_argument("--ds_num_total_views", type=int, help="number of total views per scene", required=True)
    parser.add_argument("--split_idx", type=int, default=None, help="split index for dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for test")
    parser.add_argument("--seed", type=int, default=0, help="seed for random number generator")
    parser.add_argument("-c", "--additional_configs", nargs="*", default=list())
    args = parser.parse_args()

    print('=' * 100)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 100)

    evaluate(args)
