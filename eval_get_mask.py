import argparse
import os
import json
from pathlib import Path
import traceback
from typing import List, Optional

import pandas as pd
import torch
from filelock import FileLock
from hamer.configs import dataset_eval_to_get_mask_config
from hamer.datasets import create_webdataset
from hamer.utils import Evaluator, recursive_to
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT

from typing import Dict, Optional
import cv2

import torch
import numpy as np
import pytorch_lightning as pl
from yacs.config import CfgNode

import webdataset as wds
from hamer.configs import to_lower
from hamer.datasets.dataset import Dataset
from hamer.datasets.image_dataset import ImageDataset
from hamer.datasets.mocap_dataset import MoCapDataset
from hamer.utils.renderer import Renderer, cam_crop_to_full
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
DEFAULT_STD = np.array([0.229, 0.224, 0.225])

def preprocess(batch, out, model_cfg):
    B = batch['img'].shape[0]
    multiplier = (2*batch['right']-1)
    pred_cam = out['pred_cam']
    pred_cam[:,1] = multiplier*pred_cam[:,1]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = torch.tensor([256, 256], device=pred_cam.device, dtype=pred_cam.dtype).unsqueeze(0).repeat(B, 1)
    multiplier = (2*batch['right']-1)
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()    
    return pred_cam_t_full, scaled_focal_length

def render_mask(renderer, scaled_focal_length, vertices, cam_t, img_size, is_right):
    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
    )    
    cam_view = renderer.render_rgba(vertices, cam_t=cam_t, render_res=img_size, is_right=is_right, **misc_args)
    alpha_channel = cam_view[..., 3]
    
    # 根据阈值创建二值 mask（透明区域为 0，非透明区域为 1）
    mask = (alpha_channel > 0.5).astype(np.uint8) * 255  # 大于阈值为 1，否则为 0

    cv2.imwrite(f'mask.png', mask)

    return mask
class MixedWebDataset(wds.WebDataset):
    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode, train: bool = False) -> None:
        super(wds.WebDataset, self).__init__()
        datasets = [create_webdataset(cfg, dataset_cfg, train=train)]
        weights = np.array([1.0])
        weights = weights / weights.sum()  # normalize
        self.append(wds.RandomMix(datasets, weights))

def tensor_to_image(tensor):
    image = tensor * DEFAULT_STD[:,None,None] + DEFAULT_MEAN[:,None,None]
    image = image.transpose(1,2,0) * 255
    image = image.astype(np.uint8)
    return image

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--results_folder', type=str, default='results', help='Path to results folder.')
    parser.add_argument('--dataset', type=str, default='FREIHAND-TRAIN,INTERHAND26M-TRAIN,HALPE-TRAIN,COCOW-TRAIN,MTC-TRAIN,RHD-TRAIN,MPIINZSL-TRAIN,HO3D-TRAIN,H2O3D-TRAIN,DEX-TRAIN', help='Dataset to evaluate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of test samples to draw')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers used for data loading')
    parser.add_argument('--log_freq', type=int, default=10, help='How often to log results')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')

    args = parser.parse_args()
    # Download and load checkpoints
    # download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)    

    # Load config and run eval, one dataset at a time
    print('Evaluating on datasets: {}'.format(args.dataset), flush=True)
    for dataset in args.dataset.split(','):
        dataset_cfg = dataset_eval_to_get_mask_config()[dataset]
        args.dataset = dataset
        run_eval(model, model_cfg, dataset_cfg, device, args, renderer)

def run_eval(model, model_cfg, dataset_cfg, device, args, renderer):

    # # List of metrics to log
    # if args.dataset in ['FREIHAND-VAL', 'HO3D-VAL']:
    #     metrics = None
    #     preds = ['vertices', 'keypoints_3d']
    #     pck_thresholds = None
    #     rescale_factor = -1
    # elif args.dataset in ['NEWDAYS-TEST-ALL', 'NEWDAYS-TEST-VIS', 'NEWDAYS-TEST-OCC',
    #                       'EPICK-TEST-ALL', 'EPICK-TEST-VIS', 'EPICK-TEST-OCC',
    #                       'EGO4D-TEST-ALL', 'EGO4D-TEST-VIS', 'EGO4D-TEST-OCC']:
    #     metrics = ['mode_kpl2']
    #     preds = None
    #     pck_thresholds = [0.05, 0.1, 0.15]
    #     rescale_factor = 2

    # Create dataset and data loader
    dataset = MixedWebDataset(model_cfg, dataset_cfg, train=False)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    # Setup evaluator object
    # evaluator = Evaluator(
    #     dataset_length=dataset.__len__(),
    #     dataset=args.dataset,
    #     keypoint_list=dataset_cfg.KEYPOINT_LIST, 
    #     pelvis_ind=model_cfg.EXTRA.PELVIS_IND, 
    #     metrics=metrics,
    #     preds=preds,
    #     pck_thresholds=pck_thresholds,
    # )
    save_dir = "frei-hand"
    os.makedirs(save_dir, exist_ok=True)

    # Go over the images in the dataset.
    for i, batch in enumerate(tqdm(dataloader)):
        batch = recursive_to(batch, device)
      
        with torch.no_grad():
            out = model(batch)
        # evaluator(out, batch)
    #     if i % args.log_freq == args.log_freq - 1:
    #         evaluator.log()
    # evaluator.log()
    # error = None
            batch_size = batch['img'].shape[0]
            pred_cam_t_full, scaled_focal_length = preprocess(batch, out, model_cfg)
            for n in range(batch_size):
                import ipdb; ipdb.set_trace()
                img_name = batch['imgname'][n]
                img = batch['img'][n].clone().detach().cpu().numpy()
                img = tensor_to_image(img)
                img = img[:, :, ::-1]
                save_path = os.path.join(save_dir, f'test_{n}.png')
                cv2.imwrite(f"img.png", img)

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )
                cv2.imwrite(f'regression_img.png', 255*regression_img[:, :, ::-1])                
                
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                img_size = np.array([256, 256])

                render_mask(renderer, scaled_focal_length, verts, cam_t, img_size, is_right)


if __name__ == '__main__':
    main()
