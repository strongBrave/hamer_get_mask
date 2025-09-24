import os
import pickle
import numpy as np
from hamer.models import MANO
from hamer.utils.geometry import aa_to_rotmat
import torch
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils.renderer import Renderer, cam_crop_to_full
import cv2
import trimesh
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
"""
body_tex has no shape
z has no shape
hand_depth_min ()
grasp_quality has no shape
obj_texture has no shape
obj_scale has no shape
coords_3d (21, 3)
sample_id has no shape
verts_3d (778, 3)
affine_transform (4, 4)
hand_depth_max ()
obj_depth_min ()
coords_2d (21, 2)
obj_visibility_ratio ()
grasp_epsilon has no shape
side has no shape
depth_min ()
pose (156,)
hand_pose (45,)
bg_path has no shape
class_id has no shape
obj_depth_max ()
pca_pose (45,)
shape (10,)
depth_max ()
grasp_volume has no shape
sh_coeffs (9,)
obj_path has no shape
trans (3,)
"""

MODE = 'train' # ['train', 'test', 'val]  
obman_root = "/home/yujunhao/data/yujunhao/dataset/obman"
shapenet_root = "/home/yujunhao/data/yujunhao/projects/obman/datasymlinks/ShapeNetCore.v2"
obman_root = os.path.join(obman_root, MODE)
rgb_folder = os.path.join(obman_root, "rgb")
segm_folder = os.path.join(obman_root, "segm")
meta_folder = os.path.join(obman_root, "meta")
coord2d_folder = os.path.join(obman_root, "coords2d")
shapenet_template = os.path.join(
shapenet_root, "{}/{}/models/model_normalized.pkl"
)
prefix_template = "{:08d}"

cam_extr = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
    ])

# Instantiate MANO model
mano_cfg = {
    'data_dir': '_DATA/data/',
    'model_path': './_DATA/data/mano',
    'gender': 'neutral',
    'num_hand_joints': 15,
    'mean_params': './_DATA/data/mano_mean_params.npz',
    'create_body_pose': False
}
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

# load the obman dataset
def load_obman_sample(seleceted_idx: int = 0, debug: bool = False):
    sample = {}
    idxs = [
        int(imgname.split(".")[0])
        for imgname in sorted(os.listdir(meta_folder))
    ]

    prefixes = [prefix_template.format(idx) for idx in idxs]

    prefix = prefixes[seleceted_idx]

    meta_path = os.path.join(meta_folder, f"{prefix}.pkl")
    with open(meta_path, "rb") as f:
        meta_info = pickle.load(f)
    
    if debug:
        for key, val in meta_info.items():
            try:
                print(key, val.shape)
            except:
                print(f"{key} has no shape")
    sample['pca_pose'] = meta_info['pca_pose']
    sample['meta_path'] = meta_path
    
    return sample, idxs



if __name__ == "__main__":
    # 在 CPU 上构建 MANO，仅用于读取 PCA 组件
    mano = MANO(**mano_cfg)
    hand_components = mano.data_struct.hands_components  # numpy.ndarray, 形状 (45, 45)

    # 预先收集所有样本索引
    idxs = [
        int(imgname.split(".")[0])
        for imgname in sorted(os.listdir(meta_folder))
    ]

    # 通过 initializer 在子进程中共享 hand_components
    HAND_COMPONENTS = None

    def _init_worker(hand_components_arg):
        global HAND_COMPONENTS
        HAND_COMPONENTS = hand_components_arg

    def process_one(sample_idx: int) -> int:
        global HAND_COMPONENTS
        sample, _ = load_obman_sample(sample_idx, debug=False)
        hand_poses_pca = sample['pca_pose']  # (45,)
        # 纯 numpy 计算：pca(45,) @ components(45,45) -> (45,)
        hand_poses_full = np.einsum('i,ij->j', hand_poses_pca, HAND_COMPONENTS).reshape(-1,)
        meta_path = sample['meta_path']
        with open(meta_path, "rb") as f:
            meta_info = pickle.load(f)
        meta_info['hand_pose_from pca'] = hand_poses_full
        with open(meta_path, "wb") as f:
            pickle.dump(meta_info, f)
        return sample_idx

    # 多进程并行处理
    max_workers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(hand_components,)) as executor:
        futures = [executor.submit(process_one, i) for i in range(len(idxs))]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass