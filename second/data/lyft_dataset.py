# import json
import pickle
# from copy import deepcopy
from pathlib import Path
# import subprocess

# import fire
import numpy as np

# from second.core import box_np_ops
# from second.core import preprocess as prep
# from second.data import kitti_common as kitti
from second.data.dataset import Dataset, register_dataset
# from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import progress_bar_iter as prog_bar


@register_dataset
class LyftDataset(Dataset):

    NumPointFeatures = 4  # xyz, timestamp. set 4 to use kitti pretrain

    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 prep_func=None,
                 num_point_features=None):

        self._root_path = Path(root_path)

        # Load samples
        with open(info_path, 'rb') as f:
            data = pickle.load(f)

        self._lyft_infos = data["infos"]

        self._lyft_infos = list(
            sorted(self._lyft_infos, key=lambda e: e["timestamp"]))
        self._metadata = data["metadata"]
        self._class_names = class_names
        self._prep_func = prep_func
        # kitti map: nusc det name -> kitti eval name
        self._kitti_name_mapping = {
            "car": "car",
            "pedestrian": "pedestrian",
        }  # we only eval these classes in kitti
        self.version = self._metadata["version"]
        self.eval_version = "cvpr_2019"
        self._with_velocity = False

    def __len__(self):
        return len(self._lyft_infos)


def create_lyft_infos(root_path, version="train", max_sweeps=10):

    # TODO: Reorganize folders to /lyft/train/images

    # Import and initializes an lyftenes SDK
    from lyft_dataset_sdk.lyftdataset import LyftDataset
    lyft = LyftDataset(data_path=root_path,
                       json_path=root_path+'data',
                       verbose=True)

    # Imports indexes of the splits
    from second.data import lyft_splits as splits
    available_vers = ["train", "test"]

    # Different train/val/test splits
    assert version in available_vers
    if version == "train":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "test":
        train_scenes = splits.test
        val_scenes = []
    else:
        raise ValueError("unknown version")

    test = "test" in version  # if no test in version this will be None
    root_path = Path(root_path)

    # Filter out not existing scenes. as you may have only downloaded mini
    available_scenes = _get_available_scenes(lyft)
    available_scene_names = [s["name"] for s in available_scenes]

    # Actual train/val/test split
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in val_scenes
    ])

    # Print sizes and meta data
    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(
            f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

    # Generating training infos and metadata
    train_lyft_infos, val_lyft_infos = _fill_trainval_infos(
        lyft, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = {
        "version": version,
    }

    # Save meta data and infos, depending if test or train/val
    if test:
        print(f"test sample: {len(train_lyft_infos)}")
        data = {
            "infos": train_lyft_infos,
            "metadata": metadata,
        }
        with open(root_path / "infos_test.pkl", 'wb') as f:
            pickle.dump(data, f)
    else:
        print(
            f"train sample: {len(train_lyft_infos)}\
                    , val sample: {len(val_lyft_infos)}"
        )

        data = {
            "infos": train_lyft_infos,
            "metadata": metadata,
        }

        # Save info
        with open(root_path / "infos_train.pkl", 'wb') as f:
            pickle.dump(data, f)
        data["infos"] = val_lyft_infos
        with open(root_path / "infos_val.pkl", 'wb') as f:
            pickle.dump(data, f)


def _get_available_scenes(lyft):
    available_scenes = []
    print("total scene num:", len(lyft.scene))
    for scene in lyft.scene:
        scene_token = scene["token"]
        scene_rec = lyft.get('scene', scene_token)
        sample_rec = lyft.get('sample', scene_rec['first_sample_token'])
        sd_rec = lyft.get('sample_data', sample_rec['data']["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = lyft.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            if not sd_rec['next'] == "":
                sd_rec = lyft.get('sample_data', sd_rec['next'])
            else:
                has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes


def _fill_trainval_infos(lyft,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """
        Generates train_val infos
    """
    train_lyft_infos = []
    val_lyft_infos = []
    from pyquaternion import Quaternion

    # For each sample
    for sample in prog_bar(lyft.sample):

        # TODO: Get Rid of - only for debuggging
        if sample["scene_token"] not in train_scenes or val_scenes:
            continue

        # Getting sample data for lidar and front camera
        lidar_token = sample["data"]["LIDAR_TOP"]
        cam_front_token = sample["data"]["CAM_FRONT"]

        sd_rec = lyft.get('sample_data', sample['data']["LIDAR_TOP"])

        cs_record = lyft.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])

        pose_record = lyft.get('ego_pose', sd_rec['ego_pose_token'])

        lidar_path, boxes, _ = lyft.get_sample_data(lidar_token)

        cam_path, _, cam_intrinsic = lyft.get_sample_data(cam_front_token)

        # Checks if lidar sweeps are available - not just-key-frames
        assert Path(lidar_path).exists(), ("Sweeps not found")

        # Info init
        info = {
            "lidar_path": lidar_path,
            "cam_front_path": cam_path,
            "token": sample["token"],
            "sweeps": [],
            "lidar2ego_translation": cs_record['translation'],
            "lidar2ego_rotation": cs_record['rotation'],
            "ego2global_translation": pose_record['translation'],
            "ego2global_rotation": pose_record['rotation'],
            "timestamp": sample["timestamp"],
        }

        # Sweep Generation
        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        sd_rec = lyft.get('sample_data', sample['data']["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == "":

                sd_rec = lyft.get('sample_data', sd_rec['prev'])
                cs_record = lyft.get('calibrated_sensor',
                                     sd_rec['calibrated_sensor_token'])
                pose_record = lyft.get('ego_pose', sd_rec['ego_pose_token'])

                lidar_path = lyft.get_sample_data_path(sd_rec['token'])

                sweep = {
                    "lidar_path": lidar_path,
                    "sample_data_token": sd_rec['token'],
                    "lidar2ego_translation": cs_record['translation'],
                    "lidar2ego_rotation": cs_record['rotation'],
                    "ego2global_translation": pose_record['translation'],
                    "ego2global_rotation": pose_record['rotation'],
                    "timestamp": sd_rec["timestamp"]
                }
                l2e_r_s = sweep["lidar2ego_rotation"]
                l2e_t_s = sweep["lidar2ego_translation"]
                e2g_r_s = sweep["ego2global_rotation"]
                e2g_t_s = sweep["ego2global_translation"]
                # sweep->ego->global->ego'->lidar
                l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

                R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
                sweep["sweep2lidar_rotation"] = R.T  # points @ R.T + T
                sweep["sweep2lidar_translation"] = T
                sweeps.append(sweep)
            else:
                break
        info["sweeps"] = sweeps

        # Collects Annotations
        if not test:
            annotations = [
                lyft.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)

            # Get velocity of a box
            velocity = np.array(
                [lyft.box_velocity(token)[:2] for token in sample['anns']])

            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            # convert names
            names = [b.name for b in boxes]
            # for i in range(len(names)):
            #     if names[i] in LyftDataset.NameMapping:
            #         names[i] = LyftDataset.NameMapping[names[i]]
            names = np.array(names)

            # we need to convert rot to SECOND format.
            # change the rot format will break all checkpoint, so...
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array(
                [a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array(
                [a["num_radar_pts"] for a in annotations])

        # Split samples based on scene token
        if sample["scene_token"] in train_scenes:
            train_lyft_infos.append(info)
        else:
            val_lyft_infos.append(info)

    return train_lyft_infos, val_lyft_infos
