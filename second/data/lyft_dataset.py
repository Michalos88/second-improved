import json
import pickle
# from copy import deepcopy
from pathlib import Path
# import subprocess

import fire
import numpy as np

# from second.core import box_np_ops
# from second.core import preprocess as prep
# from second.data import kitti_common as kitti
from second.data.dataset import Dataset, register_dataset
# from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import progress_bar_iter as prog_bar
from second.data import lyft_splits as splits


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

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    def get_sensor_data(self, query):
        idx = query  # iloc of sample
        read_test_image = False

        # Not sure how is this useful
        if isinstance(query, dict):
            assert "lidar" in query
            idx = query["lidar"]["idx"]
            read_test_image = "cam" in query

        # Get sample
        info = self._lyft_infos[idx]

        # Init result - output
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "token": info["token"]
            },
        }

        lidar_path = Path(info['lidar_path'])

        # -----> if there no sweeps, we have at least one point cloud
        # Get main point cloud
        try:
            points = np.fromfile(
                str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        except ValueError as e:
            print(info['lidar_path'])
            raise e
        # Normalize points
        points[:, 3] /= 255
        points[:, 4] = 0
        sweep_points_list = [points]
        # Format time stamp
        ts = info["timestamp"] / 1e6
        # <------

        # Go through n sweeps, if n=0 this will be skipped
        for sweep in info["sweeps"]:
            # Get point clouds starting with main
            try:
                points_sweep = np.fromfile(
                    str(sweep["lidar_path"]), dtype=np.float32,
                    count=-1).reshape([-1, 5])
            except ValueError as e:
                print("sweep", sweep[lidar_path])
                print("main", info[lidar_path])
                raise e
            # Normalize points
            sweep_ts = sweep["timestamp"] / 1e6
            points_sweep[:, 3] /= 255

            # Format rotation and translation
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                "sweep2lidar_rotation"].T
            points_sweep[:, :3] += sweep["sweep2lidar_translation"]
            points_sweep[:, 4] = ts - sweep_ts
            sweep_points_list.append(points_sweep)

        points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]

        # If would like to use images, default False
        if read_test_image:
            if Path(info["cam_front_path"]).exists():
                with open(str(info["cam_front_path"]), 'rb') as f:
                    image_str = f.read()
            else:
                image_str = None
            res["cam"] = {
                "type": "camera",
                "data": image_str,
                "datatype": Path(info["cam_front_path"]).suffix[1:],
            }

        # Add point cloud to output
        res["lidar"]["points"] = points

        # Get ground truth boxes
        if 'gt_boxes' in info:
            # Filters out all annotations, without point clouds
            # However due to the fact that this field is not
            # provided by Lyft, we will accepts all annoations
            # TODO: Impute this field
            mask = info["num_lidar_pts"] == -1
            gt_boxes = info["gt_boxes"][mask]

            # Default is False
            if self._with_velocity:
                gt_velocity = info["gt_velocity"][mask]
                nan_mask = np.isnan(gt_velocity[:, 0])
                gt_velocity[nan_mask] = [0.0, 0.0]
                gt_boxes = np.concatenate([gt_boxes, gt_velocity], axis=-1)
            res["lidar"]["annotations"] = {
                'boxes': gt_boxes,
                'names': info["gt_names"][mask],
            }

        return res

    def evaluation(self, detections, output_dir):
        """kitti evaluation is very slow, remove it.
        """
        # res_kitti = self.evaluation_kitti(detections, output_dir)
        # Select evalutation prodcedure
        res = self.evaluation_kaggle(detections, output_dir)
        return res

    def evaluation_kaggle(self, detections, output_dir):

        # Getting ground truth
        gt_annos = self.ground_truth_annotations
        if gt_annos is None:
            return None
        lyft_annos = list()

        # Class name mapping
        mapped_class_names = self._class_names

        token2info = {}
        for info in self._lyft_infos:
            token2info[info["token"]] = info

        for det in detections:
            annos = []
            boxes = _second_det_to_nusc_box(det)

            boxes = _lidar_lyft_box_to_global(
                token2info[det["metadata"]["token"]], boxes,
                mapped_class_names, "lyft_cvpr_2019")

            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                lyft_anno = {
                    "sample_token": det["metadata"]["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "name": name,
                    "score": box.score,
                }
                annos.append(lyft_anno)

            lyft_annos.extend(annos)

        # TODO: Convert to pandas then csv
        res_path = Path(output_dir) / "results_lyft.pkl"

        # Save processed data
        with open(res_path, "wb") as f:
            pickle.dump(lyft_annos, f)

        # Evaluate score
        from second.data.lyft_eval import eval_main
        mAPs = list()
        for threshold in range(0.5, 1.0, 0.05):
            mAPs.append(eval_main(gt_annos, lyft_annos, threshold))

        print("Final Score = ", np.mean(mAPs))
        return None


@register_dataset
class LyftDatasetD2(LyftDataset):
    """Reducing DataSet by factor D"""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._lyft_infos = list(
            sorted(self._lyft_infos, key=lambda e: e["timestamp"]))
        self._lyft_infos = self._lyft_infos[::2]


@register_dataset
class LyftDatasetD8(LyftDataset):
    """Reducing DataSet by factor D"""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._lyft_infos = list(
            sorted(self._lyft_infos, key=lambda e: e["timestamp"]))
        self._lyft_infos = self._lyft_infos[::8]


def create_lyft_infos(root_path, version="train", max_sweeps=10):

    # TODO: Reorganize folders to /lyft/train/images

    # Import and initializes an lyftenes SDK
    from lyft_dataset_sdk.lyftdataset import LyftDataset
    lyft = LyftDataset(data_path=root_path,
                       json_path=root_path+'data',
                       verbose=True)

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

    # Check filter out scene names that are not in available scenes
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))

    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))

    # Get scene tokens for filtered scene names
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

        # Check if sample comes from either train or val sets
        if sample["scene_token"] not in train_scenes and\
                sample["scene_token"] not in val_scenes and\
                sample['token'] not in splits.blk_listed:
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

                # To make sure that broken sample does not end up in sweeps
                if sd_rec['sample_token'] in splits.blk_listed:
                    print('Sample Skipped: ', sd_rec['sample_token'])
                    continue

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
        elif sample["scene_token"] in val_scenes:
            val_lyft_infos.append(info)

    return train_lyft_infos, val_lyft_infos


def get_all_box_mean(info_path):
    det_names = ['car',
                 'truck',
                 'other_vehicle',
                 'bus',
                 'pedestrian',
                 'bicycle',
                 'motorcycle',
                 'emergency_vehicle',
                 'animal']
    det_names = sorted(det_names)
    res = {}
    details = {}
    for k in det_names:
        result = get_box_mean(info_path, k)
        details[k] = result["detail"]
        res[k] = result["box3d"]
    print(json.dumps(res, indent=2))
    return details


def get_box_mean(info_path, class_name="car",
                 eval_version="lyft_cvpr_2019"):
    with open(info_path, 'rb') as f:
        lyft_infos = pickle.load(f)["infos"]
    from second.configs.lyft.eval import eval_detection_configs
    cls_range_map = eval_detection_configs[eval_version]["class_range"]

    gt_boxes_list = []
    gt_vels_list = []
    for info in lyft_infos:
        gt_boxes = info["gt_boxes"]
        gt_vels = info["gt_velocity"]
        gt_names = info["gt_names"]
        mask = np.array([s == class_name for s in info["gt_names"]],
                        dtype=np.bool_)
        gt_names = gt_names[mask]
        gt_boxes = gt_boxes[mask]
        gt_vels = gt_vels[mask]
        det_range = np.array([cls_range_map[n] for n in gt_names])
        det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
        mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
        mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)

        gt_boxes_list.append(gt_boxes[mask].reshape(-1, 7))
        gt_vels_list.append(gt_vels[mask].reshape(-1, 2))
    gt_boxes_list = np.concatenate(gt_boxes_list, axis=0)
    gt_vels_list = np.concatenate(gt_vels_list, axis=0)
    nan_mask = np.isnan(gt_vels_list[:, 0])
    gt_vels_list = gt_vels_list[~nan_mask]

    # return gt_vels_list.mean(0).tolist()
    return {
        "box3d": gt_boxes_list.mean(0).tolist(),
        "detail": gt_boxes_list
        # "velocity": gt_vels_list.mean(0).tolist(),
    }


def _second_det_to_nusc_box(detection):
    from lyft_dataset_sdk.utils.data_classes import Box
    import pyquaternion
    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()
    box3d[:, 6] = -box3d[:, 6] - np.pi / 2
    box_list = []
    for i in range(box3d.shape[0]):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box3d[i, 6])
        velocity = (np.nan, np.nan, np.nan)
        if box3d.shape[1] == 9:
            velocity = (*box3d[i, 7:9], 0.0)
            # velo_val = np.linalg.norm(box3d[i, 7:9])
            # velo_ori = box3d[i, 6]
            # velocity = (velo_val * np.cos(velo_ori),
            #             velo_val * np.sin(velo_ori), 0.0)
        box = Box(
            box3d[i, :3],
            box3d[i, 3:6],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def _lidar_lyft_box_to_global(info,
                              boxes,
                              classes,
                              eval_version="lyft_cvpr_2019"):
    import pyquaternion
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        from second.configs.lyft.eval import eval_detection_configs
        # filter det in ego.
        cls_range_map = eval_detection_configs[eval_version]["class_range"]
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list


if __name__ == "__main__":
    fire.Fire()
