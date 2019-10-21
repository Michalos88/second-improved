import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import ray

from second.core import box_np_ops
from second.data.dataset import get_dataset_class
from second.utils.progress_bar import progress_bar_iter as prog_bar


def create_groundtruth_database(dataset_class_name,
                                data_path,
                                info_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None,
                                relative_path=True,
                                add_rgb=False,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None):

    dataset = get_dataset_class(dataset_class_name)(
        info_path=info_path,
        root_path=data_path,
    )

    root_path = Path(data_path)

    if database_save_path is None:
        database_save_path = root_path / 'gt_database'
    else:
        database_save_path = Path(database_save_path)

    # There was kitti in the file name
    if db_info_save_path is None:
        db_info_save_path = root_path / "dbinfos_train.pkl"

    database_save_path.mkdir(parents=True, exist_ok=True)
    all_db_infos = {}

    group_counter = 0
    start = 0
    # if not (db_info_save_path/"db_checkpoints").as_posix
    for j in prog_bar(list(range(start, len(dataset)))):

        image_idx = j
        sensor_data = dataset.get_sensor_data(j)

        # Not the case with nuScenes or Lyft data
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]

        # Get GT data
        points = sensor_data["lidar"]["points"]
        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]

        # Genereate groups
        group_dict = {}
        group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        # Not in case of lyft or nuScenes
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            # set group ids to number of boxes
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)

        # Get Difficulty
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        # Not in case of lyft or nuScenes
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        # Get Count of Objects and indcies of points in clouds
        num_obj = gt_boxes.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)

        # For each object in a sample
        for i in range(num_obj):

            # # Save point cloud of each object seperately
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            filepath = database_save_path / filename
            gt_points = points[point_indices[:, i]]

            gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)

            # If class name is whitelisted
            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_path = str(database_save_path.stem + "/" + filename)
                else:
                    db_path = str(filepath)
                db_info = {
                    "name": names[i],
                    "path": db_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:

                # Count objects in a group
                # In case of lyft and nuScenes, each object is a group,
                # so the counter will be always 1
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1

                db_info["group_id"] = group_dict[local_group_id]

                # Not in case of lyft or nuScenes
                if "score" in annos:
                    db_info["score"] = annos["score"][i]

                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


def create_groundtruth_database_parallel(dataset_class_name,
                                         data_path,
                                         info_path=None,
                                         used_classes=None,
                                         database_save_path=None,
                                         db_info_save_path=None,
                                         relative_path=True,
                                         add_rgb=False,
                                         lidar_only=False,
                                         bev_only=False,
                                         coors_range=None):

    dataset = get_dataset_class(dataset_class_name)(
        info_path=info_path,
        root_path=data_path,
    )

    root_path = Path(data_path)

    if database_save_path is None:
        database_save_path = root_path / 'gt_database'
    else:
        database_save_path = Path(database_save_path)

    # There was kitti in the file name
    if db_info_save_path is None:
        db_info_save_path = root_path / "dbinfos_train.pkl"

    database_save_path.mkdir(parents=True, exist_ok=True)

    # if not (db_info_save_path/"db_checkpoints").as_posix

    all_db_infos = defaultdict(list)

    @ray.remote
    def get_all_objects(j):

        # group_counter = 0
        # if not (db_info_save_path/"db_checkpoints").as_posix

        image_idx = j
        sensor_data = dataset.get_sensor_data(j)

        # Not the case with nuScenes or Lyft data
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]

        # Get GT data
        points = sensor_data["lidar"]["points"]
        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]

        # # Genereate groups
        # group_dict = {}
        # group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        # # Not in case of lyft or nuScenes
        # if "group_ids" in annos:
        #     group_ids = annos["group_ids"]
        # else:
        #     # set group ids to number of boxes
        #     group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)

        # Get Difficulty
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        # Not in case of lyft or nuScenes
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        # Get Count of Objects and indcies of points in clouds
        num_obj = gt_boxes.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)

        object_instances = list()
        # For each object in a sample
        for i in range(num_obj):

            # # Save point cloud of each object seperately
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            filepath = database_save_path / filename
            gt_points = points[point_indices[:, i]]

            gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)

            # If class name is whitelisted
            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_path = str(database_save_path.stem + "/" + filename)
                else:
                    db_path = str(filepath)
                db_info = {
                    "name": names[i],
                    "path": db_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }
                # local_group_id = group_ids[i]
                # if local_group_id >= 0:

                # Not in case of lyft or nuScenes
                if "score" in annos:
                    db_info["score"] = annos["score"][i]

                object_instances.append(db_info)
        return object_instances

    ray.init()
    all_object_instanctes = [
            get_all_objects.remote(idx) for idx in range(len(dataset))]
    collected_objects = ray.get(all_object_instanctes)

    group_counter = 0
    for sub_list in collected_objects:
        for object_ in sub_list:
            object_['group_id'] = group_counter
            all_db_infos[object_['name']].append(object_)
            group_counter += 1
    # Not sure if it's a bug, but it seems like group_id is rather
    # # global_id of instance object read
    # if local_group_id not in group_dict:
    #     group_dict[local_group_id] = group_counter
    #     group_counter += 1
    #
    # db_info["group_id"] = group_dict[local_group_id]
#     all_db_infos = ray.get(all_db_infos_par)
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)
