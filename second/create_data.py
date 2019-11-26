from pathlib import Path

import fire

from second.data.all_dataset import create_groundtruth_database,\
        create_groundtruth_database_parallel


def kitti_data_prep(root_path):
    import second.data.kitti_dataset as kitti_ds
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    create_groundtruth_database("KittiDataset",
                                root_path,
                                Path(root_path) / "kitti_infos_train.pkl")


def nuscenes_data_prep(root_path, version, dataset_name, max_sweeps=10):
    import second.data.nuscenes_dataset as nu_ds
    nu_ds.create_nuscenes_infos(root_path, version=version,
                                max_sweeps=max_sweeps)
    name = "infos_train.pkl"
    if version == "v1.0-test":
        name = "infos_test.pkl"
    create_groundtruth_database(dataset_name,
                                root_path, Path(root_path) / name)


def lyft_data_prep(root_path, version, dataset_name, max_sweeps=0,
                   names_upsample=False):

    import second.data.lyft_dataset as lyft_ds
    lyft_ds.create_lyft_infos(root_path, version=version,
                              max_sweeps=max_sweeps,
                              names_upsample=names_upsample)

    name = "infos_train.pkl"
    if version == "test":
        name = "infos_test.pkl"
    create_groundtruth_database_parallel(dataset_name,
                                         root_path, Path(root_path) / name)


if __name__ == '__main__':
    fire.Fire()
