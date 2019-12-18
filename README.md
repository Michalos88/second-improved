# Second-Improved
Implementation of SECOND paper for 3D Object Detection with following performance improvements:
1. Full Lyft Dataset Integration 
2. Parallel Data Preperation with Ray 
3. Checkpoints during training
4. Class upsampling, as in [CBGS paper](https://arxiv.org/abs/1908.09492)
5. Mean IOU Computation, just like in Lyft Kaggle Competition
6. Parallel Score Computation
7. Debugged config usage (some configs were not trully connected to anything)
8. Added PathLib Support
9. Added Scripts for Evaluation, Training and Data Prep
10. Handling of corrupted scenes in Lyft DataSet

This repo is based on [@traveller59's second.pytorch](https://github.com/traveller59/second.pytorch).

Using this code and configuration, I won 27th place in [2019 Lyft 3D Object Dectection Kaggle Competition](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/leaderboard). 

## Instalation

### Clone code

```bash
git clone https://github.com/traveller59/second.pytorch.git
cd ./second.pytorch/second
```

### CMake
```bash
wget "https://github.com/Kitware/CMake/releases/download/v3.15.4/cmake-3.15.4.tar.gz"
tar xf cmake-3.15.4.tar.gz
cd ./cmake-3.15.4/
./configure
make
make install
export PATH=/usr/local/bin:$PATH
cmake --version
```

### Pytorch
```bash
conda create -n env_stereo python=3.6
conda activate env_stereo
conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch
```

### spconv
```bash
git clone https://github.com/Michalos88/spconv --recursive
sudo apt-get install libboost-all-dev
cd spconv/
python setup.py bdist_wheel
cd ./dist/
pip install spconv-1.1-cp36-cp36m-linux_x86_64.whl
```

### Python Dependencies

```bash
conda install scikit-image scipy numba pillow matplotlib
```

```bash
pip install fire tensorboardX protobuf opencv-python ray 
```

If you want to use NuScenes dataset, you need to install [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit).
If you want to use Lyft dataset, you need to install [lyft-devkit](https://github.com/lyft/nuscenes-devkit).

### Setup cuda for numba 

you need to add following environment variable for numba.cuda, you can add them to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

### add second.pytorch/ to PYTHONPATH

```bash
export PATHONPATH=.
```

## Prepare dataset

* KITTI Dataset preparation

Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```

Then run
```bash
python create_data.py kitti_data_prep --data_path=KITTI_DATASET_ROOT
```

* [NuScenes](https://www.nuscenes.org) Dataset preparation

Download NuScenes dataset:
```plain
└── NUSCENES_TRAINVAL_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       └── v1.0-trainval <-- metadata and annotations
└── NUSCENES_TEST_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       └── v1.0-test     <-- metadata
```

Then run
```bash
python create_data.py nuscenes_data_prep --data_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --max_sweeps=10
python create_data.py nuscenes_data_prep --data_path=NUSCENES_TEST_DATASET_ROOT --version="v1.0-test" --max_sweeps=10
--dataset_name="NuscenesDataset"
```

* [Lyft](https://www.nuscenes.org) Dataset preparation

Download NuScenes dataset:
```plain
└── ../lyft_data/train
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       └── v1.0-trainval <-- metadata and annotations
└── ../lyft_data/test
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       └── v1.0-test     <-- metadata
```
Then run
```bash
python create_data.py lyft_data_prep --data_path=../lyft_data/train --version="v1.0-trainval" 
python create_data.py nuscenes_data_prep --data_path=../lyft_data/testT --version="v1.0-test" 
--dataset_name="LyftDataset"
```

## Usage

### train

I recommend to use script.py to train and eval. see script.py for more details.

#### train with single GPU

```bash
python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir
```

#### train with multiple GPU (need test, I only have one GPU)

Assume you have 4 GPUs and want to train with 3 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,3 python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir --multi_gpu=True
```

Note: The batch_size and num_workers in config file is per-GPU, if you use multi-gpu, they will be multiplied by number of GPUs. Don't modify them manually.

You need to modify total step in config file. For example, 50 epochs = 15500 steps for car.lite.config and single GPU, if you use 4 GPUs, you need to divide ```steps``` and ```steps_per_eval``` by 4.

#### train with fp16 (mixed precision)

Modify config file, set enable_mixed_precision to true.

* Make sure "/path/to/model_dir" doesn't exist if you want to train new model. A new directory will be created if the model_dir doesn't exist, otherwise will read checkpoints in it.

* training process use batchsize=6 as default for 1080Ti, you need to reduce batchsize if your GPU has less memory.

* Currently only support single GPU training, but train a model only needs 20 hours (165 epoch) in a single 1080Ti and only needs 50 epoch to reach 78.3 AP with super converge in car moderate 3D in Kitti validation dateset.

### evaluate

```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir --measure_time=True --batch_size=1
```

* detection result will saved as a result.pkl file in model_dir/eval_results/step_xxx or save as official KITTI label format if you use --pickle_result=False.

### pretrained model

You can download pretrained models in [google drive](https://drive.google.com/open?id=1YOpgRkBgmSAJwMknoXmitEArNitZz63C). The ```car_fhd``` model is corresponding to car.fhd.config.

Note that this pretrained model is trained before a bug of sparse convolution fixed, so the eval result may slightly worse.

## Docker (Deprecated. I can't push docker due to network problem.)

You can use a prebuilt docker for testing:
```
docker pull scrin/second-pytorch
```
Then run:
```
nvidia-docker run -it --rm -v /media/yy/960evo/datasets/:/root/data -v $HOME/pretrained_models:/root/model --ipc=host second-pytorch:latest
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=/root/model/car
```




