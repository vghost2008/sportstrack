#sportstrack

This repository is an official implementation of the [SportsTrack: An Innovative Method for Tracking Athletes in Sports Scenes](https://arxiv.org/abs/2211.07173)

1st place of 2022 ECCV Sports MOT challenge.

## Result On SportsMOT test

|Method|HOTA|AssA|DetA|MOTA|weight|
|---|---|---|---|---|---|
|SportsTrack|76.264|73.538|79.180|89.316|[baidu](https://pan.baidu.com/s/1_LP0F-EblkiZ8olI4iv1Pw?pwd=nvq4)|

## Quick Start

- Download SportsMOT dataset from [here](https://codalab.lisn.upsaclay.fr/competitions/4433)

- set data root path of sports MOT dataset

```
export SPORTSTRACK_ROOT=...
```

```
ls  ${SPORTSTRACK_ROOT}/data/sportsmot_publish/dataset/test
```

output
```
v_1UDUODIBSsc_c001  v_2ChiYdg5bxI_c058  v_2Dw9QNH5KtU_c014  v_7FTsO8S3h88_c007  v_9p0i81kAEwE_c010  v_BdD9xu0E2H4_c011  v_czYZnO9QxYQ_c020
...
```

- build gmc

build gmc need opencv4

```
cd gmc
mkdir build
cd build
cmake ..
make
./gmc ${SPORTSTRACK_ROOT}/data/sportsmot_publish/dataset/test/*   #this will generate a gmc.txt file in each input directorys
./gmc ${SPORTSTRACK_ROOT}/data/sportsmot_publish/dataset/val/* 
```
- run algorithm 

```
python main.py --split val --gpus 0    #track on val dataset 
```

or 

```
python main.py --split test --gpus 0  #track on test dataset
```

or use multi process

```
python multi_process_main.py --split test --gpus 0   #use multiprocess to track on test dataset 
```

- post process

```
python postprocess_tools/merge_tracks.py --image_dir ... --input_txt_dir ... --output_txt_dir ...
python postprocess_tools/interpolation.py --input_txt_dir ... --output_txt_dir ...
```

Example:

```
 python postprocess_tools/merge_tracks.py --image_dir ${SPORTSTRACK_ROOT}/data/sportsmot_publish/dataset/test --input_txt_dir ${SPORTSTRACK_ROOT}/tmp/sportsmot-test/PDSMV2SportsTrackerT1/data/ --output_txt_dir ${SPORTSTRACK_ROOT}/tmp/sportsmot-test/PDSMV2SportsTrackerT1/data_merge
 python postprocess_tools/interpolation.py --input_txt_dir ${SPORTSTRACK_ROOT}/tmp/sportsmot-test/PDSMV2SportsTrackerT1/data_merge/ --output_txt_dir ${SPORTSTRACK_ROOT}/tmp/sportsmot-test/PDSMV2SportsTrackerT1/data_inter
```

## requirements

- pytorch
- [wml2](https://github.com/vghost2008/wml2)
 
install wml2

```
git clone git@github.com:vghost2008/wml2.git
cd wml2
export PYTHONPATH=${PYTHONPATH}:`pwd`
```


## Team members

- Jie Wang
- Xiaodong Yang
- YuZhou Peng
- Ting Wang 
- Yanming Zhang


## Acknowledge

We acknowledge the excellent implementation from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) , [FastReID](https://github.com/JDAI-CV/fast-reid) and [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).

