#sportstrack

1st place of 2022 ECCV Sports MOT challenge.

## Result On SportsMOT test

|Method|HOTA|AssA|DetA|MOTA|
|---|---|---|---|---|
|SportsTrack|76.264|73.538|79.180|89.316|

## Quick Start

- set data root path of sports MOT dataset

```
export SPORTSTRACK_ROOT=...
```
- run alogrithm 

```
python main.py --split val --gpus 0
```

or 

```
python main.py --split test --gpus 0
```

- post process

```
python postprocess_tools/main.py
python postprocess_tools/interpolation.py
```

## requirements

- [wml2](https://github.com/vghost2008/wml2)
- pytorch


## Team members

- Jie Wang
- Xiaodong Yang
- Pengyu Zhou
- Ting Wang 
- Yanming Zhang


## Acknowledge

We acknowledge the excellent implementation from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) , [FastReID](https://github.com/JDAI-CV/fast-reid) and [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).





