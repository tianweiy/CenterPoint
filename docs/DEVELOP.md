# Develop

This document provide tutorials to develop CenterPoint. 

## New dataset 

TODO

## New Task 

- If you interested in developing tracking algorithms based on our detection results, please refer to [NUSC](NUSC.md) and [WAYMO](WAYMO.md). For more advanced tasks like motion prediction, you may need to store the final bev feature map computed [here](https://github.com/tianweiy/CenterPoint/blob/1ecebf980f75cfe7f53cc52032b184192891c9b9/det3d/models/necks/rpn.py#L159). 

- You will also need to add files to [`det3d/datasets/pipelines/preprocess.py`](../det3d/datasets/pipelines/preprocess.py) to specify the data generation during training and training. 

- You may also need to change the collate function in [collate.py](https://github.com/tianweiy/CenterPoint/blob/1ecebf980f75cfe7f53cc52032b184192891c9b9/det3d/torchie/parallel/collate.py#L91) and data_loading function in [trainer.py](https://github.com/tianweiy/CenterPoint/blob/1ecebf980f75cfe7f53cc52032b184192891c9b9/det3d/torchie/trainer/trainer.py#L34) 

## New Architecture 

Please add any 3D backbone in `det3d/models/backbones`, any 2D backbones in `det3d/models/necks`, and any two-stage refinement modules in `det3d/models/second_stage`. 

If you have any suggestions for improving this codebase for development, please open an issue or send us an email. 