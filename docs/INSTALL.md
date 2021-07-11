## Installation
Modified from [det3d](https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)'s original document.

### Requirements

- Linux
- Python 3.6+
- PyTorch 1.1 or higher
- CUDA 10.0 or higher
- CMake 3.13.2 or higher
- [APEX](https://github.com/nvidia/apex)
- [`spconv v1.0 (commit 734277)`](https://github.com/traveller59/spconv/commit/73427720a539caf9a44ec58abe3af7aa9ddb8e39) or [`spconv v1.2`](https://github.com/traveller59/spconv)


### Install CenterPoint  

```shell
# this assumes that pytorch is already installed 
git clone https://github.com/tianweiy/CenterPoint.git  && cd CenterPoint
pip install -r requirements.txt

# add CenterPoint to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_CENTERPOINT"
```

#### spconv
- If you use PyTorch 1.1, then make sure you install the `spconv v1.0` with ([commit 734277](https://github.com/traveller59/spconv/commit/73427720a539caf9a44ec58abe3af7aa9ddb8e39)) instead of the latest one.
- If you use PyTorch 1.3+, then you need to install the `spconv v1.2`. 
- Try to use `spconv v1.2` if you can, it consumes far less GPU memory than `spconv v1.0` and enables you to train VoxelNet model on Waymo using a 11GB GPU. 

#### APEX (Distributed Training Only)

```shell
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 5633f6  # recent commit doesn't build in our system 
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### CUDA Extensions
```shell
bash setup.sh  
```

#### Check out [GETTING_START](GETTING_START.md) to prepare the data and play with all those pretrained models. 