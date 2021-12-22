## Installation
Modified from [det3d](https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)'s original document.

### Requirements

- Linux
- Python 3.6+
- PyTorch 1.1 or higher
- CUDA 10.0 or higher
- CMake 3.13.2 or higher
- [APEX](https://github.com/nvidia/apex)
- [spconv](https://github.com/traveller59/spconv/commit/73427720a539caf9a44ec58abe3af7aa9ddb8e39) 

#### Notes
- Both Spconv 1.x and 2.x work.
- Recent pytorch/spconv/cuda version will be faster and consume less memory. 
- If you have problem installing apex, you can change the apex syncbn to torch's native sync bn at https://github.com/tianweiy/CenterPoint/blob/3fd0b8745b77575cb9810035aafc76796613f942/det3d/torchie/apis/train.py#L268.

we have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04
- Python: 3.6.5/3.7.10 
- PyTorch: 1.1/1.9/1.10.1
- spconv: 1.0/1.2.1/master
- CUDA: 10.0/11.1

### Basic Installation 

```bash
# basic python libraries
conda create --name centerpoint python=3.6
conda activate centerpoint
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
git clone https://github.com/tianweiy/CenterPoint.git
cd CenterPoint
pip install -r requirements.txt

# add CenterPoint to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_CENTERPOINT"
```

### Advanced Installation 

#### nuScenes dev-kit

```bash
git clone https://github.com/tianweiy/nuscenes-devkit

# add the following line to ~/.bashrc and reactivate bash (remember to change the PATH_TO_NUSCENES_DEVKIT value)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_NUSCENES_DEVKIT/python-sdk"
```

#### Cuda Extensions

```bash
# set the cuda path(change the path to your own cuda location) 
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_PATH=/usr/local/cuda-10.0
export CUDA_HOME=/usr/local/cuda-10.0
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

# Rotated NMS 
cd ROOT_DIR/det3d/ops/iou3d_nms
python setup.py build_ext --inplace

# Deformable Convolution (Optional and only works with old torch versions e.g. 1.1)
cd ROOT_DIR/det3d/ops/dcn
python setup.py build_ext --inplace
```

#### APEX (Optional)

```bash
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 5633f6  # recent commit doesn't build in our system 
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### spconv
```bash
sudo apt-get install libboost-all-dev
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv && git checkout 7342772
python setup.py bdist_wheel
cd ./dist && pip install *
```

#### Check out [GETTING_START](GETTING_START.md) to prepare the data and play with all those pretrained models. 
