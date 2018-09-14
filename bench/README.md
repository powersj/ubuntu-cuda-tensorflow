# ubuntu-cuda-tensorflow

Personal notes of playing with Tensorflow with CUDA support. Includes building Tensorflow from source to support CUDA 9.1 in Ubuntu 18.04 LTS.

## Running bench.py

Below is an example of using an AWS p3.2xlarge instance with the daily Bionic image.

First, update system and install nVidia CUDA :

```shell
sudo apt-get update
sudo apt-get install --yes nvidia-headless-390 nvidia-utils-390 nvidia-cuda-toolkit python3-virtualenv
sudo shutdown -r now
```

After reboot, setup the GPU for max. frequency, configure a virtualenv, and install a locally configured and built version of Tensorflow from a Python wheel:

```shell
sudo nvidia-smi -ac 877,1530
python3 -m virtualenv -p /usr/bin/python3 venv
. venv/bin/activate
pip install tabulate
pip install tensorflow.whl
```

Then the user can run the bench command and get output:

```shell
$ ./bench
<output TBD>
```

## Alternative Method

```shell
sudo apt-get update
sudo apt-get upgrade --yes
sudo apt-get install --yes nvidia-headless-390 nvidia-utils-390 nvidia-cuda-toolkit python3-virtualenv
# On new /numa volume
sudo ./cuda_9.0.176_384.81_linux-run --tmpdir=/cuda/tmp --override
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda-9.0/include/
sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
sudo shutdown -r now
```

### Setup CUDA

```shell
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
```
