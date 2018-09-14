# README

Papers similar to this:

* http://cs229.stanford.edu/proj2017/final-reports/5242076.pdf


## Dependencies

```shell
sudo apt-get update
sudo apt-get install --yes nvidia-headless-390 nvidia-utils-390 nvidia-cuda-toolkit python3-virtualenv libcupti-dev
```

## CUDA Setup

```shell
sudo /cuda/drivers/cuda_9.0.176_384.81_linux-run --override
sudo cp -P /cuda/drivers/cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/
sudo cp /cuda/drivers/cuda/include/cudnn.h /usr/local/cuda-9.0/include/
sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
echo 'PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}' | tee -a ~/.bashrc
sudo sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia.conf"
sudo ldconfig
sudo shutdown -r now
```

## Python Setup

```shell
python3 -m virtualenv -p /usr/bin/python3 venv
. venv/bin/activate
pip install keras numpy pandas tensorflow-gpu
sudo nvidia-smi -ac 877,1530
```
