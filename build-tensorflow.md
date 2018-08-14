# Building Tensorflow 1.10 with CUDA 9.1 Support
#
# This is mainly pulled directly from the following blog post:
# https://medium.com/@asmello/how-to-install-tensorflow-cuda-9-1-into-ubuntu-18-04-b645e769f01d
#

## Setup LXC
lxc launch ubuntu-daily:bionic bionic-tensorflow
lxc exec bionic-tensorflow bash

##  Install Bazel
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
apt-get update
apt-get install --yes openjdk-8-jdk bazel

# Tensorflow Build Dependencies & CUDA Tools
apt-get install --yes nvidia-headless-390 nvidia-utils-390 nvidia-cuda-toolkit libcupti-dev python3-numpy python3-dev python3-pip python3-wheel

mkdir -p /usr/local/cuda /usr/local/cuda/extras/CUPTI /usr/local/cuda/nvvm
ln -s /usr/bin /usr/local/cuda/bin
ln -s /usr/include /usr/local/cuda/include
ln -s /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64
ln -s /usr/local/cuda/lib64 /usr/local/cuda/lib
ln -s /usr/include /usr/local/cuda/extras/CUPTI/include
ln -s /usr/lib/x86_64-linux-gnu /usr/local/cuda/extras/CUPTI/lib64
ln -s /usr/lib/nvidia-cuda-toolkit/libdevice /usr/local/cuda/nvvm/libdevice

# Install cuDNN and NCCL
# TODO: Download cuDNN and NCCL from nVidia, requires login
cd cuDNN
cp include/* /usr/local/cuda/include/
cp lib64/libcudnn.so.7.1.3 lib64/libcudnn_static.a /usr/local/cuda/lib64/
cd /usr/lib/x86_64-linux-gnu
ln -s libcudnn.so.7.1.3 libcudnn.so.7
ln -s libcudnn.so.7 libcudnn.so

# Install NCCL
cd NCCL
cp *.txt /usr/local/cuda/nccl
cp include/*.h /usr/include/
cp lib/libnccl.so.2.1.15 lib/libnccl_static.a /usr/lib/x86_64-linux-gnu/
ln -s /usr/include/nccl.h /usr/local/cuda/nccl/include/nccl.h
cd /usr/lib/x86_64-linux-gnu
ln -s libnccl.so.2.1.15 libnccl.so.2
ln -s libnccl.so.2 libnccl.so
for i in libnccl*; do ln -s /usr/lib/x86_64-linux-gnu/$i /usr/local/cuda/nccl/lib/$i; done

# Update python links
update-alternatives --install /usr/bin/python python /usr/bin/python3 100 --slave /usr/bin/pip pip /usr/bin/pip3

# Get Tensorflow and checkout 1.10
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout r1.10

# Example configure output:
# https://paste.ubuntu.com/p/74yhnKBrQk/
# 7.0 is the Tesla v100
./configure

bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
