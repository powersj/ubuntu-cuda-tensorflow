# Building Tensorflow 1.10 with CUDA 9.1 Support

This is mainly pulled directly from an excellent [blog post](https://medium.com/@asmello/how-to-install-tensorflow-cuda-9-1-into-ubuntu-18-04-b645e769f01d)

## Setup LXC

Best to do this in an isolated enviornment like a LXC:

```shell
lxc launch ubuntu-daily:bionic bionic-tensorflow
lxc exec bionic-tensorflow bash
```

##  Install Bazel

This is the build tool:

```shell
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
apt-get update
apt-get install --yes openjdk-8-jdk bazel
```

## Tensorflow Build Dependencies & CUDA Tools

This includes all the necessary build tools and CUDA packages. Afterwards it links all the installed packages to the `/usr/local/cuda` directory which is what Tensorflow is expecting.

```shell
apt-get install --yes nvidia-headless-390 nvidia-utils-390 nvidia-cuda-toolkit libcupti-dev python3-numpy python3-dev python3-pip python3-wheel

mkdir -p /usr/local/cuda /usr/local/cuda/extras/CUPTI /usr/local/cuda/nvvm
sudo ln -s /usr/bin /usr/local/cuda/bin
sudo ln -s /usr/include /usr/local/cuda/include
sudo ln -s /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64
sudo ln -s /usr/local/cuda/lib64 /usr/local/cuda/lib
sudo ln -s /usr/include /usr/local/cuda/extras/CUPTI/include
sudo ln -s /usr/lib/x86_64-linux-gnu /usr/local/cuda/extras/CUPTI/lib64
sudo ln -s /usr/lib/nvidia-cuda-toolkit/libdevice /usr/local/cuda/nvvm/libdevice
```

## Install cuDNN and NCCL

I need to investigate how much of these next two steps are required. However, in the example blog post the user is required to download cuDNN and NCCL from nVidia, which requires a login. Then similiar to the nVidia drivers, the user has to move these files into an area that is more expected by the Tensorflow build, namely `/usr/local/cuda`:

```shell
# wherever you unziped the downloaded versions to
cd /root/cuda
sudo cp include/* /usr/local/cuda/include/
# assumes version 7.1.3
sudo cp lib64/libcudnn.so.7.1.3 lib64/libcudnn_static.a /usr/local/cuda/lib64/
cd /usr/lib/x86_64-linux-gnu
sudo ln -s libcudnn.so.7.1.3 libcudnn.so.7
sudo ln -s libcudnn.so.7 libcudnn.so

# wherever you unziped the downloaded versions to
cd /root/NCCL
sudo cp *.txt /usr/local/cuda/nccl
sudo cp include/*.h /usr/include/
# assumes version 2.1.15
sudo cp lib/libnccl.so.2.1.15 lib/libnccl_static.a /usr/lib/x86_64-linux-gnu/
sudo ln -s /usr/include/nccl.h /usr/local/cuda/nccl/include/nccl.h
cd /usr/lib/x86_64-linux-gnu
sudo ln -s libnccl.so.2.1.15 libnccl.so.2
sudo ln -s libnccl.so.2 libnccl.so
for i in libnccl*; do sudo ln -s /usr/lib/x86_64-linux-gnu/$i /usr/local/cuda/nccl/lib/$i; done
```

## Link Python to Python3

There is no python executable in Ubuntu 18.04 LTS and this expects one. While I would never recommend anyone ever do something like this it will allow the build to complete.

```shell
update-alternatives --install /usr/bin/python python /usr/bin/python3 100 --slave /usr/bin/pip pip /usr/bin/pip3
```

## Tensorflow

After all that, it is time for the main event. Here is an example set of [configure](https://paste.ubuntu.com/p/74YJCfTG6W/) answers. GCC 7 and later are not supported and for the record the 7.0 is the Tesla v100.

```shell
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout r1.10
./configure
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```
