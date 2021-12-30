# WPOD Net


This repo is a TensorRT C++ implement of WPOD Net.
Tested on platform: NVIDIA Jetson NX + Deepstream 5.0. 

## Config

- Set FP16 on/off in wpod.cpp
- Edit CONF_THRESH & NMS_THRESH in common.hpp

## Installation


### Install SVD libs
    cd singular
    mkdir build && cd build
    cmake ..
    make all && make install
    cd ..

### Build WPOD 
    mkdir build && cd build
    cmake ..
    make all
    

## Run


### Build WPOD TensorRT engine
    ./wpod c
After that, we can get wpod.engine here.

### Test 
    ./wpod <dir to image>
Output images will be save in /output folder.



## Deepstream-app

Test the engine with NVIDIA Deepstream. Just go to deepstream_app_wpod folder.

### Build lib
    cd nvdsinfer_wpod
    make all
    cd ..
libnvdsinfer_custom_impl_Wpod.so will appear. Note that you may have to edit path to some libraries in Makefile.

### Run deepstream-app
    deepstream-app -c deepstream_app_config_wpod.txt
Edit 2 config file: deepstream_app_config_wpod.txt & config_infer_primary.txt.
    

