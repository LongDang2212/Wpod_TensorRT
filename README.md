/************************ WPOD Net ************************/

/**** Config ****/

    - Edit directory to weight file (.wts) in wpod.cpp
    - Set FP16 on/off
    - Edit CONF_THRESH & NMS_THRESH in common.hpp

/**** Build ****/

    - Build SVD lib:    git clone https://github.com/kikuomax/singular.git
			cd singular
                        mkdir build && cd build
                        cmake ..
                        make all && make install
    - Build:            cd .. 
                        mkdir build && cd build
                        cmake ..
                        make all

/**** Run ****/

    - Build engine: ./wpod c
    - Test:         ./wpod <dir_to_image>

/**** Deepstream-app ****/

    - Build lib: cd nvdsinfer_wpod
                 make all
                 cd ../deepstream_app_wpod
    - Config: Edit config file
    - Run# Wpod_TensorRT
