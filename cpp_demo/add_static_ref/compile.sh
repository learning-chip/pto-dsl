bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    -I${ASCEND_TOOLKIT_HOME}/include \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -o add_lib.so add.cpp 
