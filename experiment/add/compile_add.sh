bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 \
    -I${PTO_LIB_PATH}/include \
    -I${PTO_LIB_PATH}/include/common \
    -o add_custom_lib.so add_custom.cpp 
