bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I${PTO_LIB_PATH}/include \
    ./matmul_edited.cpp \
    -o ./matmul_kernel.so
