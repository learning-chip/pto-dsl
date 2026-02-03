python matmul.py > matmul.pto
ptoas matmul.pto -o matmul_generated.cpp

# manually change: matmul_generated.cpp -> matmul_edited.cpp 
# TODO: use a Python script to patch source code

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I${PTO_LIB_PATH}/include \
    ./matmul_edited.cpp \
    -o ./matmul_kernel.so
