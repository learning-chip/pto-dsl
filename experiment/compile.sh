python ./relu_builder.py > relu.pto
ptoas relu.pto -o relu.cpp 2>debug_as.log

# temporary fix for macro guards, better let `ptoas` insert it
(echo '#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)'; cat relu.cpp; echo '#endif') > relu.cpp.tmp && mv relu.cpp.tmp relu.cpp

bisheng \
    -I${PTO_LIB_PATH}/include/pto \
    -fPIC -shared -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    ./relu.cpp \
    -o ./relu_kernel.so
