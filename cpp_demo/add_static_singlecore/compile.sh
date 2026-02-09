bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I${ASCEND_TOOLKIT_HOME}/include \
    ./add.cpp \
    -o ./add.so

# $ASCEND_TOOLKIT_HOME points /usr/local/Ascend/cann-8.5.0