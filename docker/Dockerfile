FROM quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11

ARG ARCH=x86_64
# ARG ARCH=aarch64
ARG RELEASE_REPO=https://github.com/learning-chip/PTOAS
ARG RELEASE_TAG=20260205.post1

WORKDIR /installers/

ARG WHEEL_NAME=ptoas-0.1.1-cp311-none-manylinux_2_34_${ARCH}.whl
RUN wget $RELEASE_REPO/releases/download/${RELEASE_TAG}/${WHEEL_NAME}
RUN pip install ./${WHEEL_NAME}
RUN python -c "import mlir.ir; from mlir.dialects import pto"

ARG CLI_TAR_NAME=ptoas-bin-${ARCH}.tar.gz
RUN wget $RELEASE_REPO/releases/download/${RELEASE_TAG}/${CLI_TAR_NAME}
RUN tar zxf ./${CLI_TAR_NAME}
