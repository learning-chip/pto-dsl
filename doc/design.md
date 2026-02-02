Compile steps in order:
1. Pythonic frontend (hybrid tracing + ast), like https://github.com/zhangstevenunity/PTOAS/pull/13
2. Pybind calls to IR builder, like https://github.com/zhangstevenunity/PTOAS/pull/12
3. Get cpp source by `ptoas` on the PTO IR, with `--enable-insert-sync` as default, ref https://github.com/zhangstevenunity/PTOAS/issues/10
4. Preprocess cpp source: inserting macro guards and caller wrappers like [here](https://github.com/tile-ai/tilelang-ascend/blob/bf0109627418c0093fab1fddccbae58e63e4eb12/src/target/codegen_ascend_pto.cc#L2269)
5. Get shared lib by `bisheng` on the cpp source
6. Load so and kernel launch with torch_npu inputs
7. Accuracy and performance check in torch_npu

(5-7 are ready in https://gitcode.com/cann/pto-isa/pull/166, https://gitcode.com/cann/pto-isa/pull/239, https://gitcode.com/cann/pto-isa/pull/221)
