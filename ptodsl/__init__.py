"""
ptodsl: high-level DSL for PTO MLIR.
Use ir_builder() and register_function to build modules that match low-level mlir.ir/mlir.dialects.
"""

from contextlib import contextmanager
from mlir.ir import Context, Location, Module, InsertionPoint, Attribute
from mlir.dialects import func, pto

from ptodsl import language


class _BuilderContext:
    def __init__(self, ctx, module, insertion_point):
        self.ctx = ctx
        self.module = module
        self.insertion_point = insertion_point
        self.attrs = None


@contextmanager
def ir_builder():
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)
        m = Module.create()
        with InsertionPoint(m.body):
            builder = _BuilderContext(ctx, m, InsertionPoint(m.body))
            builder.attrs = {
                "PIPE_MTE2": Attribute.parse("#pto.pipe<PIPE_MTE2>"),
                "PIPE_V": Attribute.parse("#pto.pipe<PIPE_V>"),
                "PIPE_MTE3": Attribute.parse("#pto.pipe<PIPE_MTE3>"),
                "EVENT_ID0": Attribute.parse("#pto.event<EVENT_ID0>"),
            }
            token = language._builder_var.set(builder)
            try:
                yield m
            finally:
                language._builder_var.reset(token)
        m.operation.verify()


def register_function(fn):
    """Decorator: add the function as an MLIR FuncOp to the current module."""
    import inspect
    sig = inspect.signature(fn)
    param_types = [
        sig.parameters[p].annotation
        for p in sig.parameters
        if sig.parameters[p].annotation is not inspect.Parameter.empty
    ]
    if len(param_types) != len(sig.parameters):
        raise ValueError("All parameters must be annotated with types")
    fn_ty = func.FunctionType.get(param_types, [])
    func_op = func.FuncOp(fn.__name__, fn_ty)
    with InsertionPoint(func_op.add_entry_block()):
        fn(*func_op.arguments)
        func.ReturnOp([])
    return fn
