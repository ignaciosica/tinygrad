from typing import Union
import numpy as np
import unittest
from dataclasses import replace

from tinygrad.codegen.kernel import Opt, OptOps, KernelOptError, Kernel
from tinygrad.ops import UOp, Ops
from tinygrad.device import Device, Buffer
from tinygrad.tensor import Tensor
from tinygrad.engine.realize import run_schedule, CompiledRunner
from tinygrad.helpers import Context
from tinygrad.dtype import dtypes

def helper_realized_ast(r:Union[Tensor, list[Tensor]]) -> tuple[UOp, list[Buffer]]:
  if isinstance(r, Tensor): r = [r]
  s = Tensor.schedule(*r)
  run_schedule(s[:-1])
  assert s[-1].ast.op is Ops.SINK, f"helper_realized_ast expects a SINK {s[-1]}"
  bufs = [Buffer((x).device, x.size, x.dtype).allocate() if i < len(s[-1].ast.src) else x for i,x in enumerate(s[-1].bufs)]
  return s[-1].ast, bufs

def helper_lds_allclose(opts:list[Opt], expected_bufs, N=16, M=16, K=16, dtype_in=dtypes.float, acc_dtype=dtypes.float, append_lds_opts=True):
  with Context(DEBUG=0): a, b = Tensor.rand(M, K, dtype=dtype_in).realize(), Tensor.rand(K, N, dtype=dtype_in).realize()
  realized_ast, bufs = helper_realized_ast(a.matmul(b, dtype=acc_dtype))
  k = Kernel(realized_ast)
  if append_lds_opts: opts += [Opt(OptOps.LDS, 0, None), Opt(OptOps.LDS, 1, None), Opt(OptOps.LDS, 2, None)]
  for opt in opts:
    k.apply_opt(opt)
  prg = k.to_program()
  CompiledRunner(replace(prg, device=Device.DEFAULT)).exec(bufs)

  atol, rtol = 1e-4, 1e-4
  if dtype_in == dtypes.half: atol, rtol = 1e-2, 1e-2
  np.testing.assert_allclose(bufs[0].numpy().reshape((M,N)), a.numpy() @ b.numpy(), atol=atol, rtol=rtol)

  assert k.smem_usage == sum([sz for _, sz in expected_bufs]), f"{k.smem_usage=} != {sum([sz for _, sz in expected_bufs])=}"
  assert k.smem_usage <= k.opts.shared_max, f"{k.smem_usage} > {k.opts.shared_max}"

  local_buffers = [uop for uop in k.uops if uop.op is Ops.DEFINE_LOCAL]
  assert len(local_buffers) == len(expected_bufs), f"Expected exactly {len(expected_bufs)} local buffers, got {len(local_buffers)}"
  for i,(buf, sz) in enumerate(expected_bufs):
    assert local_buffers[i].arg == f"lds{buf}", f"Expected buffer argument index {buf}, got {local_buffers[i].arg}"
    expected_dtype = (acc_dtype if buf == 0 else dtype_in).ptr(sz, local=True)
    assert local_buffers[i].dtype == expected_dtype, f"Expected buffer dtype {expected_dtype}, got {local_buffers[i].dtype} for {opts=}"
    # TODO: check all access to the global buffer are proxied through the local buffer

@unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
class TestLDS(unittest.TestCase):
  def test_lds_args(self):
    realized_ast, _ = helper_realized_ast(Tensor.rand(4, 4) @ Tensor.rand(4, 4))
    k = Kernel(realized_ast)
    valid_opts = [Opt(OptOps.LDS, 0, None),
                  Opt(OptOps.LDS, 1, None),
                  Opt(OptOps.LDS, 2, None)]
    for opt in valid_opts:
      k = Kernel(realized_ast)
      k.apply_opt(opt)

    invalid_opts = [Opt(OptOps.LDS, -1, None),
                    Opt(OptOps.LDS, 3, None)]
    for opt in invalid_opts:
      k = Kernel(realized_ast)
      with self.assertRaises(KernelOptError):
        k.apply_opt(opt)

  def test_lds_basic(self):
    # output
    helper_lds_allclose(opts=[Opt(OptOps.LDS, 0, None)], expected_bufs=[(0,1)], append_lds_opts=False)
    # input
    helper_lds_allclose(opts=[Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LDS, 1, None)], expected_bufs=[(1,4)], append_lds_opts=False)
    helper_lds_allclose(opts=[Opt(OptOps.UNROLL, 0, 4),Opt(OptOps.LDS, 2, None)], expected_bufs=[(2,4)], append_lds_opts=False)
    # multi
    helper_lds_allclose(opts=[Opt(OptOps.LDS, 0, None), Opt(OptOps.LDS, 1, None)], expected_bufs=[(0,1),(1,1)], append_lds_opts=False)
    helper_lds_allclose(opts=[], expected_bufs=[(0,1),(1,1),(2,1)])

  def test_lds_unroll(self):
    # unroll doesn't change local output buffer size
    for sz in [2,4,8]:
      helper_lds_allclose(opts=[Opt(OptOps.UNROLL, 0, sz)], expected_bufs=[(0,1),(1,sz),(2,sz)])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_lds_local(self):
    # if only locals are applied, local buffer size for output should be prod(locals)

    basic_local_opts = [Opt(OptOps.LOCAL, 0, 2)]
    helper_lds_allclose(opts=basic_local_opts, expected_bufs=[(0,2),(1,2),(2,1)])

    multi_local_opts = [Opt(OptOps.LOCAL, 0, 2),
                        Opt(OptOps.LOCAL, 0, 8)]
    helper_lds_allclose(opts=multi_local_opts, expected_bufs=[(0,16),(1,16),(2,1)])

    multi_axis_local_opts = [Opt(OptOps.LOCAL, 1, 4),
                             Opt(OptOps.LOCAL, 0, 2)]
    helper_lds_allclose(opts=multi_axis_local_opts, expected_bufs=[(0,8),(1,2),(2,4)])

    full_local_opts = [Opt(OptOps.LOCAL, 0, 16),
                       Opt(OptOps.LOCAL, 0, 16)]
    helper_lds_allclose(opts=full_local_opts, expected_bufs=[(0,256),(1,16),(2,16)])

  def test_lds_upcast(self):
    # if only upcasts are applied, local buffer size for output should be prod(upcast)

    basic_upcast_opts = [Opt(OptOps.UPCAST, 0, 2)]
    helper_lds_allclose(opts=basic_upcast_opts, expected_bufs=[(0,2),(1,2),(2,1)])

    multi_upcast_opts = [Opt(OptOps.UPCAST, 0, 2),
                         Opt(OptOps.UPCAST, 0, 8)]
    helper_lds_allclose(opts=multi_upcast_opts, expected_bufs=[(0,16),(1,16),(2,1)])

    multi_axis_upcast_opts = [Opt(OptOps.UPCAST, 1, 4),
                              Opt(OptOps.UPCAST, 0, 2)]
    helper_lds_allclose(opts=multi_axis_upcast_opts, expected_bufs=[(0,8),(1,2),(2,4)])

    full_upcast_opts = [Opt(OptOps.UPCAST, 0, 16),
                        Opt(OptOps.UPCAST, 0, 16)]
    helper_lds_allclose(opts=full_upcast_opts, expected_bufs=[(0,256),(1,16),(2,16)])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_lds_tc(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if tc.dtype_in == dtypes.bfloat16 or tc.dtype_out == dtypes.bfloat16: continue
      (N, M, K) = tc.dims
      sz = 64

      opts = [Opt(OptOps.TC, 0, (-1, 0))]
      helper_lds_allclose(opts=opts, expected_bufs=[(0,M*N),(1,K*N),(2,M*K)], N=N, M=M, K=K, dtype_in=tc.dtype_in, acc_dtype=tc.dtype_out)

      opts = [Opt(OptOps.TC, 0, (-1, 0)),
              Opt(OptOps.LOCAL, 0, 2),
              Opt(OptOps.UPCAST, 1, 2)]
      helper_lds_allclose(opts=opts, expected_bufs=[(0,M*N*4),(1,K*N*2),(2,M*K*2)], N=sz, M=sz, K=sz, dtype_in=tc.dtype_in, acc_dtype=tc.dtype_out)

      opts = [Opt(OptOps.TC, 0, (-1, 0)),
              Opt(OptOps.UNROLL, 0, 2)]
      helper_lds_allclose(opts=opts, expected_bufs=[(0,M*N),(1,K*N*2),(2,M*K*2)], N=sz, M=sz, K=sz, dtype_in=tc.dtype_in, acc_dtype=tc.dtype_out)

      opts = [Opt(OptOps.TC, 0, (-1, 0)),
              Opt(OptOps.UNROLL, 0, 2),
              Opt(OptOps.UPCAST, 1, 2)]
      helper_lds_allclose(opts=opts, expected_bufs=[(0,M*N*2),(1,K*N*2),(2,M*K*4)], N=sz, M=sz, K=sz, dtype_in=tc.dtype_in, acc_dtype=tc.dtype_out)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_lds_full(self):
    opts = [Opt(OptOps.LOCAL, 0, 2),
            Opt(OptOps.UPCAST, 1, 2)]
    helper_lds_allclose(opts=opts, expected_bufs=[(0,4),(1,2),(2,2)])

    opts = [Opt(OptOps.LOCAL, 0, 2),
            Opt(OptOps.UPCAST, 0, 4),
            Opt(OptOps.LOCAL, 1, 8)]
    helper_lds_allclose(opts=opts, expected_bufs=[(0,64),(1,8),(2,8)])

    opts = [Opt(OptOps.LOCAL, 0, 16),
            Opt(OptOps.UPCAST, 1, 2)]
    helper_lds_allclose(opts=opts, expected_bufs=[(0,16),(1,16),(2,1)]) # upcasting local dim

    opts = [Opt(OptOps.LOCAL, 0, 16),
            Opt(OptOps.UPCAST, 0, 16)]
    helper_lds_allclose(opts=opts, expected_bufs=[(0,256),(1,16),(2,16)])

    opts = [Opt(OptOps.LOCAL, 1, 16),
            Opt(OptOps.UPCAST, 1, 16)]
    helper_lds_allclose(opts=opts, expected_bufs=[(0,16),(1,1),(2,16)])

    opts = [Opt(OptOps.LOCAL, 1, 4),
            Opt(OptOps.UNROLL, 0, 2),
            Opt(OptOps.UPCAST, 0, 2)]
    helper_lds_allclose(opts=opts, expected_bufs=[(0,8),(1,4),(2,8)])

if __name__ == '__main__':
  unittest.main()