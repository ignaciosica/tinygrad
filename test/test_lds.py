from typing import cast
import numpy as np
import torch
import unittest
from dataclasses import replace

from tinygrad.codegen.kernel import Opt, OptOps, KernelOptError, Kernel
from tinygrad.ops import Ops
from tinygrad.device import Device
from tinygrad.tensor import Tensor
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import Context, CI, getenv
from tinygrad.dtype import dtypes, PtrDType, _to_np_dtype
from test.test_linearizer import helper_realized_ast

def helper_lds_allclose(r:Tensor, desired:np.ndarray, opts:list[Opt]|None=None, desired_bufs_sizes:list[tuple[int, int]]|None=None,
                        rtol:float=1e-5, atol:float=1e-5, apply_lds=True):
  realized_ast, bufs = helper_realized_ast(r)

  if opts is None: opts = []
  if desired_bufs_sizes is None: desired_bufs_sizes = [(i, 1) for i in range(len(bufs))]
  if apply_lds: opts += [Opt(OptOps.LDS, i, None) for i in range(len(bufs))]

  k = Kernel(realized_ast).apply_opts(opts)
  CompiledRunner(replace(k.to_program(), device=Device.DEFAULT)).exec(bufs)

  np.testing.assert_allclose(bufs[0].numpy().reshape(r.shape), desired, atol=atol, rtol=rtol)
  local_bufs = [uop for uop in k.uops if uop.op is Ops.DEFINE_LOCAL]
  global_bufs = [uop for uop in k.uops if uop.op is Ops.DEFINE_GLOBAL]

  assert k.smem_usage == sum([sz for _, sz in desired_bufs_sizes]), f"{k.smem_usage=} != {sum([sz for _, sz in desired_bufs_sizes])=}"
  assert k.smem_usage <= k.opts.shared_max, f"{k.smem_usage} > {k.opts.shared_max}"
  assert len(local_bufs) == len(desired_bufs_sizes), f"Expected exactly {len(desired_bufs_sizes)} local buffers, got {len(local_bufs)}"

  for i,(buf, sz) in enumerate(desired_bufs_sizes):
    assert local_bufs[i].arg == f"lds{buf}", f"Expected buffer argument index lds{buf}, got {local_bufs[i].arg}"
    assert local_bufs[i].dtype.base == global_bufs[i].dtype.base, f"Buffer base dtype mismatch {local_bufs[i].dtype=} {global_bufs[i].dtype=}"
    assert cast(PtrDType,local_bufs[i].dtype).size == sz, f"Expected buffer sz {sz}, got {cast(PtrDType,local_bufs[i].dtype).size=} for {opts=}"

def helper_lds_matmul(opts:list[Opt], desired_bufs_sizes, N=32, M=64, K=16, dtype_in=dtypes.float, acc_dtype=dtypes.float, apply_lds=True):
  def init_matrix(rows, cols, dtype_in) -> Tensor:
    rng = np.random.default_rng()
    if (np_dtype := _to_np_dtype(dtype_in)) is None: np_dtype = np.float32
    return Tensor(rng.random((rows, cols), dtype=np.float32).astype(np_dtype)).cast(dtype_in).realize()
  with Context(DEBUG=0): a, b = init_matrix(M, K, dtype_in), init_matrix(K, N, dtype_in)
  rtol, atol = 3e-2 if dtype_in is dtypes.half else 1e-4, 1e-4

  helper_lds_allclose(a.matmul(b, dtype=acc_dtype), a.numpy() @ b.numpy(), opts, desired_bufs_sizes, rtol, atol, apply_lds)

@unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "tests require shared")
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

  def test_invalid_lds(self):
    realized_ast, _ = helper_realized_ast(Tensor.rand(4, 4) @ Tensor.rand(4, 4))
    invalid_opts = [
      [Opt(OptOps.PADTO, 1, 7), Opt(OptOps.LDS, 2, None)],
      [Opt(OptOps.LDS, 2, None), Opt(OptOps.PADTO, 1, 7)],
      [Opt(OptOps.LDS, 2, None), Opt(OptOps.UPCAST, 2, 2)],
      [Opt(OptOps.LDS, 0, None), Opt(OptOps.LOCAL, 2, 2)],
      [Opt(OptOps.LDS, 0, None), Opt(OptOps.UNROLL, 0, 2)],
      [Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.LDS, 0, None)],
      [Opt(OptOps.LDS, 0, None), Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.GROUP, 0, 2), Opt(OptOps.LDS, 0, None)],
      [Opt(OptOps.LDS, 0, None), Opt(OptOps.GROUP, 0, 2)],
    ]
    for opts in invalid_opts:
      k = Kernel(realized_ast)
      with self.assertRaises(KernelOptError):
        for opt in opts: k.apply_opt(opt)

  def test_invalid_lds_pad(self):
    realized_ast, _ = helper_realized_ast(Tensor.rand(20,7).pad(((0,32),(0,16))).sum())
    k = Kernel(realized_ast)
    with self.assertRaises(KernelOptError): k.apply_opt(Opt(OptOps.LDS, 1, None))

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_lds_basic(self):
    # output
    helper_lds_matmul(opts=[Opt(OptOps.LDS, 0, None)], desired_bufs_sizes=[(0,1)], apply_lds=False)
    # input
    helper_lds_matmul(opts=[Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LDS, 1, None)], desired_bufs_sizes=[(1,4)], apply_lds=False)
    helper_lds_matmul(opts=[Opt(OptOps.UNROLL, 0, 4),Opt(OptOps.LDS, 2, None)], desired_bufs_sizes=[(2,4)], apply_lds=False)
    # multi
    helper_lds_matmul(opts=[Opt(OptOps.LDS, 0, None), Opt(OptOps.LDS, 1, None)], desired_bufs_sizes=[(0,1),(1,1)], apply_lds=False)
    helper_lds_matmul(opts=[], desired_bufs_sizes=[(0,1),(1,1),(2,1)])

  def test_lds_unroll(self):
    # unroll doesn't change local output buffer size
    for sz in [2,4,8]:
      helper_lds_matmul(opts=[Opt(OptOps.UNROLL, 0, sz)], desired_bufs_sizes=[(0,1),(1,sz),(2,sz)])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_lds_local(self):
    # if only locals are applied, local buffer size for output should be prod(locals)

    basic_local_opts = [Opt(OptOps.LOCAL, 0, 2)]
    helper_lds_matmul(opts=basic_local_opts, desired_bufs_sizes=[(0,2),(1,2),(2,1)])

    multi_local_opts = [Opt(OptOps.LOCAL, 0, 2),
                        Opt(OptOps.LOCAL, 0, 8)]
    helper_lds_matmul(opts=multi_local_opts, desired_bufs_sizes=[(0,16),(1,16),(2,1)])

    multi_axis_local_opts = [Opt(OptOps.LOCAL, 1, 4),
                             Opt(OptOps.LOCAL, 0, 2)]
    helper_lds_matmul(opts=multi_axis_local_opts, desired_bufs_sizes=[(0,8),(1,2),(2,4)])

    full_local_opts = [Opt(OptOps.LOCAL, 0, 64),
                       Opt(OptOps.LOCAL, 0, 4)]
    helper_lds_matmul(N=4, opts=full_local_opts, desired_bufs_sizes=[(0,256),(1,64),(2,4)])

  def test_lds_upcast(self):
    # if only upcasts are applied, local buffer size for output should be prod(upcast)

    basic_upcast_opts = [Opt(OptOps.UPCAST, 0, 2)]
    helper_lds_matmul(opts=basic_upcast_opts, desired_bufs_sizes=[(0,2),(1,2),(2,1)])

    multi_upcast_opts = [Opt(OptOps.UPCAST, 0, 2),
                         Opt(OptOps.UPCAST, 0, 8)]
    helper_lds_matmul(opts=multi_upcast_opts, desired_bufs_sizes=[(0,16),(1,16),(2,1)])

    multi_axis_upcast_opts = [Opt(OptOps.UPCAST, 1, 4),
                              Opt(OptOps.UPCAST, 0, 2)]
    helper_lds_matmul(opts=multi_axis_upcast_opts, desired_bufs_sizes=[(0,8),(1,2),(2,4)])

    full_upcast_opts = [Opt(OptOps.UPCAST, 0, 8),
                        Opt(OptOps.UPCAST, 0, 8),
                        Opt(OptOps.UPCAST, 0, 16)]
    helper_lds_matmul(opts=full_upcast_opts, desired_bufs_sizes=[(0,1024),(1,64),(2,16)])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_lds_tc(self):
    for i, tc in enumerate(Device[Device.DEFAULT].renderer.tensor_cores):
      if tc.dtype_in is dtypes.bfloat16 or tc.dtype_out is dtypes.bfloat16: continue
      (N, M, K) = tc.dims
      sz = 64

      opts = [Opt(OptOps.TC, 0, (i, 0, 1))]
      helper_lds_matmul(opts, desired_bufs_sizes=[(0,M*N),(1,M*K),(2,K*N)], N=N, M=M, K=K*2, dtype_in=tc.dtype_in, acc_dtype=tc.dtype_out)

      opts = [Opt(OptOps.TC, 0, (i, 0, 1)),
              Opt(OptOps.UNROLL, 0, 2)]
      helper_lds_matmul(opts, desired_bufs_sizes=[(0,M*N),(1,M*K*2),(2,K*N*2)], N=sz, M=sz, K=sz, dtype_in=tc.dtype_in, acc_dtype=tc.dtype_out)

      opts = [Opt(OptOps.TC, 0, (i, 0, 1)),
              Opt(OptOps.UNROLL, 0, 2),
              Opt(OptOps.UPCAST, 1, 2)]
      helper_lds_matmul(opts, desired_bufs_sizes=[(0,M*N*2),(1,M*K*2),(2,K*N*4)], N=sz, M=sz, K=sz, dtype_in=tc.dtype_in, acc_dtype=tc.dtype_out)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_lds_tc_local(self):
    for i, tc in enumerate(Device[Device.DEFAULT].renderer.tensor_cores):
      if tc.dtype_in is dtypes.bfloat16 or tc.dtype_out is dtypes.bfloat16: continue
      (N, M, K) = tc.dims
      sz = 64

      opts = [Opt(OptOps.TC, 0, (i, 0, 1)),
              Opt(OptOps.LOCAL, 0, 2),
              Opt(OptOps.UPCAST, 1, 2)]
      helper_lds_matmul(opts, desired_bufs_sizes=[(0,M*N*4),(1,M*K*2),(2,K*N*2)], N=sz, M=sz, K=sz, dtype_in=tc.dtype_in, acc_dtype=tc.dtype_out)

      opts = [Opt(OptOps.TC, 0, (i, 0, 1)),
              Opt(OptOps.LOCAL, 0, 2),
              Opt(OptOps.LOCAL, 1, 2)]
      helper_lds_matmul(opts, desired_bufs_sizes=[(0,M*N*4),(1,M*K*2),(2,K*N*2)], N=sz, M=sz, K=sz, dtype_in=tc.dtype_in, acc_dtype=tc.dtype_out)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_lds_full(self):
    opts = [Opt(OptOps.LOCAL, 0, 2),
            Opt(OptOps.UPCAST, 1, 2),
            Opt(OptOps.SWAP, 0, 1)]
    helper_lds_matmul(opts=opts, desired_bufs_sizes=[(0,4),(1,2),(2,2)])

    opts = [Opt(OptOps.LOCAL, 0, 2),
            Opt(OptOps.UPCAST, 0, 4),
            Opt(OptOps.LOCAL, 1, 8)]
    helper_lds_matmul(opts=opts, desired_bufs_sizes=[(0,64),(1,8),(2,8)])

    opts = [Opt(OptOps.LOCAL, 0, 64),
            Opt(OptOps.UPCAST, 1, 2)]
    helper_lds_matmul(opts=opts, desired_bufs_sizes=[(0,64),(1,64),(2,1)]) # upcasting local dim

    opts = [Opt(OptOps.LOCAL, 0, 64),
            Opt(OptOps.UPCAST, 0, 16)]
    helper_lds_matmul(opts=opts, desired_bufs_sizes=[(0,1024),(1,64),(2,16)])

    opts = [Opt(OptOps.LOCAL, 1, 16),
            Opt(OptOps.UPCAST, 1, 2)]
    helper_lds_matmul(opts=opts, desired_bufs_sizes=[(0,32),(1,1),(2,32)])

    opts = [Opt(OptOps.LOCAL, 1, 4),
            Opt(OptOps.UNROLL, 0, 2),
            Opt(OptOps.UPCAST, 0, 2)]
    helper_lds_matmul(opts=opts, desired_bufs_sizes=[(0,8),(1,4),(2,8)])

@unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "tests require shared")
class TestLDSOps(unittest.TestCase):

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "tests require locals")
  def test_lds_transpose(self):
    with Context(DEBUG=0): a = Tensor.rand((sz:=256), sz).realize()
    opts = [Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.LOCAL, 1, 4)]
    helper_lds_allclose(a.transpose().contiguous(), a.numpy().T, opts, [(0,128), (1,128)])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "tests require locals")
  def test_lds_reduce_sum(self):
    with Context(DEBUG=0): a = Tensor.rand((sz:=256), sz).realize()
    opts = [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.LOCAL, 0, 4)]
    helper_lds_allclose(a.sum(axis=1), a.numpy().sum(axis=1), opts, [(0,128), (1,128)], rtol=1e-4, atol=1e-4)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "tests require locals")
  def test_lds_elementwise_broadcast(self):
    with Context(DEBUG=0):
      a = Tensor.rand((sz:=256), sz).realize()
      b = Tensor.rand(sz, 1).realize()
    opts = [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.LOCAL, 1, 8), Opt(OptOps.LOCAL, 0, 4)]
    # b is broadcasted so local buffer shape only depends on locals/upcasts to dim 0
    helper_lds_allclose(a + b, a.numpy() + b.numpy(), opts, [(0,128),(1,128),(2,16)], rtol=1e-4, atol=1e-4)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "tests require locals")
  @unittest.skipIf(CI and Device.DEFAULT in {"AMD", "NV", "CUDA"}, "CI is really slow here")
  @unittest.skipIf(getenv("MOCKGPU", 0), "MOCKGPU is really slow here")
  def test_lds_conv2d(self):
    BS = 8
    CIN, COUT, HW = 32, 32, 32
    K = 3
    with Context(DEBUG=0):
      a = Tensor.rand(BS, CIN, HW, HW).realize()
      b = Tensor.rand(COUT, CIN, K, K).realize()

    ta, tb = torch.from_numpy(a.numpy()).to("cpu"), torch.from_numpy(b.numpy()).to("cpu")
    tc = torch.conv2d(ta, tb).numpy()

    opts = [Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UPCAST, 2, 2), Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 4)]

    helper_lds_allclose(a.conv2d(b), tc, opts, [(0,32),(1,8),(2,4)], rtol=1e-4, atol=1e-4)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "tests require locals")
  @unittest.skipIf(CI and Device.DEFAULT in {"AMD", "NV", "CUDA"}, "CI is really slow here")
  @unittest.skipIf(getenv("MOCKGPU", 0), "MOCKGPU is really slow here")
  def test_lds_conv2d_variant(self):
    BS = 8
    CIN, COUT, HW = 32, 32, 32
    K = 3
    with Context(DEBUG=0):
      a = Tensor.rand(BS, CIN, HW, HW).realize()
      b = Tensor.rand(COUT, CIN, K, K).realize()

    ta, tb = torch.from_numpy(a.numpy()).to("cpu"), torch.from_numpy(b.numpy()).to("cpu")
    tc = torch.conv2d(ta, tb).numpy()

    opts = [Opt(OptOps.UPCAST, 0, 2),
            Opt(OptOps.LOCAL, 0, 2),
            Opt(OptOps.LOCAL, 1, 4),
            Opt(OptOps.LDS, 0, None),
            Opt(OptOps.LDS, 2, None)]

    helper_lds_allclose(a.conv2d(b), tc, opts, [(0,16),(2,4)], rtol=1e-4, atol=1e-4, apply_lds=False)

  def test_various_ops(self):
    with Context(DEBUG=0): a, b = Tensor.rand(4, 4).realize(), Tensor.rand(4, 4).realize()

    # basic arithmetic & broadcasting
    helper_lds_allclose(a + b, a.numpy() + b.numpy())
    helper_lds_allclose(a - b, a.numpy() - b.numpy())
    helper_lds_allclose(a * b, a.numpy() * b.numpy())
    helper_lds_allclose(a.div(b), a.numpy() / b.numpy())
    helper_lds_allclose(a.maximum(b), np.maximum(a.numpy(), b.numpy()))

    # reductions
    helper_lds_allclose(a.sum(), np.array(a.numpy().sum()))
    helper_lds_allclose(a.sum(axis=0), a.numpy().sum(axis=0))
    helper_lds_allclose(a.mean(axis=1), a.numpy().mean(axis=1))

    # unary / activation
    helper_lds_allclose(a.relu(), np.maximum(a.numpy(), 0))
    helper_lds_allclose(a.sigmoid(), 1 / (1 + np.exp(-a.numpy())))
    helper_lds_allclose(a.tanh(), np.tanh(a.numpy()))

    ap = a.abs() + 1e-3  # ensure positivity
    helper_lds_allclose(ap.sqrt(), np.sqrt(ap.numpy()))
    helper_lds_allclose(ap.log(), np.log(ap.numpy()))
    helper_lds_allclose(ap.exp(), np.exp(ap.numpy()))

    # logical / comparison
    helper_lds_allclose(a.isfinite(), np.isfinite(a.numpy()))
    helper_lds_allclose(a.isnan(), np.isnan(a.numpy()))
    helper_lds_allclose(a.isinf(), np.isinf(a.numpy()))

if __name__ == '__main__':
  unittest.main()
