# sudo -E env "PATH=$PATH" CNT=1 DISABLE_COMPILER_CACHE=1 ncu --set full --import-source yes --target-processes all --call-stack -o profile -f python extra/gemm/cuda_gemm.py

import pathlib, statistics, numpy as np
from tinygrad import dtypes
from tinygrad.helpers import getenv, flat_mv
from tinygrad.dtype import _to_np_dtype
from tinygrad.runtime.ops_cuda import CUDAAllocator, CUDADevice, CUDAProgram, CUDACompiler

dtype_in = dtypes.half if getenv("HALF") else dtypes.float
acc_dtype = dtypes.half if getenv("ACC_HALF") else dtypes.float

N = getenv("N", 4096)
CNT = getenv("CNT", 10)
ATOL = getenv("ATOL", 1e-4)
RTOL = getenv("RTOL", 3e-2)
KERNEL = getenv("KERNEL", 0)

if __name__ == "__main__":
  rng = np.random.default_rng()
  na = rng.random((N, N), dtype=np.float32).astype(_to_np_dtype(dtype_in))-0.5
  nb = rng.random((N, N), dtype=np.float32).astype(_to_np_dtype(dtype_in))-0.5
  nc = np.empty(N*N, np.float32)

  device = CUDADevice("cuda:0")
  cudaalloc = CUDAAllocator(device)

  a = cudaalloc.alloc(N*N*dtype_in.itemsize)
  b = cudaalloc.alloc(N*N*dtype_in.itemsize)
  c = cudaalloc.alloc(N*N*(acc_dtype or dtype_in).itemsize)

  cudaalloc._copyin(a, bytearray(na))
  cudaalloc._copyin(b, bytearray(nb))

  compiler = CUDACompiler(device.arch)
  kernels = [
    ("temp", "r_128_64_8_16_1024_4_4_4", "r_128_64_8_16_1024_4_4_4", (64, 128, 1), (8, 16, 1), ()),
    ("max_kernels", "nv.fp16_fp32_fp32.max", "wmma_example", (64, 32, 1), (128, 1, 1), (4096, 4096)),
    ]
  path = pathlib.Path(__file__).parent / kernels[KERNEL][0] / f"{kernels[KERNEL][1]}.cu"
  with open(path, "r", encoding="utf-8") as f: src = f.read()
  program = CUDAProgram(device, kernels[KERNEL][2], compiler.compile(src))

  FLOPS = N*N*N*2
  BW = N*N*3*4

  tm = statistics.mean([program(c, a, b, global_size=kernels[KERNEL][3], local_size=kernels[KERNEL][4], wait=True, vals=kernels[KERNEL][5]) for _ in range(CNT)])
  print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s")
  cudaalloc._copyout(flat_mv(nc.data), c)
  np.testing.assert_allclose(nc.reshape(N,N), na.astype(np.float32) @ nb.astype(np.float32), rtol=RTOL, atol=ATOL)