# sudo -E env "PATH=$PATH" CNT=1 DISABLE_COMPILER_CACHE=1 ncu --set full --import-source yes --target-processes all --call-stack -o profile -f python extra/gemm/cuda_gemm.py

import numpy as np
import statistics
from tinygrad import dtypes
from tinygrad.helpers import getenv, flat_mv
from tinygrad.dtype import _to_np_dtype
from tinygrad.runtime.ops_cuda import CUDAAllocator, CUDADevice, CUDAProgram, CUDACompiler

dtype_in = dtypes.half if getenv("HALF") else dtypes.float
acc_dtype = dtypes.half if getenv("ACC_HALF") else None

N = getenv("N", 4096)
CNT = getenv("CNT", 10)
ATOL = getenv("ATOL", 1e-4)
RTOL = getenv("RTOL", 3e-2)

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
  src = \
"""
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
extern "C" __global__ void __launch_bounds__(128) r_128_64_8_16_1024_4_4_4(float* data0, float* data1, float* data2) {
  int gidx0 = blockIdx.x; /* 64 */
  int gidx1 = blockIdx.y; /* 128 */
  int lidx0 = threadIdx.x; /* 8 */
  int lidx1 = threadIdx.y; /* 16 */
  int alu0 = (lidx1<<2);
  int alu1 = (gidx0<<6);
  int alu2 = (lidx0<<14);
  int alu3 = (gidx1<<17);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  float acc4 = 0.0f;
  float acc5 = 0.0f;
  float acc6 = 0.0f;
  float acc7 = 0.0f;
  float acc8 = 0.0f;
  float acc9 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;
  float acc12 = 0.0f;
  float acc13 = 0.0f;
  float acc14 = 0.0f;
  float acc15 = 0.0f;
  for (int ridx4 = 0; ridx4 < 1024; ridx4++) {
    int alu4 = (alu3+alu2+(ridx4<<2));
    float4 val0 = *((float4*)((data1+alu4)));
    float4 val1 = *((float4*)((data1+(alu4+4096))));
    float4 val2 = *((float4*)((data1+(alu4+8192))));
    float4 val3 = *((float4*)((data1+(alu4+12288))));
    int alu5 = (alu1+alu0+(ridx4<<14));
    float4 val4 = *((float4*)((data2+alu5)));
    float4 val5 = *((float4*)((data2+(alu5+4096))));
    float4 val6 = *((float4*)((data2+(alu5+8192))));
    float4 val7 = *((float4*)((data2+(alu5+12288))));
    acc0 = (acc0+(val0.x*val4.x)+(val0.y*val5.x)+(val0.z*val6.x)+(val0.w*val7.x));
    acc1 = (acc1+(val1.x*val4.x)+(val1.y*val5.x)+(val1.z*val6.x)+(val1.w*val7.x));
    acc2 = (acc2+(val2.x*val4.x)+(val2.y*val5.x)+(val2.z*val6.x)+(val2.w*val7.x));
    acc3 = (acc3+(val3.x*val4.x)+(val3.y*val5.x)+(val3.z*val6.x)+(val3.w*val7.x));
    acc4 = (acc4+(val0.x*val4.y)+(val0.y*val5.y)+(val0.z*val6.y)+(val0.w*val7.y));
    acc5 = (acc5+(val1.x*val4.y)+(val1.y*val5.y)+(val1.z*val6.y)+(val1.w*val7.y));
    acc6 = (acc6+(val2.x*val4.y)+(val2.y*val5.y)+(val2.z*val6.y)+(val2.w*val7.y));
    acc7 = (acc7+(val3.x*val4.y)+(val3.y*val5.y)+(val3.z*val6.y)+(val3.w*val7.y));
    acc8 = (acc8+(val0.x*val4.z)+(val0.y*val5.z)+(val0.z*val6.z)+(val0.w*val7.z));
    acc9 = (acc9+(val1.x*val4.z)+(val1.y*val5.z)+(val1.z*val6.z)+(val1.w*val7.z));
    acc10 = (acc10+(val2.x*val4.z)+(val2.y*val5.z)+(val2.z*val6.z)+(val2.w*val7.z));
    acc11 = (acc11+(val3.x*val4.z)+(val3.y*val5.z)+(val3.z*val6.z)+(val3.w*val7.z));
    acc12 = (acc12+(val0.x*val4.w)+(val0.y*val5.w)+(val0.z*val6.w)+(val0.w*val7.w));
    acc13 = (acc13+(val1.x*val4.w)+(val1.y*val5.w)+(val1.z*val6.w)+(val1.w*val7.w));
    acc14 = (acc14+(val2.x*val4.w)+(val2.y*val5.w)+(val2.z*val6.w)+(val2.w*val7.w));
    acc15 = (acc15+(val3.x*val4.w)+(val3.y*val5.w)+(val3.z*val6.w)+(val3.w*val7.w));
  }
  int alu23 = (alu1+alu3+alu2+alu0);
  *((float4*)((data0+alu23))) = make_float4(acc0,acc4,acc8,acc12);
  *((float4*)((data0+(alu23+4096)))) = make_float4(acc1,acc5,acc9,acc13);
  *((float4*)((data0+(alu23+8192)))) = make_float4(acc2,acc6,acc10,acc14);
  *((float4*)((data0+(alu23+12288)))) = make_float4(acc3,acc7,acc11,acc15);
}
"""
  program = CUDAProgram(device, "r_128_64_8_16_1024_4_4_4", compiler.compile(src))

  FLOPS = N*N*N*2
  BW = N*N*3*4

  global_size, local_size = (64, 128, 1), (8, 16, 1)
  tm = statistics.mean([program(c, a, b, global_size=global_size, local_size=local_size, wait=True) for _ in range(CNT)])
  print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s")
  cudaalloc._copyout(flat_mv(nc.data), c)
  np.testing.assert_allclose(nc.reshape(N,N), na.astype(np.float32) @ nb.astype(np.float32), rtol=RTOL, atol=ATOL)