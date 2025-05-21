import subprocess, hashlib, tempfile, ctypes, pathlib
from typing import Callable
from tinygrad.helpers import to_char_p_p, init_c_var, getenv
import tinygrad.runtime.autogen.nvrtc as nvrtc
from tinygrad.device import Compiler, CompileError

PTX, CUDA_PATH = getenv("PTX"), getenv("CUDA_PATH", "")  # PTX shouldn't be here, in fact, it shouldn't exist

def _get_bytes(arg, get_str, get_sz, check) -> bytes:
  sz = init_c_var(ctypes.c_size_t(), lambda x: check(get_sz(arg, ctypes.byref(x))))
  return ctypes.string_at(init_c_var(ctypes.create_string_buffer(sz.value), lambda x: check(get_str(arg, x))), size=sz.value)

def nvrtc_check(status, ctx=None):
  if status != 0:
    err_log = _get_bytes(ctx, nvrtc.nvrtcGetProgramLog, nvrtc.nvrtcGetProgramLogSize, lambda _: None).decode() if ctx else ""
    raise CompileError(f"Nvrtc Error {status}, {ctypes.string_at(nvrtc.nvrtcGetErrorString(status)).decode()}\n{err_log}")

def jitlink_check(status, ctx=None):
  if status != 0:
    err_log = _get_bytes(ctx, nvrtc.nvJitLinkGetErrorLog, nvrtc.nvJitLinkGetErrorLogSize, lambda _: None).decode() if ctx else ""
    raise CompileError(f"NvJitLink Error {status}, {nvrtc.nvJitLinkResult__enumvalues.get(status, 'Unknown')}\n{err_log}")

def cuda_disassemble(lib:bytes, arch:str):
  try:
    fn = (pathlib.Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
    with open(fn, "wb") as f: f.write(lib)
    subprocess.run(["ptxas", f"-arch={arch}", "-o", fn, fn], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # optional ptx -> sass
    print(subprocess.check_output(['nvdisasm', fn]).decode('utf-8'))
  except Exception as e: print("Failed to generate SASS", str(e), "Make sure your PATH contains ptxas/nvdisasm binary of compatible version.")

class CUDACompiler(Compiler):
  def __init__(self, arch:str, cache_key:str="cuda"):
    self.arch, self.compile_options = arch, [f'--gpu-architecture={arch}']
    self.compile_options += [f"-I{CUDA_PATH}/include"] if CUDA_PATH else ["-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include"]
    nvrtc_check(nvrtc.nvrtcVersion((nvrtcMajor := ctypes.c_int()), (nvrtcMinor := ctypes.c_int())))
    if (nvrtcMajor.value, nvrtcMinor.value) >= (12, 4): self.compile_options.append("--minimal")
    super().__init__(f"compile_{cache_key}_{self.arch}")
  def _compile_program(self, src:str, nvrtc_get_content:Callable, nvrtc_get_size:Callable) -> bytes:
    nvrtc_check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog := nvrtc.nvrtcProgram()), src.encode(), "<null>".encode(), 0, None, None))
    nvrtc_check(nvrtc.nvrtcCompileProgram(prog, len(self.compile_options), to_char_p_p([o.encode() for o in self.compile_options])), prog)
    data = _get_bytes(prog, nvrtc_get_content, nvrtc_get_size, nvrtc_check)
    nvrtc_check(nvrtc.nvrtcDestroyProgram(ctypes.byref(prog)))
    return data
  def compile(self, src:str) -> bytes: return self._compile_program(src, nvrtc.nvrtcGetPTX, nvrtc.nvrtcGetPTXSize)
  def disassemble(self, lib:bytes): cuda_disassemble(lib, self.arch)

class NVCompiler(CUDACompiler):
  def __init__(self, arch:str): super().__init__(arch, cache_key="nv")
  def compile(self, src:str) -> bytes: return self._compile_program(src, nvrtc.nvrtcGetCUBIN, nvrtc.nvrtcGetCUBINSize)

class PTXCompiler(Compiler):
  def __init__(self, arch:str, cache_key="ptx"):
    self.arch = arch
    super().__init__(f"compile_{cache_key}_{self.arch}")
  def compile(self, src:str) -> bytes: return src.replace("TARGET", self.arch).replace("VERSION", "7.8" if self.arch >= "sm_89" else "7.5").encode()
  def disassemble(self, lib:bytes): cuda_disassemble(lib, self.arch)

class NVPTXCompiler(PTXCompiler):
  def __init__(self, arch:str): super().__init__(arch, cache_key="nv_ptx")
  def compile(self, src:str) -> bytes:
    jitlink_check(nvrtc.nvJitLinkCreate(handle := nvrtc.nvJitLinkHandle(), 1, to_char_p_p([f'-arch={self.arch}'.encode()])), handle)
    jitlink_check(nvrtc.nvJitLinkAddData(handle, nvrtc.NVJITLINK_INPUT_PTX, ptxsrc:=super().compile(src), len(ptxsrc), "<null>".encode()), handle)
    jitlink_check(nvrtc.nvJitLinkComplete(handle), handle)
    data = _get_bytes(handle, nvrtc.nvJitLinkGetLinkedCubin, nvrtc.nvJitLinkGetLinkedCubinSize, jitlink_check)
    jitlink_check(nvrtc.nvJitLinkDestroy(handle))
    return data
