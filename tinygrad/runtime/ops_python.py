# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Optional, Any, TYPE_CHECKING, Literal
from dataclasses import dataclass, replace, field
import pickle, base64, itertools, time, struct, sys, os
from tinygrad.dtype import DType, dtypes, ImageDType, PtrDType, truncate
from tinygrad.helpers import all_same, getenv, flatten, get_single_element
from tinygrad.device import Compiled, Compiler, Allocator
from tinygrad.uop.ops import exec_alu, Ops, UOp, GroupOp
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer, MetalRenderer, AMDRenderer, IntelRenderer, ClangRenderer

# --- Configuration ---
# Set the buffer width for visualization from environment variables, defaulting to 8.
BUF_WIDTH = int(os.getenv("N", 8))

# Global dictionary to hold the visualization state for different buffers.
# The key is the buffer ID, and the value is a list of CellState objects.
viz_bufs: dict[int, list["CellState"]] = {}


# --- Data Structure for Cell State ---


@dataclass(frozen=True, kw_only=True)
class CellState:
  """
  Represents the state of a single memory cell for visualization.

  Attributes:
      action: The last action performed ('read' or 'write').
      threads: A list of thread IDs that have accessed this cell.
               Resets on a write, appends on a read.
      vec: The vector index associated with a write action.
      count: The number of consecutive actions of the same type.
             Resets on a write or when a read follows a write.
  """

  action: Optional[Literal["read", "write"]] = None
  threads: list[int] = field(default_factory=list)
  vec: int = 0
  count: int = 0

  def _wrap_colour(self, text: str) -> str:
    """Applies ANSI color codes to the text based on the cell's state."""
    if not self.action or self.count == 0:
      return text

    # Green for "read", Red for "write"
    base_code = 32 if self.action == "read" else 31

    # Intensity increases with the access count
    if self.count <= 3:
      code, style = base_code, "0;"  # Normal
    elif self.count <= 6:
      code, style = base_code + 60, "0;"  # Bright
    else:
      code, style = base_code + 60, "1;"  # Bright and Bold

    return f"\033[{style}{code}m{text}\033[0m"

  def __str__(self) -> str:
    """Returns a string representation of the cell for printing."""
    if not self.threads:
      return " " * 16  # Return a fixed-width empty string for alignment

    # Format the list of threads, e.g., "t[01,02]"
    thread_part = f"t[{','.join(f'{th:02d}' for th in self.threads)}]"
    vec_part = f"v[{self.vec}]"
    count_part = f"({self.count:02d})"

    # Combine parts and pad to ensure consistent column width
    full_str = f"{thread_part}{vec_part}{count_part}"
    return self._wrap_colour(f"{full_str:<16}")


# --- Visualization Functions ---


def print_viz_bufs():
  """Prints all visualization buffers in reverse order of their ID."""
  # Iterating over a reversed list of items to avoid issues with dict size changes
  for buf_id in reversed(list(viz_bufs.keys())):
    print(f"\nbuf[{buf_id}]")
    print_viz_buf(buf_id)
  time.sleep(0.05)


def print_viz_buf(buf_id: int):
  """Prints a single buffer, formatted into rows based on BUF_WIDTH."""
  viz_buf = viz_bufs.get(buf_id, [])
  for i in range(0, len(viz_buf), BUF_WIDTH):
    print(" | ".join(map(str, viz_buf[i : i + BUF_WIDTH])))


# --- Memory Access Functions ---


def _load(m: tuple, i: Optional[int], t: int = 0) -> float:
  """
  Helper function to load a value from a memory buffer and update its visualization state for a 'read'.
  A 'read' action appends the thread ID to the list. If a read follows a write,
  the thread list and count are reset.
  """
  if i is None:
    return 0.0
  if not 0 <= i < len(m[0]):
    raise IndexError(f"Load out of bounds: size is {len(m[0])}, access is {i}")

  buf_id = m[1]
  current_state = viz_bufs[buf_id][i]

  # If the last action was not a read, start a new sequence of readers.
  is_new_read_sequence = current_state.action != "read"

  # Start a new thread list if it's a new read sequence, otherwise get the current list.
  new_threads = [] if is_new_read_sequence else list(current_state.threads)

  # Add the current thread ID if it's not already in the list.
  if t not in new_threads:
    new_threads.append(t)

  # Reset the count for a new read sequence, otherwise increment.
  new_count = 1 if is_new_read_sequence else current_state.count + 1

  viz_bufs[buf_id][i] = replace(
    current_state,
    action="read",
    threads=sorted(new_threads),  # Keep threads sorted for consistent display
    count=new_count,
  )
  return m[0][i]


def _store(m: tuple, i: int, v: float, t: int = 0, j: int = 0):
  """
  Helper function to store a value into a memory buffer and update its visualization state for a 'write'.
  A 'write' action resets the cell's state, including the thread list and count.
  """
  if not 0 <= i < len(m[0]):
    raise IndexError(f"Store out of bounds: size is {len(m[0])}, access is {i}")

  buf_id = m[1]
  # A write resets the cell's history. The thread list is reset to the current
  # writer, and the count is reset to 1.
  viz_bufs[buf_id][i] = CellState(action="write", threads=[t], vec=j, count=1)
  m[0][i] = v


def load(inp: list, j: int = 0) -> list:
  """
  Dispatcher function for loading values from memory.
  It has two modes based on the structure of the 'inp' argument:
  1. Gated Load (if len(inp) == 2): Processes inputs with an associated gate flag.
  2. Threaded Load (otherwise): Processes inputs, assigning a unique thread ID to each.
  """
  # Mode 1: Gated load. inp format: [((m,x,gate),...), (default,...)]
  if len(inp) == 2:
    return [_load(m, x + j if x is not None else None) if gate else default for (m, x, gate), default in zip(*inp)]
  # Mode 2: Threaded load. inp format: [((m,x,_),...),]
  return [_load(m, x + j if x is not None else None, t=t) for t, (m, x, _) in enumerate(inp[0])]

class PythonProgram:
  def __init__(self, name:str, lib:bytes):
    self.uops: list[tuple[Ops, Optional[DType], list[int], Any]] = pickle.loads(lib)
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    time.sleep(1)
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      # print(f"\n{idxs}")
      # NOTE: block level, print at this stage to show parallelism across the threadblock
      ul: dict[int, Any] = {}
      dl: dict[int, DType] = {}
      pbufs: list[memoryview] = list(bufs)
      pvals: list[int] = list(vals)
      mem: int = 0
      i = 0
      loop_ends: dict[int, int] = {}
      while i < len(self.uops):
        uop, dtype, idp, arg = self.uops[i]
        void_ops = {Ops.STORE, Ops.ENDRANGE, Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK}
        if uop is Ops.DEFINE_REG: idp = [idp[0]]
        inp = [ul[v] for v in idp if self.uops[v][0] not in void_ops]
        dtp = [dl[v] for v in idp if self.uops[v][0] not in void_ops]
        if getenv("TRACE"): print(i, uop, dtype, arg, inp, dtp)
        if uop is Ops.STORE:
          assert len(inp) == 2, "expected store is ([(memory, offset, gate)], [value])"
          if dtp[1].count > 1:
            for j,val in enumerate(inp[1]):
              for t,((m,o,g),v) in enumerate(zip(inp[0], val)):
                if g: _store(m, o+j, v, t=t, j=j)
            print_viz_bufs()
          else:
            for t,((m,o,g),v) in enumerate(zip(*inp)):
              if g: _store(m, o, v, t=t, j=0)
            print_viz_bufs()
          i += 1
          continue
        if uop is Ops.ENDRANGE:
          loop_ends[idp[0]] = i
          i = idp[0]
          continue
        if uop in (Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK):
          # in the python emulator, the warp is always in sync
          i += 1
          continue
        assert dtype is not None, f"{uop} is missing a dtype"
        dl[i] = dtype
        if uop in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL}:
          assert dtype.fmt is not None and isinstance(dtype, PtrDType)
          if TYPE_CHECKING or sys.version_info < (3, 12): assert dtype.fmt != "e"
          buf = memoryview(bytearray(dtype.size*dtype.itemsize)) if uop is Ops.DEFINE_LOCAL else pbufs.pop(0)
          buf_casted = buf.cast(dtype.fmt)
          if arg not in viz_bufs: viz_bufs[arg] = [CellState() for _ in range(len(buf_casted))]
          ul[i] = [(buf_casted, arg)] * warp_size
        elif uop is Ops.DEFINE_VAR:
          ul[i] = [pvals.pop(0)] * warp_size
        elif uop is Ops.SPECIAL:
          if arg[0][0] == 'g': ul[i] = [idxs[2-int(arg[0][-1])]] * warp_size
          elif arg[0][0] == 'l': ul[i] = [x[2-int(arg[0][-1])] for x in warp]
        elif uop is Ops.CONST: ul[i] = [arg] * warp_size
        elif uop is Ops.DEFINE_REG:
          ul[i] = [[inp[0][0][0]] * warp_size for _ in range(dtype.count)] if dtype.count > 1 else [inp[0][0]] * warp_size
        elif uop is Ops.INDEX:
          ret:list = []
          if isinstance(dtp[0], ImageDType):
            for m,ox,oy in zip(inp[0], inp[1][0], inp[1][1]):
              if ox < 0 or ox >= dtp[0].shape[1] or oy < 0 or oy >= dtp[0].shape[0]: ret.append((m, None))
              else: ret.append((m, ox*4 + oy*dtp[0].shape[1]*4))
          else:
            for m,o in zip(inp[0], inp[1]): ret.append((m,o))
          ul[i] = [(m,o,g) for (m,o),g in zip(ret, inp[2] if len(inp) == 3 else [True]*len(ret))] # set the gate last
        elif uop is Ops.CAST and isinstance(dtype, PtrDType):
          ul[i] = inp[0]
        elif uop is Ops.RANGE:
          if i not in ul: ul[i] = [0] * warp_size
          else:
            for j in range(len(ul[i])):
              ul[i][j] += 1
            if ul[i][0] == inp[0][0]:
              del ul[i]
              i = loop_ends[i] + 1
              continue
        elif uop is Ops.VECTORIZE: ul[i] = inp
        elif uop is Ops.BITCAST:
          assert dtp[0].fmt and dtype.fmt
          pack_format, unpack_format = str(warp_size) + dtp[0].fmt, str(warp_size) + dtype.fmt
          ul[i] = list(struct.unpack(unpack_format, struct.pack(pack_format, *inp[0])))
        elif uop is Ops.CAST:
          ul[i] = [truncate.get(dtype, lambda dt: dt)(dtypes.as_const(x, dtype)) for x in inp[0]]
        elif uop is Ops.LOAD:
          if dtype.count > 1:
            ul[i] = [load([inp[i][j] if i != 0 and dtp[i].count > 1 else inp[i] for i in range(len(inp))], j) for j in range(dtype.count)]
            # print_viz_buf(2)
            print_viz_bufs()
            # print("-lv-")
          else:
            ul[i] = load(inp)
            # print_viz_buf(2)
            print_viz_bufs()
            # print("-ls-")
        elif uop is Ops.ASSIGN:
          for j in range(len(inp[0])): inp[0][j] = inp[1][j]
          ul[i] = inp[0]
        elif uop is Ops.GEP: ul[i] = inp[0][get_single_element(arg)]
        elif uop is Ops.WMMA:
          # here are the models for the WMMA instruction on the different hardware
          def wmma_helper(WARP_THREADS, K, NUM_A, NUM_B, NUM_C, a_elem, b_elem, c_map):
            for cc, tinp, num in zip(("A", "B", "C"), inp, (NUM_A, NUM_B, NUM_C)):
              assert len(tinp) == num, f"{cc} must have {num} elements per thread, it has {len(tinp)}"
              assert len(flatten(tinp)) == num * warp_size, f"WMMA must have {num * warp_size} total elements for {cc} in WMMA"
            assert warp_size > 0 and warp_size % WARP_THREADS == 0, f"must have multiples of {WARP_THREADS} warp threads"
            out = [inp[2][elem_idx][:] for elem_idx in range(NUM_C)]
            for goff in range(0, warp_size, WARP_THREADS):
              for lane_id in range(WARP_THREADS):
                for elem_idx in range(NUM_C): # calculate new muls and add to acc
                  (c_i, c_j) = c_map(lane_id, elem_idx)
                  out[elem_idx][goff+lane_id] += sum(a_elem(inp[0], _k, c_j, goff) * b_elem(inp[1], c_i, _k, goff) for _k in range(K))
            return out

          # TODO: refactor these to a shared TensorCoreLayout in kernel.py
          if arg[4] == "METAL":
            # A (2 elements on 32 threads): row major
            def a_b_elem(x, i, j, goff): return x[(i%2)][goff+(i//2)%2+(j%4)*2+(i//4)*8+(j//4)*16]
            # (i, j), C, D (2 elements on 32 threads): row major same as A/B
            def c_map(lane, elem): return (elem + ((lane%2)*2) + ((lane//8)%2)*4, ((lane//2)%4) + (lane//16)*4)
            ul[i] = wmma_helper(32, 8, 2, 2, 2, a_b_elem, a_b_elem, c_map)
          elif arg[4] == "AMD" and arg[5] == 64:
            def a_elem(x, k, row, goff): return x[k%4][goff + (k//4)*16 + row]
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff) # pylint: disable=arguments-out-of-order
            def c_map(lane, elem): return (lane%16, (lane//16)*4 + elem)
            ul[i] = wmma_helper(64, 16, 4, 4, 4, a_elem, b_elem, c_map)
          elif arg[4] == "AMD" and len(inp[0]) == 8: # RDNA4
            def a_elem(x, k, row, goff): return x[k - [0, 4, 4, 8][k//4]][goff + row + [0, 16, 0, 16][k//4]]
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)
            def c_map(lane, elem): return (lane%16, (lane//16)*8 + elem)
            ul[i] = wmma_helper(32, 16, 8, 8, 8, a_elem, b_elem, c_map)
          elif arg[4] == "AMD":
            # A (16 elements on 32 threads): col major, lane 16-32 == lane 0-15
            def a_elem(x, k, row, goff):
              assert x[k][goff+row] == x[k][goff+row+16], "warp elements not duplicated properly across lanes"
              return x[k][goff+row]
            # B (16 elements on 32 threads): row major, lane 16-32 == lane 0-15
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)  # pylint: disable=arguments-out-of-order
            def c_map(lane, elem): return (lane%16, lane//16+elem*2) # (i, j), C, D (8 elements on 32 threads): row major
            ul[i] = wmma_helper(32, 16, 16, 16, 8, a_elem, b_elem, c_map)
          elif arg[4] == "CUDA":
            # (col, row) given (lane, elem) for C & D (4 elements on 32 threads); shared by all tc shapes with M=16 N=8
            def c_map(lane, elem): return (elem%2 + (lane%4)*2, lane//4 + (elem//2)*8)

            if arg[1] == (8,16,16):
              def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2 + (k//8)*4][goff + (k//2)%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%2 + (k//8)*2][goff + (k//2)%4 + col*4]
              ul[i] = wmma_helper(32, 16, 8, 4, 4, a_elem, b_elem, c_map)

            elif arg[1] == (8,16,8) and arg[2] == dtypes.half:
              def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2][goff + k//2 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%2][goff + k//2 + col*4]
              ul[i] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

            elif arg[1] == (8,16,8) and arg[2] == dtypes.float:
              def a_elem(x, k, row, goff): return x[(k//4)*2 + row//8][goff + k%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k//4][goff + k%4 + col*4]
              ul[i] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

            else: raise NotImplementedError(f"unimplemented tensor core {arg}")
          elif arg[4] == "INTEL":
            # A (16 elements on 8 threads)
            def a_elem(x, k, row, goff): return x[k%2+row*2][goff+k//2]
            # B (16 elements on 8 threads)
            def b_elem(x, col, k, goff): return x[k][goff+col]
            # C, D (8 elements on 8 threads)
            def c_map(lane, elem): return (lane, elem)
            ul[i] = wmma_helper(8, 16, 16, 16, 8, a_elem, b_elem, c_map)
          elif arg[4] == "CPU":
            def elem(x, col, row, _): return x[col+row][0] # k is always 0
            def c_map(_, elem): return (elem%16, elem//16)
            ul[i] = wmma_helper(1, 1, 16, 16, 256, elem, elem, c_map)
          else: raise NotImplementedError(f"unimplemented tensor core {arg}")
        elif uop in GroupOp.ALU:
          assert all_same([len(x) for x in inp]), f"{[len(x) for x in inp]} doesn't match on {uop}"
          assert all_same([dtype] + dtp) or uop in {Ops.CMPNE, Ops.CMPLT, Ops.WHERE}, f"dtype mismatch on {uop}"
          ul[i] = [exec_alu(uop, dtype, p) for p in zip(*inp)]
        assert i in ul, (uop, dtype, idp, arg)
        i += 1
    time.sleep(4)
    return time.perf_counter() - st

class PythonRenderer(Renderer):
  device = "PYTHON"
  def __init__(self):
    if getenv("EMULATE_METAL"): self.device, self.tensor_cores = "METAL", MetalRenderer.tensor_cores
    if getenv("EMULATE_AMD"): self.device, self.tensor_cores = "AMD", AMDRenderer.tensor_cores
    if getenv("EMULATE_AMD_MFMA"): self.device, self.tensor_cores = "AMD", AMDRenderer.tensor_cores_mfma
    if getenv("EMULATE_AMD_RDNA4"): self.device, self.tensor_cores = "AMD", AMDRenderer.tensor_cores_rdna4
    if getenv("EMULATE_CUDA"): self.device, self.tensor_cores = "CUDA", CUDARenderer.tc_sm80
    if getenv("EMULATE_CUDA_SM75"): self.device, self.tensor_cores = "CUDA", CUDARenderer.tc_sm75
    if getenv("EMULATE_INTEL"): self.device, self.suffix, self.tensor_cores = "INTEL", "INTEL", IntelRenderer.tensor_cores
    if getenv("EMULATE_AMX"): self.device, self.tensor_cores = "CPU", ClangRenderer.tensor_cores

  def render(self, uops:list[UOp]) -> str:
    lops = [(u.op, u.dtype, [uops.index(v) for v in u.src], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()

class PythonCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class PythonAllocator(Allocator['PythonDevice']):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview): dest[:] = src
  def _copyout(self, dest:memoryview, src): dest[:] = src

class PythonDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, PythonAllocator(self), PythonRenderer(), PythonCompiler(), PythonProgram)
