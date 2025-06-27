# opt opinionatedly transforms an ast into an optimized ast using either heuristics or beam search

from tinygrad.opt.kernel import Kernel
from tinygrad.opt.heuristic import hand_coded_optimizations
from tinygrad.uop.ops import UOp
from tinygrad.helpers import NOOPT, BEAM, USE_TC, getenv
from tinygrad.renderer import Renderer

def get_optimized_ast(ast:UOp, renderer:Renderer) -> UOp:
  """
  Optimize an AST based on heuristics or BEAM search.

  Args:
    ast: The Ops.SINK rooted AST
    renderer: The renderer used to generate the code

  Returns:
    The Ops.SINK rooted AST transformed to apply the opts and with a KernelInfo in the arg.
  """

  k = Kernel(ast, opts=renderer)
  if ast.arg is not None and ast.arg.opts_to_apply is not None: k.apply_opts(ast.arg.opts_to_apply)
  elif not NOOPT:
    if not k.apply_tensor_cores(USE_TC.value): k.apply_opts(hand_coded_optimizations(k))
    if getenv("SMEM", 1):
      from tinygrad.opt.kernel import Opt, OptOps
      for buf in range(8):
        try:
          k.apply_opt(Opt(OptOps.PROMOTE_SMEM, buf, None))
        except Exception as e: print(e)
    if BEAM >= 1:
      from tinygrad.opt.search import beam_search, bufs_from_lin
      kb = Kernel(ast, opts=renderer)
      rawbufs = bufs_from_lin(kb, allocate=False)
      k = beam_search(kb, rawbufs, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
  return k.get_optimized_ast()
