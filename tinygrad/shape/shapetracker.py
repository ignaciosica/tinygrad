# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
from dataclasses import dataclass
import functools, colorsys
from typing import Optional, Callable
from tinygrad.helpers import merge_dicts, getenv
from tinygrad.shape.view import View, strides_for_shape, unravel
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, Variable, sint, sint_to_uop, Context, PatternMatcher, UPat, GroupOp
from tinygrad.codegen.symbolic import split_uop, symbolic_flat, uop_given_valid, simplify_valid

# If a node overflow, its srcs need to be checked to see if this overflow is the result of an ALU operation,
# or that the node simply inherits the dtype from srcs. Upcast is either `Ops.CAST`+`replace` or just `replace`.
def handle_upcast(u: UOp) -> UOp|None:
  dtype = dtypes.int64.vec(u.dtype.count) if u.dtype.count > 1 else dtypes.int64
  # check for overflow, upcast this to int64
  if u.vmax > dtypes.max(dtypes.int) or u.vmin < dtypes.min(dtypes.int):
    return u.replace(dtype=dtype, src=tuple([x.cast(dtype) for x in u.src]))
  # if any inputs are int64 and this *doesn't* overflow, cast back to int
  if any(x.dtype == dtypes.int64 for x in u.src):
    return u.replace(dtype=dtype, src=tuple([x.cast(dtype) for x in u.src])).cast(u.dtype)
  return None
pm_upcast = PatternMatcher([(UPat(GroupOp.ALU, dtype=dtypes.int, name="u"), handle_upcast),])

@functools.cache
def views_to_indexed_uops(views: tuple[View, ...], _idxs:Optional[tuple[UOp, ...]]=None) -> tuple[UOp, UOp]:
  idx, valid = views[-1].to_indexed_uops(_idxs)
  for view in reversed(views[0:-1]):
    view = view.minify()
    idx, valid = view.to_indexed_uops([sint_to_uop(i) for i in unravel(view.shape, idx)], valid)
  # symbolic
  idx, valid = graph_rewrite(UOp.sink(idx, valid), symbolic_flat, name="indexing sym @ 1").src
  # simplify
  if (newvalid:=simplify_valid(valid)) is not None: valid = newvalid
  if (newidx:=uop_given_valid(valid, idx)) is not None: idx = newidx
  # symbolic again, upcast if needed
  return graph_rewrite(UOp.sink(idx, valid), symbolic_flat+pm_upcast, name="indexing sym @ 2").src

@functools.cache
def views_to_real_strides(views: tuple[View, ...], ignore_valid=False) -> tuple[Optional[sint], ...]:
  # NOTE: if a stride is not always valid, it will be None
  if len(views) == 1 and views[-1].mask is None: return views[-1].strides
  ret: list[Optional[sint]] = [None] * len(views[-1].shape)
  idx, valid = views_to_indexed_uops(views)
  for c in split_uop(idx, Ops.ADD):
    if c.op is Ops.RANGE: ret[c.arg] = 1
    if c.op is Ops.MUL and c.src[0].op is Ops.RANGE and c.src[1].op is Ops.CONST: ret[c.src[0].arg] = c.src[1].arg
    if c.op is Ops.MUL and c.src[1].op is Ops.RANGE and c.src[0].op is Ops.CONST: ret[c.src[1].arg] = c.src[0].arg
  used_ranges = [x.arg for x in idx.toposort() if x.op is Ops.RANGE]
  ret = [x if i in used_ranges else 0 for i,x in enumerate(ret)]
  if not ignore_valid:
    for masked_axis in [x.arg for x in valid.toposort() if x.op is Ops.RANGE]: ret[masked_axis] = None
  return tuple(ret)

@dataclass(frozen=True, order=True)
class ShapeTracker:
  views: tuple[View, ...]

  def __add__(self, st:ShapeTracker) -> ShapeTracker:
    ret = self
    for v in st.views: ret = ShapeTracker(ret.views + (v,)).simplify() # one view at a time = better simplification
    return ret

  def invert(self, out_shape:tuple[sint, ...]) -> Optional[ShapeTracker]:
    inverted_views:list[View] = []
    for v,s in zip(self.views[::-1], [x.shape for x in self.views[::-1][1:]]+[out_shape]):
      if (inverted:= v.invert(s)) is None: return None
      inverted_views.append(inverted)
    return ShapeTracker(tuple(inverted_views)).reshape(out_shape)

  @staticmethod
  def from_shape(shape:tuple[sint, ...], strides:tuple[sint, ...]|None=None) -> ShapeTracker: return ShapeTracker((View.create(shape, strides),))

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def consecutive(self) -> bool: return len(self.views) == 1 and (v:=self.views[0]).mask is None and v.strides == strides_for_shape(v.shape)

  @property
  def shape(self) -> tuple[sint, ...]: return self.views[-1].shape

  @property
  def size(self) -> int: return self.views[-1].size()

  def reduce(self, axis:tuple[int, ...]) -> tuple[sint, ...]: return tuple(1 if i in axis else s for i,s in enumerate(self.shape))

  def to_uop(self) -> UOp: return UOp(Ops.VIEW, dtypes.void, (), self)
  def to_indexed_uops(self, _idxs:Optional[list[UOp]|tuple[UOp, ...]]=None) -> tuple[UOp, UOp]:
    return views_to_indexed_uops(self.views, tuple(_idxs) if _idxs is not None else None)

  # upper bound on buffer size required to fit this shapetracker
  def real_size(self) -> int:
    if 0 in self.shape: return 0
    view = (v.shrink(v.mask) if (v:=self.views[0]).mask else v)
    idx, _ = views_to_indexed_uops((view,))
    assert idx.vmax < 1e12, f"real_size broken for {self}"
    return int(idx.vmax + 1)

  def vars(self) -> set[Variable]: return set().union(*[v.vars() for v in self.views])

  @property
  def var_vals(self) -> dict[Variable, int]: return merge_dicts([dict([v.unbind()]) for v in self.vars()])

  def unbind(self) -> tuple[ShapeTracker, dict[Variable, int]]:
    unbound_views, var_vals = zip(*[v.unbind() for v in self.views])
    if all(len(x) == 0 for x in var_vals): return self, {}
    return ShapeTracker(tuple(unbound_views)), merge_dicts(var_vals)
  def substitute(self, dvars:dict[UOp, UOp]): return ShapeTracker(tuple(x.substitute(dvars) for x in self.views))

  def real_strides(self, ignore_valid=False) -> tuple[Optional[sint], ...]:
    with Context(TRACK_MATCH_STATS=0): return views_to_real_strides(self.views, ignore_valid)
  def unit_stride_axes(self, ignore_valid=False) -> list[int]: return [i for i,st in enumerate(self.real_strides(ignore_valid)) if st == 1]

  def axis_is_masked(self, axis:int) -> bool:
    with Context(TRACK_MATCH_STATS=0):
      _, valid = self.to_indexed_uops()
      return axis in [x.arg for x in graph_rewrite(valid, symbolic_flat).toposort() if x.op is Ops.RANGE]

  def simplify(self) -> ShapeTracker:
    if len(self.views) >= 2 and (new_view := self.views[-2] + self.views[-1]) is not None:
      return ShapeTracker(self.views[:-2] + (new_view,)).simplify()
    return self

  # *** under this line are the movement ops ***

  def pad(self, arg: tuple[tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].pad(arg), ))
  def shrink(self, arg: tuple[tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].shrink(arg), ))
  def expand(self, new_shape: tuple[sint, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].expand(new_shape), ))
  def permute(self, axis: tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].permute(axis), ))
  def flip(self, mul: tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].flip(mul), ))

  def reshape(self, new_shape: tuple[sint, ...]) -> ShapeTracker:
    if getenv("MERGE_VIEW", 1) and (new_view := self.views[-1].reshape(new_shape)) is not None: return ShapeTracker(self.views[0:-1] + (new_view,))
    return ShapeTracker(self.views + (View.create(new_shape), ))

  def mop(self, op, arg): return mops[op](self, arg)

  def viz(self):
    from tabulate import tabulate
    layout: dict = {}
    with Context(TRACK_MATCH_STATS=0):
      for i in range(0, self.size):
        logical_coords: tuple[UOp, ...] = tuple(sint_to_uop(c) for c in unravel(self.shape, i))
        idx, idx_valid = self.to_indexed_uops(logical_coords)
        if idx_valid.arg: layout.setdefault(idx.arg, []).append(i)

    matrix, elems, width, w, tidx = None, [], 1, len(str(self.size - 1)), getenv("VIZ_TILE_TIDX", -1)
    def ansi(t: int) -> str:
      _R, _G, _B = (int(x * 5 + 0.5) for x in colorsys.hsv_to_rgb(t / (self.size - 1), 0.65, 0.80))
      return f"\x1b[38;5;{17 + 36 * _R + 6 * _G + _B}m{t:0{w}d}\x1b[0m" if tidx == -1 or tidx == t else f"{t:0{w}d}"

    for i in range(max(layout.keys()) + 1):
      if i not in layout: elems += [""]
      else:
        idxs = tuple(sorted(set(idx for idx in layout[i])))
        elems += [f"{','.join((f'{chr(10)}' if i > 0 and i % 4 == 0 else '') + ansi(idx) for i,idx in enumerate(idxs))}"]

    for stride, shape in sorted((stride, shape) for stride, shape in zip(self.real_strides(True), self.shape) if stride != 0):
      if width == stride and width * shape <= getenv("VIZ_TILE_MAX_WIDTH", 32): width *= shape
      else: break
    if len(elems) % width != 0: width = 1 # fallback to width 1 for some cases of padding

    matrix = [elems[i : i + width] for i in range(0, len(elems), width)]
    if matrix: print(tabulate(matrix, tablefmt="simple_grid", showindex=True, headers=tuple(f"{i:02d}" for i in range(width))))
    else: print("<< failed to viz tile >>")

mops: dict[Ops, Callable] = {Ops.RESHAPE: ShapeTracker.reshape, Ops.PERMUTE: ShapeTracker.permute, Ops.EXPAND: ShapeTracker.expand,
                             Ops.SHRINK: ShapeTracker.shrink, Ops.FLIP: ShapeTracker.flip, Ops.PAD: ShapeTracker.pad}
