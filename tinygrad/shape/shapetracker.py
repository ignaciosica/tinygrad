# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
from dataclasses import dataclass
import functools
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

  def coord(self, index: int) -> int:
    """
    Given a flat logical index for the current shape of the ShapeTracker,
    returns the corresponding physical offset in the underlying buffer.
    """
    # Ensure the index is valid for the current logical shape.
    # If self.size is 0 (e.g., shape has a zero dimension), no index is valid.
    assert 0 <= index < self.size, f"Index {index} out of bounds for ShapeTracker with size {self.size}"

    # Unravel the flat logical index into multi-dimensional coordinates.
    # tinygrad.shape.view.unravel handles self.shape == () (scalar case) by returning ().
    logical_coords: tuple[sint, ...] = unravel(self.shape, index) # type: ignore
    # print(logical_coords)
    # Convert coordinates to UOp constants.
    # sint_to_uop creates UOp(Ops.CONST, ..., arg=value_of_c)
    coord_uops: tuple[UOp, ...] = tuple(sint_to_uop(c) for c in logical_coords)

    # Get the symbolic physical index and validity from the views.
    # self.to_indexed_uops calls the cached views_to_indexed_uops with these constant UOps.
    # All symbolic simplifications are applied within views_to_indexed_uops.
    final_idx_uop, final_valid_uop = self.to_indexed_uops(coord_uops)

    # After simplifications with constant inputs, the resulting index UOp should be a constant.
    if final_idx_uop.op != Ops.CONST:
      # This indicates an issue, e.g., unbound Variables in views affecting indexing,
      # or incomplete simplification. The offset must be a concrete integer.
      raise RuntimeError(f"Coordinate calculation for index {index} (logical coords {logical_coords}) "
                         f"did not simplify to a constant offset. Got: {final_idx_uop}")

    # The argument of the constant UOp is the integer offset.
    offset_val = final_idx_uop.arg

    # Ensure the argument is indeed an integer, as sint can be Variable.
    # This should hold if op is CONST and simplifications worked as expected.
    if not isinstance(offset_val, int):
      raise RuntimeError(f"Expected integer offset, but got {type(offset_val)}: {offset_val}. UOp: {final_idx_uop}")

    # Note on validity:
    # final_valid_uop indicates if this logical index maps to a valid (unmasked) physical location.
    # If final_valid_uop evaluates to CONST 0, the index points to a masked area (e.g., padding).
    # The current requirement is to "return the coord offset" regardless.
    # Example: if final_valid_uop.arg == 0, offset_val could be an address in a padding region.
    # One might add a warning or an error here if strict validity is required:
    if final_valid_uop.op == Ops.CONST and final_valid_uop.arg == 0:
      print(f"Warning: Index {index} (logical coords {logical_coords}) maps to a masked-out " \
            f"physical location (offset {offset_val}). View mask might be active.")

    return offset_val

  def with_shape_one_for_zero_strides(self) -> ShapeTracker:
    """
    Creates a new ShapeTracker based on the current one.
    In the new ShapeTracker, dimensions that had a stride of 0 in the original's
    effective strides (real_strides) will have their shape set to 1 and
    their stride set to 0. Other dimensions will retain their original shape
    and their original effective stride.

    This transformation primarily affects the last view of the ShapeTracker,
    modifying its shape and strides while preserving its offset and mask.
    The resulting ShapeTracker (with the potentially new list of views)
    is then simplified.

    This can be useful, for example, to get a representation of a tensor
    where broadcasted dimensions (which have stride 0) are explicitly set to shape 1.

    Returns:
        A new ShapeTracker instance reflecting these changes.
    """
    if not self.views:
      # This state should ideally not be reached if ShapeTracker instances are
      # always created with at least one view (e.g., via ShapeTracker.from_shape).
      raise AssertionError("ShapeTracker must have at least one view to perform this operation.")

    current_shape = self.shape  # Effective shape from self.views[-1].shape

    # Retrieve the real (effective) strides of the current ShapeTracker.
    # We use ignore_valid=True to get the underlying geometric stride values,
    # as the check is for a numerical stride of 0, irrespective of masking.
    original_real_strides = self.real_strides(ignore_valid=True)

    if not current_shape:  # Handles the scalar case where shape is ()
      return self  # No dimensions to modify, return the original ShapeTracker

    new_shape_parts: list[sint] = list(current_shape)
    # Initialize with None, as strides can be None if they are complex/unknown
    new_strides_parts: list[Optional[sint]] = [None] * len(current_shape)

    for i in range(len(current_shape)):
      stride_i = original_real_strides[i]

      is_deterministically_zero = False
      if stride_i is not None:  # The stride must be a known value (not None)
        if isinstance(stride_i, int):
          is_deterministically_zero = stride_i == 0
        elif isinstance(stride_i, Variable):
          # A Variable is considered deterministically zero if its symbolic range is precisely [0,0].
          is_deterministically_zero = stride_i.min == 0 and stride_i.max == 0
        # Note: If stride_i could be other symbolic types, this check might need to be extended.

      if is_deterministically_zero:
        new_shape_parts[i] = 1  # Set shape of this dimension to 1
        new_strides_parts[i] = 0  # Stride for a unit dimension is conventionally 0
      else:
        new_shape_parts[i] = current_shape[i]  # Keep the original shape component for this dimension
        new_strides_parts[i] = stride_i  # Keep the original real stride (can be int, Variable, or None)

    final_new_shape: tuple[sint, ...] = tuple(new_shape_parts)
    # The elements of new_strides_parts are Optional[sint], matching the expected tuple type.
    final_new_strides: tuple[Optional[sint], ...] = tuple(new_strides_parts)

    original_last_view = self.views[-1]

    # Create a new View object that will replace the last view.
    # This new view uses the calculated new_shape and new_strides,
    # but importantly preserves the offset and mask from the original last view.
    modified_last_view = View.create(shape=final_new_shape, strides=final_new_strides, offset=original_last_view.offset, mask=original_last_view.mask)

    # Construct the tuple of views for the new ShapeTracker.
    if len(self.views) == 1:
      # If the original ShapeTracker had only one view, the new one will also have one (the modified view).
      new_views_tuple = (modified_last_view,)
    else:
      # If there were multiple views, keep all views except the last one, and append the modified_last_view.
      new_views_tuple = self.views[:-1] + (modified_last_view,)

    # Create the new ShapeTracker with the updated list of views and simplify it.
    # Simplification might merge the modified_last_view with the preceding view if possible.
    return ShapeTracker(new_views_tuple).simplify()

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

mops: dict[Ops, Callable] = {Ops.RESHAPE: ShapeTracker.reshape, Ops.PERMUTE: ShapeTracker.permute, Ops.EXPAND: ShapeTracker.expand,
                             Ops.SHRINK: ShapeTracker.shrink, Ops.FLIP: ShapeTracker.flip, Ops.PAD: ShapeTracker.pad}
